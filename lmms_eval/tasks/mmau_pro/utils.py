"""lmms-eval adapter for the official three-part MMAU-Pro evaluation.

The generation model produces ``model_output`` inside lmms-eval.  Per-sample
processing only records the fields required by the official evaluators.  The
expensive Qwen judge and NV-Embed models are loaded later, one at a time, in
short-lived worker processes during metric aggregation. Before those workers
start, the generated answers are saved and lmms-eval's existing model cleanup
is confirmed. Process exit fully releases one evaluator before the next.
"""

import gc
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from statistics import pstdev

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmau.utils import doc_to_audio as mmau_doc_to_audio
from lmms_eval.tasks.mmau.utils import doc_to_text as mmau_doc_to_text


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_AUDIO_CACHE_DIR = Path("/tmp/lmms_eval_audio_cache/mmau_pro")
DEFAULT_AUDIO_CACHE_SUBDIR = "_audio_cache"

# Evaluator defaults come from the official MMAU-Pro comprehensive evaluator.
_EVALUATOR_CONFIG = {
    "qwen_judge_model": "Qwen/Qwen2.5-7B-Instruct",
    "nvembed_model": "nvidia/NV-Embed-v2",
    "evaluator_local_files_only": False,
    "evaluator_device": "cuda",
    "evaluator_dtype": "auto",
    "nvembed_device_map": None,
    "judge_dtype": "bfloat16",
    "judge_max_new_tokens": 512,
    "judge_temperature": 0.1,
}


# ---------------------------------------------------------------------------
# Dataset and prompt adapters，this steps is to deal with the data and put into the evaluator to evaluate
# ---------------------------------------------------------------------------


def _configure_evaluators(lmms_eval_specific_kwargs=None):
    """Capture YAML/env evaluator settings for the later aggregation callback.

    lmms-eval does not pass ``lmms_eval_specific_kwargs`` directly to the
    aggregation function.  Therefore the evaluator config is captured when
    doc_to_text/doc_to_audio runs, and environment variables are checked again
    inside model-loading functions as a cache-safe fallback.
    """
    raw_kwargs = lmms_eval_specific_kwargs or {}
    kwargs = {}
    if isinstance(raw_kwargs, dict):
        # Accept both the raw YAML shape {"default": {...}} and lmms-eval's
        # flattened shape where default keys have already been merged.
        kwargs.update(raw_kwargs.get("default", {}))
        kwargs.update({key: value for key, value in raw_kwargs.items() if key != "default"})
    for key in _EVALUATOR_CONFIG:
        if key in kwargs:
            _EVALUATOR_CONFIG[key] = kwargs[key]

    env_overrides = {
        "qwen_judge_model": os.getenv("MMAU_PRO_QWEN_JUDGE_MODEL"),
        "nvembed_model": os.getenv("MMAU_PRO_NVEMBED_MODEL"),
        "evaluator_local_files_only": os.getenv("MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY"),
        "evaluator_device": os.getenv("MMAU_PRO_EVALUATOR_DEVICE"),
        "evaluator_dtype": os.getenv("MMAU_PRO_EVALUATOR_DTYPE"),
        "nvembed_device_map": os.getenv("MMAU_PRO_NVEMBED_DEVICE_MAP"),
        "judge_dtype": os.getenv("MMAU_PRO_JUDGE_DTYPE"),
        "judge_max_new_tokens": os.getenv("MMAU_PRO_JUDGE_MAX_NEW_TOKENS"),
        "judge_temperature": os.getenv("MMAU_PRO_JUDGE_TEMPERATURE"),
    }
    for key, value in env_overrides.items():
        if value not in (None, ""):
            _EVALUATOR_CONFIG[key] = value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, (str, bytes, dict)) and hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _clean_text(value):
    return "" if value is None else str(value).strip()


def _choices(doc):
    """MMAU-Pro stores choices as list[str]; also accept MMAU JSON strings."""
    raw_choices = doc.get("choices") or []
    if isinstance(raw_choices, str):
        try:
            raw_choices = json.loads(raw_choices)
        except json.JSONDecodeError:
            raw_choices = [raw_choices]
    return [_clean_text(choice) for choice in _as_list(raw_choices) if _clean_text(choice)]


def _dataset_root(lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    dataset_path = kwargs.get("dataset_path")
    if not dataset_path:
        return None
    path = Path(dataset_path).expanduser()
    return path if path.is_absolute() or path.exists() else None


def _resolve_audio_path(path_value, dataset_root):
    path = Path(_clean_text(path_value)).expanduser()
    if not path.is_absolute() and dataset_root is not None:
        path = dataset_root / path
    return str(path)


def mmau_pro_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    """Return one simple-bridge audio item per MMAU-Pro audio path.

    MMAU-Pro normally contains persistent ``audio_path: list[str]`` values, so
    those files can be passed directly to qwen-omni-utils.  If a converted
    dataset exposes a HuggingFace Audio dict instead, delegate to MMAU's
    existing WAV cache implementation.
    """
    _configure_evaluators(lmms_eval_specific_kwargs)
    audio_values = doc.get("audio_path") or doc.get("audio_paths")

    if not audio_values:
        audio = doc.get("audio")
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            # Reuse MMAU's WAV writer while giving MMAU-Pro its own cache root.
            audio_kwargs = dict(lmms_eval_specific_kwargs or {})
            audio_kwargs.setdefault("audio_cache_subdir", DEFAULT_AUDIO_CACHE_SUBDIR)
            if not audio_kwargs.get("dataset_path") and not audio_kwargs.get("audio_cache_dir"):
                audio_kwargs["audio_cache_dir"] = str(DEFAULT_AUDIO_CACHE_DIR)
            cached_items = mmau_doc_to_audio(doc, audio_kwargs)
            return [
                {
                    "type": "audio",
                    "audio": item.get("audio") or item.get("url"),
                }
                for item in cached_items
            ]
        audio_values = audio

    dataset_root = _dataset_root(lmms_eval_specific_kwargs)
    contents = []
    for audio_value in _as_list(audio_values):
        if isinstance(audio_value, dict):
            audio_value = audio_value.get("path") or audio_value.get("url")
        if audio_value:
            contents.append(
                {
                    "type": "audio",
                    "audio": _resolve_audio_path(audio_value, dataset_root),
                }
            )

    if not contents:
        eval_logger.warning(f"No audio found for MMAU-Pro sample {doc.get('id', 'unknown')}")
    return contents


def mmau_pro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build MCQ, open-ended, or instruction-following generation prompts."""
    _configure_evaluators(lmms_eval_specific_kwargs)
    kwargs = lmms_eval_specific_kwargs or {}
    choices = _choices(doc)
    category = _clean_text(doc.get("category")).lower()

    # Reuse MMAU's established A/B/C/D prompt formatter for normal MCQs.
    if choices and category not in {"open", "instruction following"} and len(choices) <= 10:
        mmau_doc = {
            **doc,
            "question": _clean_text(doc.get("question")),
            "choices": json.dumps(choices, ensure_ascii=False),
        }
        return mmau_doc_to_text(
            mmau_doc,
            {
                "pre_prompt": kwargs.get("pre_prompt", ""),
                "post_prompt": kwargs.get(
                    "post_prompt",
                    "\nAnswer with the option's letter from the given choices directly.",
                ),
            },
        )

    # MMAU currently formats up to ten choices. Preserve the same textual
    # contract for the unlikely case that MMAU-Pro contains more.
    if choices and category not in {"open", "instruction following"}:
        choice_text = "\n".join(
            f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices[: len(LETTERS)])
        )
        return (
            f"{kwargs.get('pre_prompt', '')}{_clean_text(doc.get('question'))}\n"
            f"{choice_text}{kwargs.get('post_prompt', '')}"
        )

    pre_prompt = kwargs.get("pre_prompt", "")
    if category == "instruction following":
        post_prompt = kwargs.get("instruction_post_prompt", "")
    else:
        post_prompt = kwargs.get("open_post_prompt", "\nAnswer the question directly.")
    return f"{pre_prompt}{_clean_text(doc.get('question'))}{post_prompt}"


def mmau_pro_doc_to_choice(doc):
    return _choices(doc)


def _to_builtin(value):
    """Make parquet/numpy values JSON-safe for the aggregate result report."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _to_builtin(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def mmau_pro_process_results(doc, results):
    """Preserve all fields needed by the three official aggregators.

    No evaluator model is loaded here.  Loading a 7B judge once per sample
    would be prohibitively expensive and would overlap with the generation
    model's GPU allocation.
    """
    payload = {
        "id": _clean_text(doc.get("id")),
        "category": _clean_text(doc.get("category")),
        "question": _clean_text(doc.get("question")),
        "answer": _clean_text(doc.get("answer")),
        "choices": _choices(doc),
        "task_identifier": _clean_text(doc.get("task_identifier")),
        "kwargs": _to_builtin(doc.get("kwargs") or {}),
        "model_output": _clean_text(results[0] if results else ""),
    }
    return {"mmau_pro_comprehensive": payload}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clear_device_cache():
    """Release Python garbage and return unused CUDA allocations to PyTorch."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                with torch.cuda.device(device_index):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except RuntimeError:
                        pass
    except ImportError:
        pass


def _find_lmms_eval_model():
    """Find the generation wrapper owned by lmms_eval.evaluator.evaluate."""
    frame = inspect.currentframe()
    found = None
    try:
        frame = frame.f_back
        while frame is not None:
            if (
                frame.f_globals.get("__name__") == "lmms_eval.evaluator"
                and frame.f_code.co_name == "evaluate"
            ):
                found = frame.f_locals.get("lm")
                break
            frame = frame.f_back
    finally:
        # Frame objects form reference cycles; never retain one after lookup.
        del frame
    return found


def _cuda_module_devices(module):
    devices = set()
    for values in (module.parameters(), module.buffers()):
        for value in values:
            if value.device.type == "cuda":
                devices.add(value.device)
    return devices


def _cuda_memory_snapshot(torch):
    return {
        index: {
            "allocated_gib": torch.cuda.memory_allocated(index) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(index) / (1024**3),
        }
        for index in range(torch.cuda.device_count())
    }


def _detach_mllm_comp_runtime(lm, torch):
    """Drop MLLM-Comp's nested runtime, which lmms-eval clean() cannot see."""
    bridge = vars(lm).pop("_platform_bridge", None)
    if bridge is None:
        return False

    runtime = getattr(bridge, "runtime", None)
    model = getattr(runtime, "model", None)
    if isinstance(model, torch.nn.Module):
        devices = _cuda_module_devices(model)
        if devices:
            eval_logger.info(
                "MMAU-Pro releasing nested MLLM-Comp generation model on "
                f"{', '.join(str(device) for device in sorted(devices, key=str))}"
            )

    # BackboneRuntime and LMMSEvalRequestBridge are ordinary Python objects,
    # not nn.Modules. lmms-eval's base clean() only removes direct module
    # attributes, so explicitly break the platform-owned strong references.
    if runtime is not None:
        runtime.model = None
        runtime.processor = None
    bridge.runtime = None
    bridge.facade = None
    del model, runtime, bridge
    return True


def _release_generation_model():
    """Ensure lmms-eval's completed generation model no longer owns VRAM.

    The aggregation API does not receive the model directly, but it runs below
    ``lmms_eval.evaluator.evaluate`` while that function still owns ``lm``.
    lmms-eval normally calls ``lm.clean()`` before post-processing. Calling the
    same public cleanup hook here is intentionally idempotent: it also covers
    versions where cleanup happens later, without copying the large Qwen2.5-
    Omni model to host RAM. Restricting lookup to that exact module/function
    pair avoids guessing among arbitrary live torch modules.
    """
    import torch

    lm = _find_lmms_eval_model()
    if lm is None:
        raise RuntimeError(
            "MMAU-Pro could not access lmms-eval's generation model; refusing "
            "to load GPU evaluators while the upstream model may still occupy VRAM"
        )

    clean = getattr(lm, "clean", None)
    if not callable(clean):
        raise RuntimeError(
            f"MMAU-Pro cannot release generation wrapper {type(lm).__name__}: "
            "the wrapper has no clean() method"
        )

    before = _cuda_memory_snapshot(torch)
    try:
        clean()
    except Exception as error:
        raise RuntimeError(
            "MMAU-Pro failed to release the generation model before loading "
            f"GPU evaluators: {error}"
        ) from error
    released_platform_runtime = _detach_mllm_comp_runtime(lm, torch)
    _clear_device_cache()

    remaining_modules = []
    for name, value in vars(lm).items():
        if isinstance(value, torch.nn.Module):
            devices = _cuda_module_devices(value)
            if devices:
                remaining_modules.append(
                    f"{name} ({', '.join(str(device) for device in sorted(devices, key=str))})"
                )
    if remaining_modules:
        raise RuntimeError(
            "MMAU-Pro generation wrapper still owns CUDA modules after clean(): "
            f"{', '.join(remaining_modules)}; GPU evaluator loading was cancelled"
        )

    after = _cuda_memory_snapshot(torch)
    eval_logger.info(
        f"MMAU-Pro released generation model {type(lm).__name__} before judging; "
        f"platform_runtime={released_platform_runtime}, "
        f"CUDA memory before={before}, after={after}"
    )


def _evaluator_load_kwargs(dtype_config_key="evaluator_dtype"):
    """Return deterministic placement settings for evaluator models.

    The generation model is released before evaluator workers start. Each
    evaluator can therefore use the configured GPU, and worker process exit
    guarantees that its VRAM is returned before the next evaluator loads.
    """
    import torch

    raw_device = _clean_text(_EVALUATOR_CONFIG.get("evaluator_device") or "cuda").lower()
    if raw_device == "auto":
        device = "auto"
        device_map = "auto"
    else:
        try:
            parsed_device = torch.device(raw_device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid MMAU-Pro evaluator_device: {raw_device!r}") from error
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"MMAU-Pro evaluator_device={raw_device!r} requires CUDA, but CUDA is unavailable"
            )
        device = str(parsed_device)
        device_map = {"": device}

    raw_dtype = _clean_text(_EVALUATOR_CONFIG.get(dtype_config_key) or "auto").lower()
    dtype_aliases = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if raw_dtype not in dtype_aliases:
        supported = ", ".join(dtype_aliases)
        raise ValueError(
            f"Invalid MMAU-Pro evaluator_dtype: {raw_dtype!r}; expected one of {supported}"
        )

    return device, {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype_aliases[raw_dtype],
    }


def _model_input_device(model):
    """Locate the device that should receive tokenized model inputs."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None:
            return weight.device
    return next(model.parameters()).device


def _run_isolated_evaluator(evaluator, items):
    """Run one model-backed evaluator in a fresh process.

    The upstream generation model has already been released. A fresh
    interpreter lets the configured GPU run exactly one evaluator and returns
    all of that evaluator's CUDA allocations when the worker exits.
    """
    if not items:
        return []
    if evaluator not in {"qwen_judge", "nvembed"}:
        raise ValueError(f"Unknown MMAU-Pro evaluator worker: {evaluator!r}")

    _configure_evaluators()
    worker_path = Path(__file__).with_name("evaluator_worker.py")
    if not worker_path.is_file():
        raise FileNotFoundError(f"MMAU-Pro evaluator worker is missing: {worker_path}")

    raw_device = _clean_text(_EVALUATOR_CONFIG.get("evaluator_device") or "cuda").lower()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    if raw_device == "cpu" or raw_device.startswith("cpu:"):
        # device_map={"": "cpu"} is normally enough, but trust_remote_code
        # checkpoints may contain their own placement logic. CUDA visibility
        # is the enforceable boundary for a CPU-only evaluator worker.
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["NVIDIA_VISIBLE_DEVICES"] = "void"

    payload = {
        "config": _to_builtin(dict(_EVALUATOR_CONFIG)),
        "items": _to_builtin(items),
    }
    _clear_device_cache()
    with tempfile.TemporaryDirectory(prefix=f"mmau_pro_{evaluator}_") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        with input_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

        eval_logger.info(
            f"Starting isolated MMAU-Pro {evaluator} worker: "
            f"device={raw_device}, samples={len(items)}"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    evaluator,
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                env=child_env,
                cwd=str(worker_path.parent),
            )
        finally:
            # Evaluator allocations disappear completely when its process
            # exits. This also releases any unused parent-side CUDA cache.
            _clear_device_cache()

        if not output_path.is_file():
            raise RuntimeError(f"MMAU-Pro {evaluator} worker produced no output")
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if not isinstance(records, list) or len(records) != len(items):
        record_count = len(records) if isinstance(records, list) else "not-a-list"
        raise RuntimeError(
            f"MMAU-Pro {evaluator} worker returned {record_count} records "
            f"for {len(items)} samples"
        )
    return records


# ---------------------------------------------------------------------------
# Official evaluator 1: Audio Instruction Following rule checks
# ---------------------------------------------------------------------------


def _count_sentences(text):
    """Use the official NLTK tokenizer, with an offline-safe fallback."""
    try:
        from nltk.tokenize import sent_tokenize

        return len(sent_tokenize(text))
    except (ImportError, LookupError):
        # Avoid downloading NLTK data during an evaluation job.
        return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def _count_keyword_frequency(text, keyword):
    return len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text.lower()))


def _alpha_normalize(text, keep_spaces=False):
    pattern = r"[^a-zA-Z ]" if keep_spaces else r"[^a-zA-Z]"
    return re.sub(pattern, "", text).lower()


def _evaluate_aif_sample(response, sample):
    """Reproduce the official task_identifier/kwargs constraint evaluator."""
    identifier = _clean_text(sample.get("task_identifier"))
    raw_kwargs = sample.get("kwargs") or {}
    if isinstance(raw_kwargs, str):
        try:
            raw_kwargs = json.loads(raw_kwargs)
        except json.JSONDecodeError:
            raw_kwargs = {}
    kwargs = _to_builtin(raw_kwargs)

    if identifier == "Include Keywords":
        keywords = _clean_text(kwargs.get("keywords")).split(", ")
        return all(keyword.lower() in response.lower() for keyword in keywords)
    if identifier == "Keyword Frequency":
        return _count_keyword_frequency(response, _clean_text(kwargs.get("keyword"))) == int(kwargs.get("N", 0))
    if identifier == "Forbidden Words":
        words = _clean_text(kwargs.get("forbidden_words")).split(", ")
        return not any(word.lower() in response.lower() for word in words)
    if identifier == "Number Paragraphs":
        return len([part for part in response.split("***") if part.strip()]) == int(kwargs.get("N", 0))
    if identifier == "Number Words (at least)":
        return len(response.split()) >= int(kwargs.get("N", 0))
    if identifier == "Number Words (at most)":
        return len(response.split()) <= int(kwargs.get("N", 0))
    if identifier == "Number Words (range)":
        return int(kwargs.get("N1", 0)) <= len(response.split()) <= int(kwargs.get("N2", 999))
    if identifier == "Number Sentences (at least)":
        return _count_sentences(response) >= int(kwargs.get("N", 0))
    if identifier == "Number Sentences (at most)":
        return _count_sentences(response) <= int(kwargs.get("N", 0))
    if identifier == "Number Sentences (range)":
        return int(kwargs.get("N1", 0)) <= _count_sentences(response) <= int(kwargs.get("N2", 999))
    if identifier == "Postscript":
        return _alpha_normalize(_clean_text(kwargs.get("postscript_marker"))) in _alpha_normalize(response)
    if identifier == "Number Placeholder":
        return len(re.findall(r"\[[^\]]+\]", response)) >= int(kwargs.get("N", 0))
    if identifier == "Number Bullets":
        return len(re.findall(r"(?:^|\n)\s*\*\s+", response)) == int(kwargs.get("N", 0))
    if identifier == "Title":
        return bool(re.search(r"<<[^>]+>>", response))
    if identifier == "Minimum Number Highlighted Section":
        return len(re.findall(r"\*([^*]+)\*", response)) >= int(kwargs.get("N", 0))
    if identifier == "Multiple Sections":
        splitter = re.escape(_clean_text(kwargs.get("section_splitter")))
        sections = [part for part in re.split(rf"\s*{splitter}\s*", response.strip()) if part.strip()]
        return len(sections) == int(kwargs.get("N", 0))
    if identifier == "Repeat Prompt":
        prompt = _clean_text(sample.get("question"))
        return response.strip().lower().startswith(prompt.lower())
    if identifier == "Two Responses":
        parts = response.split("******")
        return len(parts) == 2 and parts[0].strip().lower() != parts[1].strip().lower()
    if identifier == "All Uppercase":
        return response.isupper()
    if identifier == "All Lowercase":
        return response.islower()
    if identifier.startswith("All-capital Words"):
        count = sum(1 for word in response.split() if word.isupper())
        if identifier.endswith("(at least)"):
            return count >= int(kwargs.get("N", 0))
        if identifier.endswith("(at most)"):
            return count <= int(kwargs.get("N", 0))
        return int(kwargs.get("N1", 0)) <= count <= int(kwargs.get("N2", 999))
    if identifier == "Start Checker":
        return _alpha_normalize(response, keep_spaces=True).startswith(
            _alpha_normalize(_clean_text(kwargs.get("start_phrase")), keep_spaces=True)
        )
    if identifier == "End Checker":
        return _alpha_normalize(response, keep_spaces=True).endswith(
            _alpha_normalize(_clean_text(kwargs.get("end_phrase")), keep_spaces=True)
        )
    if identifier == "Quotation":
        stripped = response.strip()
        return stripped.startswith('"') and stripped.endswith('"')
    if identifier == "No Commas":
        return "," not in response

    eval_logger.warning(f"Unsupported MMAU-Pro instruction task_identifier: {identifier!r}")
    return False


# ---------------------------------------------------------------------------
# Official evaluator 2: Qwen2.5-7B-Instruct judge for open questions
# ---------------------------------------------------------------------------


def _create_judge_prompt(item):
    return f"""You are an expert evaluator for general open-ended question answering tasks. Please evaluate the quality of a model's response to a question.

Question: {item["question"]}

Reference Answer: {item["answer"]}

Model Response: {item["model_output"]}

Please evaluate the model response on the following criteria and provide scores from 1-5 (where 5 is best):

1. **Correctness**: How factually accurate is the response compared to the reference?
2. **Relevance**: How well does the response address the specific question asked?
3. **Completeness**: Does the response cover all important aspects mentioned in the reference?
4. **Clarity**: How clear and well-structured is the response?

For each criterion, provide:
- A score from 1-5
- A brief justification (1-2 sentences)

Format your response as:

CORRECTNESS: [score] - [justification]
RELEVANCE: [score] - [justification]{" "}
COMPLETENESS: [score] - [justification]
CLARITY: [score] - [justification]
OVERALL: [average score] - [overall assessment]"""


def _extract_judge_scores(evaluation_text):
    patterns = {
        "correctness": r"CORRECTNESS:\s*(\d+)",
        "relevance": r"RELEVANCE:\s*(\d+)",
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "clarity": r"CLARITY:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+(?:\.\d+)?)",
    }
    scores = {}
    for criterion, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
        else:
            # Match the official evaluator's neutral fallback for malformed text.
            scores[criterion] = 3.0
    # Preserve the official script's behavior: an absent OVERALL and an
    # explicitly parsed neutral OVERALL of 3.0 are both replaced by the mean
    # of the four criterion scores.
    if "overall" not in scores or scores["overall"] == 3.0:
        scores["overall"] = sum(scores[key] for key in ("correctness", "relevance", "completeness", "clarity")) / 4
    return scores


def _load_qwen_judge():
    _configure_evaluators()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = _EVALUATOR_CONFIG["qwen_judge_model"]
    local_only = _as_bool(_EVALUATOR_CONFIG.get("evaluator_local_files_only"))
    eval_logger.info(f"Loading MMAU-Pro Qwen judge: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    device, load_kwargs = _evaluator_load_kwargs("judge_dtype")
    eval_logger.info(
        f"MMAU-Pro Qwen judge placement: device={device}, "
        f"dtype={_EVALUATOR_CONFIG['judge_dtype']}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_only,
        trust_remote_code=True,
        **load_kwargs,
    ).eval()
    return model, tokenizer


def _judge_open_items(items):
    temperature = float(_EVALUATOR_CONFIG.get("judge_temperature", 0.1))
    max_new_tokens = int(_EVALUATOR_CONFIG.get("judge_max_new_tokens", 512))
    records = []
    import torch

    model, tokenizer = _load_qwen_judge()
    try:
        for item in items:
            prompt = _create_judge_prompt(item)
            messages = [
                {"role": "system", "content": "You are a helpful and objective evaluator."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(
                _model_input_device(model)
            )
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            with torch.inference_mode():
                generated = model.generate(**model_inputs, **generation_kwargs)
            continuation = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated)
            ]
            evaluation_text = tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
            records.append(
                {
                    "id": item["id"],
                    "scores": _extract_judge_scores(evaluation_text),
                    "evaluation": evaluation_text,
                }
            )
            del generated, continuation, model_inputs
    finally:
        del model
        del tokenizer
        _clear_device_cache()

    return records


# ---------------------------------------------------------------------------
# Official evaluator 3: NV-Embed semantic mapping for closed questions
# ---------------------------------------------------------------------------


def _nvembed_bidirectional_mistral_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """Forward compatible with newer Transformers MistralDecoderLayer.

    NV-Embed-v2 ships a custom BidirectionalMistralModel copied from an older
    Transformers Mistral implementation.  Newer Transformers expects
    ``position_embeddings=(cos, sin)`` to be passed into every decoder layer;
    the old NV-Embed forward only passes ``position_ids``.  Without this patch,
    MistralAttention fails with:

        TypeError: cannot unpack non-iterable NoneType object

    This implementation keeps NV-Embed's bidirectional attention-mask behavior
    while following the current Mistral model contract for rotary embeddings,
    cache objects, cache positions, and decoder-layer return values.
    """
    import torch
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_attention_mask,
        _prepare_4d_attention_mask_for_sdpa,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        eval_logger.warning("Disabling NV-Embed cache because gradient checkpointing is active")
        use_cache = False

    past_key_values_length = 0
    use_legacy_cache = False
    if use_cache:
        use_legacy_cache = past_key_values is not None and not isinstance(past_key_values, Cache)
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        elif use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if hasattr(past_key_values, "get_seq_length"):
            past_key_values_length = past_key_values.get_seq_length()
        elif hasattr(past_key_values, "get_usable_length"):
            past_key_values_length = past_key_values.get_usable_length(seq_length)

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            device=device,
        )
    else:
        cache_position = cache_position.to(device=device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    else:
        position_ids = position_ids.to(device=device).view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    attn_implementation = getattr(self, "_attn_implementation", getattr(self.config, "_attn_implementation", "eager"))
    if attention_mask is not None:
        if attn_implementation == "flash_attention_2":
            # Flash Attention consumes the original 2D padding mask.  Passing
            # None for an all-ones mask preserves fully bidirectional attention.
            attention_mask = attention_mask if torch.any(attention_mask == 0) else None
        elif attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask,
                inputs_embeds.dtype,
                tgt_len=seq_length,
            )
        else:
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                inputs_embeds.dtype,
                tgt_len=seq_length,
            )

    hidden_states = inputs_embeds

    # Newer Transformers Mistral layers require rotary embeddings to be
    # computed once by the model and passed down as (cos, sin).
    if not hasattr(self, "rotary_emb"):
        raise AttributeError("NV-Embed BidirectionalMistralModel has no rotary_emb module")
    try:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
    except TypeError:
        # Retain support for the immediately preceding Transformers RoPE API.
        position_embeddings = self.rotary_emb(
            hidden_states,
            seq_len=seq_length + past_key_values_length,
        )

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_kwargs = dict(kwargs)
        layer_kwargs.update({
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        })
        layer_signature = inspect.signature(decoder_layer.forward).parameters
        # Transformers 4.57 renamed the argument and updates Cache instances
        # in place.  Older transitional layers still use the singular name.
        if "past_key_values" in layer_signature:
            layer_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in layer_signature:
            layer_kwargs["past_key_value"] = past_key_values
        if "position_embeddings" in layer_signature:
            layer_kwargs["position_embeddings"] = position_embeddings
        if "cache_position" in layer_signature:
            layer_kwargs["cache_position"] = cache_position

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                **layer_kwargs,
            )
        else:
            layer_outputs = decoder_layer(hidden_states, **layer_kwargs)

        # Since Transformers 4.54, MistralDecoderLayer returns hidden_states as
        # a Tensor.  Indexing [0] here silently removes the batch dimension and
        # makes the next layer interpret sequence length as an attention axis.
        if torch.is_tensor(layer_outputs):
            hidden_states = layer_outputs
        else:
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None
    if use_cache and use_legacy_cache and hasattr(next_cache, "to_legacy_cache"):
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(value for value in [hidden_states, next_cache, all_hidden_states, all_self_attns] if value is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _patch_nvembed_for_current_transformers(model):
    embedding_model = getattr(model, "embedding_model", None)
    if embedding_model is None:
        return
    if embedding_model.__class__.__name__ != "BidirectionalMistralModel":
        return

    # Patch only when the installed decoder layer exposes the newer
    # position_embeddings argument.  Older compatible environments can keep the
    # model's original remote-code forward unchanged.
    first_layer = embedding_model.layers[0] if getattr(embedding_model, "layers", None) else None
    if first_layer is None:
        return
    if "position_embeddings" not in inspect.signature(first_layer.forward).parameters:
        return

    embedding_model.forward = types.MethodType(_nvembed_bidirectional_mistral_forward, embedding_model)
    eval_logger.info("Patched NV-Embed BidirectionalMistralModel.forward for current Transformers Mistral API")


def _load_nvembed():
    _configure_evaluators()

    from transformers import AutoConfig, AutoModel

    model_name = _EVALUATOR_CONFIG["nvembed_model"]
    local_only = _as_bool(_EVALUATOR_CONFIG.get("evaluator_local_files_only"))
    device, load_kwargs = _evaluator_load_kwargs()
    configured_device_map = _clean_text(
        _EVALUATOR_CONFIG.get("nvembed_device_map")
    )
    if configured_device_map:
        load_kwargs["device_map"] = configured_device_map
    eval_logger.info(
        f"Loading MMAU-Pro NV-Embed evaluator: {model_name}; "
        f"device={device}, device_map={load_kwargs['device_map']}, "
        f"dtype={_EVALUATOR_CONFIG['evaluator_dtype']}"
    )
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_only,
        trust_remote_code=True,
    )

    # NV-Embed-v2's custom modeling code initializes its tokenizer from
    # config.text_config._name_or_path.  In the downloaded config this can still
    # be a HuggingFace repo id, causing an offline run to contact hf.co even
    # when the top-level model path is local.  For local evaluator directories,
    # force that nested tokenizer/config path back to the same local directory.
    local_model_path = Path(str(model_name)).expanduser()
    if local_model_path.exists() and getattr(config, "text_config", None) is not None:
        local_model_path = local_model_path.resolve()
        config.text_config._name_or_path = str(local_model_path)
        if hasattr(config.text_config, "name_or_path"):
            config.text_config.name_or_path = str(local_model_path)
        eval_logger.info(f"Using local NV-Embed text config/tokenizer path: {local_model_path}")

    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        local_files_only=local_only,
        trust_remote_code=True,
        **load_kwargs,
    ).eval()
    eval_logger.info(
        f"MMAU-Pro NV-Embed actual hf_device_map: "
        f"{getattr(model, 'hf_device_map', None)}"
    )
    _patch_nvembed_for_current_transformers(model)
    return model


def _match_closed_items(items):
    records = []
    import torch
    import torch.nn.functional as functional

    model = _load_nvembed()
    try:
        for item in items:
            with torch.inference_mode():
                prediction_embedding = model.encode([item["model_output"]], instruction="", max_length=4096)
                choice_embeddings = model.encode(item["choices"], instruction="", max_length=4096)
                prediction_embedding = functional.normalize(prediction_embedding, p=2, dim=1)
                choice_embeddings = functional.normalize(choice_embeddings, p=2, dim=1)
                similarities = (prediction_embedding @ choice_embeddings.T).squeeze()
                best_index = int(torch.argmax(similarities).item())
                confidence = float(torch.max(similarities).item() * 100)

            matched_choice = item["choices"][best_index]
            records.append(
                {
                    "id": item["id"],
                    "matched_choice": matched_choice,
                    "confidence": confidence,
                    "correct": matched_choice == item["answer"],
                }
            )
            del prediction_embedding, choice_embeddings, similarities
    finally:
        del model
        _clear_device_cache()

    return records


# ---------------------------------------------------------------------------
# Comprehensive aggregation and report
# ---------------------------------------------------------------------------


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _classification_metrics(ground_truth, predictions):
    """Compute sklearn-compatible weighted precision/recall/F1."""
    if not ground_truth:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    labels = set(ground_truth) | set(predictions)
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total = len(ground_truth)
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(ground_truth, predictions)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(ground_truth, predictions)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(ground_truth, predictions)
        )
        support = sum(truth == label for truth in ground_truth)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weight = support / total
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight

    return {
        "accuracy": _mean(
            [truth == prediction for truth, prediction in zip(ground_truth, predictions)]
        ),
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1,
    }

#收集所有 payload，然后一次性聚合 ,分开进行evalute

def _build_summary(results, return_sample_results=False):
    open_items = [item for item in results if item["category"].lower() == "open"]
    aif_items = [item for item in results if item["category"].lower() == "instruction following"]
    closed_items = [
        item
        for item in results
        if item["category"].lower() not in {"open", "instruction following"} and len(item["choices"]) > 1
    ]

    category_results = {}
    sample_results = []

    open_records = _run_isolated_evaluator("qwen_judge", open_items)
    if open_records:
        criteria = ("correctness", "relevance", "completeness", "clarity", "overall")
        metrics = {
            f"avg_{criterion}": _mean([record["scores"][criterion] for record in open_records])
            for criterion in criteria
        }
        metrics.update(
            {
                f"std_{criterion}": pstdev(
                    [record["scores"][criterion] for record in open_records]
                )
                for criterion in criteria
            }
        )
        metrics["good_response_rate"] = _mean(
            [record["scores"]["overall"] >= 4.0 for record in open_records]
        )
        metrics["poor_response_rate"] = _mean(
            [record["scores"]["overall"] <= 2.0 for record in open_records]
        )
        category_results["open"] = {
            "type": "openended",
            "count": len(open_items),
            "performance_score": metrics["avg_overall"] / 5.0,
            "metrics": metrics,
        }
        for item, record in zip(open_items, open_records):
            sample_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "evaluator_type": "openended_qwen_judge",
                    "evaluation_status": "evaluated",
                    "question": item["question"],
                    "answer": item["answer"],
                    "model_output": item["model_output"],
                    "scores": record["scores"],
                    "performance_score": record["scores"]["overall"] / 5.0,
                    "judge_evaluation": record["evaluation"],
                }
            )

    if aif_items:
        successes = [_evaluate_aif_sample(item["model_output"], item) for item in aif_items]
        category_results["instruction following"] = {
            "type": "aif",
            "count": len(aif_items),
            "performance_score": _mean(successes),
            "success_rate": _mean(successes),
        }
        for item, success in zip(aif_items, successes):
            sample_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "evaluator_type": "aif_rule",
                    "evaluation_status": "evaluated",
                    "question": item["question"],
                    "model_output": item["model_output"],
                    "task_identifier": item["task_identifier"],
                    "kwargs": item["kwargs"],
                    "success": bool(success),
                    "performance_score": float(success),
                }
            )

    closed_records = _run_isolated_evaluator("nvembed", closed_items)
    closed_by_category = defaultdict(lambda: {"ground_truth": [], "predictions": []})
    for item, record in zip(closed_items, closed_records):
        closed_by_category[item["category"]]["ground_truth"].append(item["answer"])
        closed_by_category[item["category"]]["predictions"].append(record["matched_choice"])
        sample_results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "evaluator_type": "closed_nvembed",
                "evaluation_status": "evaluated",
                "question": item["question"],
                "answer": item["answer"],
                "choices": item["choices"],
                "model_output": item["model_output"],
                "matched_choice": record["matched_choice"],
                "confidence": record["confidence"],
                "correct": bool(record["correct"]),
                "performance_score": float(record["correct"]),
            }
        )
    for category, values in closed_by_category.items():
        metrics = _classification_metrics(values["ground_truth"], values["predictions"])
        category_results[category] = {
            "type": "closed",
            "count": len(values["ground_truth"]),
            "performance_score": metrics["accuracy"],
            "metrics": metrics,
        }

    evaluated_items = {id(item) for item in open_items + aif_items + closed_items}
    for item in results:
        if id(item) not in evaluated_items:
            sample_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "evaluator_type": "skipped",
                    "evaluation_status": "skipped",
                    "question": item["question"],
                    "answer": item["answer"],
                    "choices": item["choices"],
                    "model_output": item["model_output"],
                    "reason": "Sample is not open/instruction-following and has fewer than two choices.",
                    "performance_score": None,
                }
            )

    evaluated_samples = sum(result["count"] for result in category_results.values())
    weighted_total = sum(
        result["performance_score"] * result["count"] for result in category_results.values()
    )
    overall = weighted_total / evaluated_samples if evaluated_samples else 0.0
    summary = {
        "evaluation_summary": {
            "total_samples": len(results),
            "evaluated_samples": evaluated_samples,
            "per_sample_results": len(sample_results),
            "overall_weighted_performance": overall,
            "overall_weighted_performance_percent": overall * 100,
            "qwen_judge_model": _EVALUATOR_CONFIG["qwen_judge_model"],
            "nvembed_model": _EVALUATOR_CONFIG["nvembed_model"],
            "evaluator_device": _EVALUATOR_CONFIG["evaluator_device"],
            "evaluator_dtype": _EVALUATOR_CONFIG["evaluator_dtype"],
            "nvembed_device_map": _EVALUATOR_CONFIG["nvembed_device_map"],
            "judge_dtype": _EVALUATOR_CONFIG["judge_dtype"],
        },
        "category_results": category_results,
    }
    if return_sample_results:
        return summary, sample_results
    return summary


def _save_generation_results(results, args=None):
    """Persist upstream answers before evaluator workers start."""
    output_path = Path(generate_submission_file("mmau_pro_generation_results.jsonl", args))
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(_to_builtin(result), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _load_generation_results(output_path):
    """Reload the persisted handoff consumed by the evaluator workers."""
    results = []
    with Path(output_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(
                    "MMAU-Pro generation result must be a JSON object at "
                    f"{output_path}:{line_number}"
                )
            results.append(item)
    return results


def _requires_model_evaluator(results):
    for item in results:
        category = _clean_text(item.get("category")).lower()
        if category == "open":
            return True
        if category != "instruction following" and len(item.get("choices") or []) > 1:
            return True
    return False


# give the final results among all the samples
def mmau_pro_aggregate_results(results, args=None):
    """Run all three official evaluators and return the weighted percentage."""
    generation_output_path = _save_generation_results(results, args)
    eval_logger.info(
        f"MMAU-Pro saved {len(results)} generation results to {generation_output_path}"
    )
    if _requires_model_evaluator(results):
        _release_generation_model()

    # Treat the durable JSONL as the boundary between generation and judging.
    # Evaluators never depend on the generation wrapper or its live objects.
    persisted_results = _load_generation_results(generation_output_path)
    summary, sample_results = _build_summary(
        persisted_results,
        return_sample_results=True,
    )
    output_path = generate_submission_file("mmau_pro_comprehensive_results.json", args)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    sample_output_path = generate_submission_file("mmau_pro_per_sample_results.jsonl", args)
    with open(sample_output_path, "w", encoding="utf-8") as handle:
        for sample_result in sample_results:
            handle.write(json.dumps(sample_result, ensure_ascii=False) + "\n")

    eval_logger.info("=" * 60)
    eval_logger.info(
        "MMAU-Pro overall weighted performance: "
        f"{summary['evaluation_summary']['overall_weighted_performance_percent']:.5f}%"
    )
    for category, result in summary["category_results"].items():
        eval_logger.info(
            f"{category}: type={result['type']}, count={result['count']}, "
            f"score={result['performance_score'] * 100:.5f}%"
        )
    eval_logger.info(f"MMAU-Pro report saved to {output_path}")
    eval_logger.info(f"MMAU-Pro per-sample evaluator results saved to {sample_output_path}")
    eval_logger.info("=" * 60)
    return round(summary["evaluation_summary"]["overall_weighted_performance_percent"], 5)
