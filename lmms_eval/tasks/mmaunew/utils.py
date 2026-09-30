"""MMAU-v05.15.25 adapter using the official AudioMCQ prompt and scorer."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Audio, Features, Sequence, Value
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmau.utils import doc_to_audio


AUDIO_MCQ_SYSTEM_PROMPT = (
    "You are an audio understanding model that answers multiple choice "
    "questions based on audio content."
)
_ANSWER_TAG_RE = re.compile(
    r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)
_LABEL_RE = re.compile(r"^\s*[\(\[]?([A-Z])[\)\]\.:]?\s*$", re.IGNORECASE)
_WORD_TOKEN_RE = re.compile(r"\b\w+\b")
_MINI_ATTRIBUTE_FIELDS = (
    "id",
    "dataset",
    "task",
    "split",
    "category",
    "sub-category",
    "difficulty",
)
_DEFAULT_AUDIO_CACHE_DIR = Path("/tmp/lmms_eval_audio_cache/mmaunew")
_DEFAULT_AUDIO_CACHE_SUBDIR = "_audio_cache"

# The official test-mini Parquet was written with datasets 4.x and stores its
# choices column as the new ``List`` feature.  Supplying this equivalent 3.6
# schema to load_dataset prevents datasets 3.6 from parsing that metadata.
MMAUNEW_TEST_MINI_FEATURES = Features(
    {
        "context": Audio(decode=False),
        "instruction": Value("string"),
        "choices": Sequence(Value("string")),
        "answer": Value("string"),
        "other_attributes": Value("string"),
    }
)


def _format_audio_mcq_prompt(question: Any, choices: Sequence[Any]) -> str:
    question_text = str(question).strip()
    option_list = "[" + ", ".join(repr(str(choice).strip()) for choice in choices) + "]"
    return (
        f"{question_text} Please choose the answer from the following options: "
        f"{option_list}. Output the final answer in <answer> </answer>."
    )


def _build_audio_mcq_messages(
    audio_items: Sequence[Mapping[str, Any]], prompt: str
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": AUDIO_MCQ_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                *[dict(item) for item in audio_items],
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _extract_final_answer(raw_response: Any) -> str:
    text = "" if raw_response is None else str(raw_response)
    tagged = _ANSWER_TAG_RE.findall(text)
    return tagged[-1].strip() if tagged else text.strip()


def _official_string_match(
    answer: Any, prediction: Any, choices: Sequence[Any]
) -> bool:
    """Exact copy of the token-set rule in AudioMCQ ``evaluation.py``."""

    def tokenize(value: Any) -> set[str]:
        text = "" if value is None else str(value)
        return set(_WORD_TOKEN_RE.findall(text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(
        incorrect_tokens
    )


def _evaluate_official_choice(
    raw_response: Any, choices: Sequence[Any], ground_truth_text: Any
) -> dict[str, Any]:
    clean_choices = [str(choice).strip() for choice in choices]
    extracted = _extract_final_answer(raw_response)
    predicted_label = ""
    predicted_text = extracted

    label_match = _LABEL_RE.fullmatch(extracted)
    if label_match:
        choice_index = ord(label_match.group(1).upper()) - ord("A")
        if 0 <= choice_index < len(clean_choices):
            predicted_label = chr(ord("A") + choice_index)
            predicted_text = clean_choices[choice_index]
        else:
            predicted_text = ""

    ground_truth = str(ground_truth_text).strip()
    has_ground_truth = bool(ground_truth)
    return {
        "score": float(
            has_ground_truth
            and _official_string_match(ground_truth, raw_response, clean_choices)
        ),
        "match_method": (
            "official_token_set" if has_ground_truth else "ground_truth_unavailable"
        ),
        "has_ground_truth": has_ground_truth,
        "prediction_label": predicted_label,
        "prediction_text": predicted_text,
        "ground_truth_text": ground_truth,
        "extracted_answer": extracted,
        "raw_response": "" if raw_response is None else str(raw_response)[:500],
    }


def _parse_mini_attributes(value: Any) -> dict[str, str]:
    """Expand the JSON metadata stored by gamma-lab-umd/MMAU-test-mini."""

    if isinstance(value, Mapping):
        attributes = dict(value)
    else:
        try:
            attributes = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid MMAU test-mini other_attributes: {value!r}"
            ) from exc

    if not isinstance(attributes, Mapping):
        raise ValueError("MMAU test-mini other_attributes must decode to an object")
    return {
        field: "" if attributes.get(field) is None else str(attributes.get(field))
        for field in _MINI_ATTRIBUTE_FIELDS
    }


def mmaunew_process_docs(dataset):
    """Normalize the official test-mini Parquet to the regular MMAU schema."""

    column_names = set(dataset.column_names)
    if "context" not in column_names:
        return dataset

    rename_map = {}
    if "audio" not in column_names:
        rename_map["context"] = "audio"
    if "question" not in column_names:
        rename_map["instruction"] = "question"
    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    # Keep the embedded bytes undecoded on both datasets 3.6 and 4.x so this
    # task can materialize them without soundfile or torchcodec decoding.
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    if "other_attributes" in dataset.column_names:
        # Restrict mapping to the metadata column so the embedded Audio feature
        # is preserved without decoding all 1,000 clips during task setup.
        dataset = dataset.map(
            _parse_mini_attributes,
            input_columns=["other_attributes"],
            remove_columns=["other_attributes"],
            desc="Normalizing MMAU-v05.15.25 test-mini metadata",
        )
    return dataset


def _dataset_root(lmms_eval_specific_kwargs=None) -> Path | None:
    kwargs = lmms_eval_specific_kwargs or {}
    configured = kwargs.get("dataset_path")
    if not configured:
        return None
    return Path(str(configured)).expanduser()


def _mini_audio_cache_dir(lmms_eval_specific_kwargs=None) -> Path:
    kwargs = lmms_eval_specific_kwargs or {}
    configured = kwargs.get("audio_cache_dir")
    if configured:
        return Path(str(configured)).expanduser()

    root = _dataset_root(lmms_eval_specific_kwargs)
    if root is not None:
        subdir = kwargs.get("audio_cache_subdir", _DEFAULT_AUDIO_CACHE_SUBDIR)
        return root / "test-mini" / str(subdir)
    return _DEFAULT_AUDIO_CACHE_DIR


def _cache_embedded_audio_bytes(
    doc, audio: Mapping[str, Any], lmms_eval_specific_kwargs=None
) -> Path:
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        raise ValueError(f"MMAU sample {doc.get('id')} has empty embedded audio bytes")

    source_name = Path(
        str(audio.get("path") or f"{doc.get('id', 'sample')}.wav")
    ).name
    if not Path(source_name).suffix:
        source_name = f"{source_name}.wav"

    cache_dir = _mini_audio_cache_dir(lmms_eval_specific_kwargs)
    output_path = cache_dir / source_name
    if not output_path.exists() or output_path.stat().st_size == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(audio_bytes))
    return output_path.resolve()


def _full_test_audio_path(doc, lmms_eval_specific_kwargs=None) -> Path:
    audio_id = str(doc.get("audio_id", "")).strip()
    if not audio_id:
        raise ValueError(f"MMAU sample {doc.get('id')} has no audio or audio_id")

    audio_path = Path(audio_id).expanduser()
    if audio_path.is_absolute():
        candidates = [audio_path]
    else:
        root = _dataset_root(lmms_eval_specific_kwargs)
        if root is None:
            raise ValueError(
                "Relative MMAU audio_id requires dataset_path in "
                "lmms_eval_specific_kwargs"
            )
        candidates = [root / audio_path, root / "test" / audio_path]

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"MMAU audio for sample {doc.get('id')} was not found; checked: {checked}"
    )


def mmaunew_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    audio = doc.get("audio")
    if audio is not None:
        if isinstance(audio, Mapping):
            if "array" in audio and "sampling_rate" in audio:
                # Compatibility with datasets <=3.6 or already-decoded inputs.
                audio_kwargs = dict(lmms_eval_specific_kwargs or {})
                audio_kwargs.setdefault(
                    "audio_cache_dir",
                    str(_mini_audio_cache_dir(lmms_eval_specific_kwargs)),
                )
                return doc_to_audio(doc, audio_kwargs)
            if audio.get("bytes") is not None:
                return [
                    {
                        "type": "audio",
                        "url": str(
                            _cache_embedded_audio_bytes(
                                doc, audio, lmms_eval_specific_kwargs
                            )
                        ),
                    }
                ]
        raise ValueError(
            f"Unsupported MMAU audio value for sample {doc.get('id')}: "
            f"{type(audio).__name__}"
        )

    return [
        {
            "type": "audio",
            "url": str(_full_test_audio_path(doc, lmms_eval_specific_kwargs)),
        }
    ]


def _choices(doc):
    choices = doc.get("choices", [])
    if isinstance(choices, str):
        choices = json.loads(choices)
    clean_choices = [str(choice).strip() for choice in choices if str(choice).strip()]
    if not clean_choices:
        raise ValueError(f"MMAU sample {doc.get('id')} has no non-empty choices")
    return clean_choices


def _answer_text(doc):
    return str(doc.get("answer", "")).strip()


def mmaunew_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return _format_audio_mcq_prompt(doc.get("question", ""), _choices(doc))


def mmaunew_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = mmaunew_doc_to_text(doc, lmms_eval_specific_kwargs)
    audio_items = mmaunew_doc_to_audio(doc, lmms_eval_specific_kwargs)
    return _build_audio_mcq_messages(audio_items, prompt)


def mmaunew_doc_to_choice(doc):
    return _choices(doc)


def mmaunew_doc_to_target(doc):
    return _answer_text(doc)


def mmaunew_process_results(doc, results):
    raw_response = results[0] if results else ""
    choices = _choices(doc)
    payload = _evaluate_official_choice(raw_response, choices, _answer_text(doc))
    payload.update(
        {
            "id": str(doc.get("id", "")),
            "task": str(doc.get("task", "")),
            "difficulty": str(doc.get("difficulty", "")),
            "sub-category": str(doc.get("sub-category", "")),
        }
    )

    submission = dict(doc)
    submission["model_output"] = "" if raw_response is None else str(raw_response)
    submission["model_prediction"] = payload["prediction_text"]
    return {"mmaunew_accuracy": payload, "mmaunew_submission": submission}


def mmaunew_aggregate_results(results):
    if not results:
        return 0.0

    evaluable = [result for result in results if result.get("has_ground_truth", True)]
    if not evaluable:
        eval_logger.warning("MMAU-new has no ground-truth answers; accuracy is unavailable")
        return 0.0

    dimensions = (
        ("Task-wise", "task"),
        ("Difficulty-wise", "difficulty"),
        ("Sub-category-wise", "sub-category"),
    )
    for title, key in dimensions:
        grouped_scores = defaultdict(list)
        for result in evaluable:
            grouped_scores[result.get(key) or "Unknown"].append(
                float(result.get("score", 0.0))
            )
        eval_logger.info(f"MMAU-new {title} Accuracy:")
        for group in sorted(grouped_scores):
            scores = grouped_scores[group]
            eval_logger.info(
                f"  {group}: {sum(scores) * 100.0 / len(scores):.2f}% "
                f"over {len(scores)} samples"
            )

    total = sum(float(result.get("score", 0.0)) for result in evaluable)
    accuracy = total * 100.0 / len(evaluable)
    eval_logger.info(
        f"MMAU-new Total Accuracy: {accuracy:.5f}% over {len(evaluable)} samples"
    )
    return round(accuracy, 5)


def mmaunew_aggregate_submission(results, args):
    is_test_mini = bool(results) and all(
        str(result.get("split", "")).strip() == "test-mini" for result in results
    )
    filename = (
        "mmaunew_test_mini_results.json"
        if is_test_mini
        else "mmaunew_submission.json"
    )
    path = generate_submission_file(filename, args)
    keys_to_keep = [
        "id",
        "audio_id",
        "question",
        "choices",
        "answer",
        "model_output",
        "model_prediction",
        "dataset",
        "task",
        "split",
        "category",
        "sub-category",
        "difficulty",
    ]
    filtered_results = [
        {key: result[key] for key in keys_to_keep if key in result}
        for result in results
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(filtered_results, handle, ensure_ascii=False, indent=2)
    eval_logger.info(f"MMAU-new submission saved to {path}")
