import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize(value: Any) -> str:
    text = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _choices(doc: Dict[str, Any]) -> List[str]:
    raw_choices = doc.get("choices") or []
    if isinstance(raw_choices, str):
        try:
            raw_choices = json.loads(raw_choices)
        except json.JSONDecodeError:
            raw_choices = [raw_choices]
    return [_clean_text(choice) for choice in _as_list(raw_choices) if _clean_text(choice)]


def _is_mcq(doc: Dict[str, Any], choices: Optional[List[str]] = None) -> bool:
    choices = _choices(doc) if choices is None else choices
    task_type = _clean_text(doc.get("task_classification")).lower()
    return len(choices) >= 2 and task_type != "open-ended"


def _dataset_root(lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> Path:
    kwargs = lmms_eval_specific_kwargs or {}
    dataset_path = kwargs.get("dataset_path")
    if dataset_path:
        return Path(dataset_path).expanduser()
    return Path("/tmp/lmms_eval_audio_cache/mmau_pro")


def _resolve_audio_path(path_value: Any, dataset_root: Path) -> str:
    path = Path(_clean_text(path_value)).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    return str(path)


def mmau_pro_doc_to_audio(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    dataset_root = _dataset_root(lmms_eval_specific_kwargs)
    audio_paths = doc.get("audio_path") or doc.get("audio_paths") or doc.get("audio") or []

    audios = []
    for audio_path in _as_list(audio_paths):
        if isinstance(audio_path, dict):
            audio_path = audio_path.get("path") or audio_path.get("url")
        if not audio_path:
            continue
        audios.append({"type": "audio", "url": _resolve_audio_path(audio_path, dataset_root)})

    if not audios:
        eval_logger.warning(f"No audio path found for MMAU-Pro sample {doc.get('id', 'unknown')}")
    return audios


def mmau_pro_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    question = _clean_text(doc.get("question"))
    choices = _choices(doc)

    if _is_mcq(doc, choices):
        option_lines = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
        post_prompt = kwargs.get("mcq_post_prompt", "\nAnswer with the option's letter from the given choices directly.")
        return f"{pre_prompt}{question}\n{option_lines}{post_prompt}"

    post_prompt = kwargs.get("open_post_prompt", "\nAnswer the question directly and concisely.")
    return f"{pre_prompt}{question}{post_prompt}"


def _parse_mcq_response(response: str, choices: List[str]) -> str:
    response = _clean_text(response)
    padded = f" {response.strip()} "

    for idx, choice in enumerate(choices):
        letter = LETTERS[idx]
        patterns = (
            rf"\({letter}\)",
            rf"\b{letter}\b",
            rf"\b{letter}\.",
            rf"\boption\s+{letter}\b",
        )
        if any(re.search(pattern, padded, flags=re.IGNORECASE) for pattern in patterns):
            return choice

    normalized_response = _normalize(response)
    for choice in choices:
        normalized_choice = _normalize(choice)
        if normalized_choice and normalized_choice in normalized_response:
            return choice

    return response


def mmau_pro_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0] if results else ""
    choices = _choices(doc)
    is_mcq = _is_mcq(doc, choices)
    parsed_prediction = _parse_mcq_response(response, choices) if is_mcq else _clean_text(response)
    target = _clean_text(doc.get("answer"))

    score = None
    if is_mcq:
        score = 1.0 if _normalize(parsed_prediction) == _normalize(target) else 0.0

    sample = dict(doc)
    sample["model_output"] = response
    sample["parsed_prediction"] = parsed_prediction
    sample["is_mcq"] = is_mcq
    sample["simple_mcq_score"] = score

    return {
        "mmau_pro_mcq_accuracy": {
            "score": score,
            "is_mcq": is_mcq,
            "category": doc.get("category"),
            "task_classification": doc.get("task_classification"),
            "id": doc.get("id"),
        },
        "submission": sample,
    }


def mmau_pro_aggregate_mcq_accuracy(results: List[Dict[str, Any]]) -> float:
    scored = [result for result in results if result.get("score") is not None]
    if not scored:
        eval_logger.warning("No MCQ samples found for MMAU-Pro simple accuracy.")
        return 0.0

    overall = round(sum(result["score"] for result in scored) * 100 / len(scored), 5)
    eval_logger.info("=" * 50)
    eval_logger.info(f"MMAU-Pro simple MCQ accuracy: {overall} over {len(scored)} samples")
    eval_logger.info("=" * 50)
    return overall


def mmau_pro_aggregate_submission(results: List[Dict[str, Any]], args) -> float:
    path = generate_submission_file("mmau_pro_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    eval_logger.info(f"MMAU-Pro predictions saved to {path}.")
    return 0.0
