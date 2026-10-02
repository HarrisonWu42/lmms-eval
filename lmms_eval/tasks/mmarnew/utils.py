"""AudioMCQ MMAR task backed directly by the released benchmark JSONL."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger


AUDIO_MCQ_SYSTEM_PROMPT = (
    "You are an audio understanding model that answers multiple choice "
    "questions based on audio content."
)
EXPECTED_ROW_COUNT = 1_000
_ANSWER_TAG_RE = re.compile(
    r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)
_WORD_TOKEN_RE = re.compile(r"\b\w+\b")


def mmarnew_process_docs(dataset):
    """Validate the canonical AudioMCQ JSONL without rewriting its records."""

    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"AudioMCQ MMAR row count changed: {len(dataset)} != {EXPECTED_ROW_COUNT}"
        )
    required = {
        "answer",
        "audio_path",
        "id",
        "multi_choice",
        "question_text",
        "question_type",
    }
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"AudioMCQ MMAR JSONL is missing columns: {missing}")
    return dataset


def _choices(doc: Mapping[str, Any]) -> list[str]:
    raw_choices = doc.get("multi_choice") or []
    choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    if not choices:
        raise ValueError(f"MMAR sample {doc.get('id')} has no non-empty choices")
    return choices


def _answer_text(doc: Mapping[str, Any]) -> str:
    answer = str(doc.get("answer", "")).strip()
    if not answer:
        raise ValueError(f"MMAR sample {doc.get('id')} has an empty answer")
    if answer not in _choices(doc):
        raise ValueError(
            f"MMAR sample {doc.get('id')} answer is not present in multi_choice: "
            f"{answer!r}"
        )
    return answer


def _cached_audio_path(
    doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, Any] | None
) -> Path:
    kwargs = lmms_eval_specific_kwargs or {}
    cache_dir_value = kwargs.get("audio_cache_dir")
    if not cache_dir_value:
        raise ValueError("mmarnew requires lmms_eval_specific_kwargs.audio_cache_dir")

    source_name = Path(str(doc.get("audio_path", ""))).name
    filename = source_name or f"{doc.get('id', 'sample')}.wav"
    audio_path = Path(str(cache_dir_value)).expanduser() / filename
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Cached MMAR WAV is missing for {doc.get('id')}: {audio_path}. "
            "Generate the WAV cache from the original Parquet audio before "
            "running the JSONL-backed task."
        )
    return audio_path


def mmarnew_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    return [
        {
            "type": "audio",
            "url": str(_cached_audio_path(doc, lmms_eval_specific_kwargs)),
        }
    ]


def _format_audio_mcq_prompt(question: Any, choices: Sequence[Any]) -> str:
    question_text = str(question).strip()
    option_list = "[" + ", ".join(repr(str(choice).strip()) for choice in choices) + "]"
    return (
        f"{question_text} Please choose the answer from the following options: "
        f"{option_list}. Output the final answer in <answer> </answer>."
    )


def mmarnew_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return _format_audio_mcq_prompt(doc.get("question_text", ""), _choices(doc))


def mmarnew_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = mmarnew_doc_to_text(doc, lmms_eval_specific_kwargs)
    audio_items = mmarnew_doc_to_audio(doc, lmms_eval_specific_kwargs)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": AUDIO_MCQ_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [*audio_items, {"type": "text", "text": prompt}],
        },
    ]


def mmarnew_doc_to_choice(doc):
    return _choices(doc)


def mmarnew_doc_to_target(doc):
    return _answer_text(doc)


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


def mmarnew_process_results(doc, results):
    raw_response = results[0] if results else ""
    choices = _choices(doc)
    answer = _answer_text(doc)
    payload = {
        "id": str(doc.get("id", "")),
        "score": float(_official_string_match(answer, raw_response, choices)),
        "match_method": "official_token_set",
        "prediction_text": _extract_final_answer(raw_response),
        "ground_truth_text": answer,
        "question_type": str(doc.get("question_type", "")),
        "dataset_name": str(doc.get("dataset_name", "")),
        "raw_response": "" if raw_response is None else str(raw_response)[:500],
    }
    return {"mmarnew_accuracy": payload}


def mmarnew_aggregate_results(results):
    if not results:
        return 0.0

    question_type_scores = defaultdict(list)
    total = 0.0
    for result in results:
        score = float(result.get("score", 0.0))
        total += score
        question_type_scores[result.get("question_type") or "Unknown"].append(score)

    accuracy = total * 100.0 / len(results)
    eval_logger.info("MMAR-new AudioMCQ question-type accuracy:")
    for question_type in sorted(question_type_scores):
        scores = question_type_scores[question_type]
        eval_logger.info(
            f"  {question_type}: {sum(scores) * 100.0 / len(scores):.2f}% "
            f"({len(scores)} samples)"
        )
    eval_logger.info(f"MMAR-new overall accuracy: {accuracy:.2f}%")
    return accuracy
