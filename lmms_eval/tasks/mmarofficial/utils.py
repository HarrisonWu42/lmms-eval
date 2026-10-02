"""MMAR's original accuracy protocol using the released official metadata.

Read question, choices, and answer directly from MMAR-meta.jsonl. The local
Parquet mirror supplies the already-materialized WAVs, not ground-truth labels.
Apply official evaluation.py's token-set rule to the unmodified prediction.
This task does not implement the separate MMAR-Rubrics reasoning evaluation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger


DEFAULT_AUDIO_CACHE_DIR = Path("/Users/andylubowen/Desktop/EIT-code/mmar/_audio_cache")
EXPECTED_ROW_COUNT = 1_000


def mmarofficial_process_docs(dataset):
    """Validate the official metadata without rewriting its text or labels."""

    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Official MMAR row count changed: {len(dataset)} != {EXPECTED_ROW_COUNT}"
        )

    required = {
        "answer",
        "audio_path",
        "category",
        "choices",
        "id",
        "modality",
        "question",
        "sub-category",
    }
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Official MMAR metadata is missing columns: {missing}")

    return dataset


def _audio_cache_path(
    doc: Mapping[str, Any],
    lmms_eval_specific_kwargs: Mapping[str, Any] | None,
) -> Path:
    """Resolve the cached WAV using the official audio filename, not the ID."""

    kwargs = lmms_eval_specific_kwargs or {}
    cache_dir = Path(str(kwargs.get("audio_cache_dir", DEFAULT_AUDIO_CACHE_DIR))).expanduser()
    source_path = str(doc.get("audio_path") or "")
    if not source_path:
        raise ValueError(f"MMAR sample {doc.get('id')} has no audio_path")

    audio_path = cache_dir / Path(source_path).name
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Cached MMAR WAV is missing for {doc.get('id')}: {audio_path}"
        )
    return audio_path


def mmarofficial_doc_to_audio(
    doc: Mapping[str, Any],
    lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return the local MMAR WAV as an lmms-eval audio item."""

    return [
        {
            "type": "audio",
            "url": str(_audio_cache_path(doc, lmms_eval_specific_kwargs)),
        }
    ]


def mmarofficial_doc_to_text(
    doc: Mapping[str, Any],
    lmms_eval_specific_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Ask for option text; reference thinking/cue/rubric fields are not input.

    The official release specifies the scoring protocol, not a common model
    inference prompt. Keep this adapter's text-answer instruction explicit.
    """

    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", "The best answer is:"))
    displayed_choices = [
        f"{chr(ord('A') + index)}. {choice_text}"
        for index, choice_text in enumerate(doc["choices"])
        if choice_text.strip()
    ]
    instruction = (
        "Listen to the audio and answer the following multiple-choice question. "
        "Respond with only the text of the correct option, not its letter."
    )
    parts = [
        pre_prompt,
        instruction,
        str(doc.get("question", "")).strip(),
        "\n".join(displayed_choices),
        post_prompt,
    ]
    return "\n".join(part for part in parts if part).strip()


def mmarofficial_doc_to_target(doc: Mapping[str, Any]) -> str:
    """Expose the official answer text in lmms-eval sample logs."""

    return doc["answer"]


def string_match(answer: str, prediction: str, choices: Sequence[str]) -> bool:
    """Mirror MMAR's original official ``evaluation.py`` token-set rule."""

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def mmarofficial_process_results(
    doc: Mapping[str, Any], results: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Score one model response with the original official evaluator."""

    prediction = results[0] if results else ""
    answer = doc["answer"]
    choices = doc["choices"]
    score = float(string_match(answer, prediction, choices))
    payload = {
        "id": str(doc.get("id", "")),
        "score": score,
        "match": int(score),
        "answer_prediction": prediction,
        "answer": answer,
        "choices": choices,
        "modality": str(doc.get("modality", "")),
        "category": str(doc.get("category", "")),
        "sub-category": str(doc.get("sub-category", "")),
    }
    return {"mmarofficial_accuracy": payload}


def mmarofficial_aggregate_results(results: Sequence[Mapping[str, Any]]) -> float:
    """Report official modality/category/sub-category and total accuracies."""

    if not results:
        eval_logger.warning("No MMAR results to aggregate")
        return 0.0

    modality_metrics: dict[str, list[float]] = defaultdict(list)
    category_metrics: dict[str, list[float]] = defaultdict(list)
    subcategory_metrics: dict[str, list[float]] = defaultdict(list)

    for result in results:
        score = float(result.get("score", 0.0))
        modality_metrics[str(result.get("modality", ""))].append(score)
        category_metrics[str(result.get("category", ""))].append(score)
        subcategory_metrics[str(result.get("sub-category", ""))].append(score)

    def log_breakdown(
        heading: str, grouped_scores: Mapping[str, Sequence[float]]
    ) -> None:
        eval_logger.info(heading)
        for name in sorted(grouped_scores):
            scores = grouped_scores[name]
            accuracy = sum(scores) * 100.0 / len(scores)
            eval_logger.info(
                f"{name} : {accuracy:.2f}% over {len(scores)} samples"
            )

    log_breakdown("Modality-wise Accuracy:", modality_metrics)
    log_breakdown("Category-wise Accuracy:", category_metrics)
    log_breakdown("Sub-category-wise Accuracy:", subcategory_metrics)

    total_accuracy = (
        sum(float(result.get("score", 0.0)) for result in results)
        * 100.0
        / len(results)
    )
    eval_logger.info(
        f"Total Accuracy: {total_accuracy:.2f}% over {len(results)} samples"
    )
    return total_accuracy
