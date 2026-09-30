"""lmms-eval adapter for the official MMSU prompt and scoring protocol.

The prompt mirrors ``dingdongwang/MMSU/mmsu_inference.py`` and the response
parser plus denominator behavior mirror ``mmsu_evaluation.py``.  The selected
remote dataset is the local lmms-eval mirror, whose ground truth is an A-D
letter rather than the official dataset's ``answer_gt`` text, so this adapter
derives ``answer_gt`` by indexing the corresponding choice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from loguru import logger as eval_logger

from lmms_eval.tasks.mmsu.utils import mmsu_doc_to_audio


CHOICE_LETTERS = ("A", "B", "C", "D")


def _choices(doc: Mapping[str, Any]) -> dict[str, str]:
    return {
        "A": str(doc.get("choice_a", "")),
        "B": str(doc.get("choice_b", "")),
        "C": str(doc.get("choice_c", "")),
        "D": str(doc.get("choice_d", "")),
    }


def _answer_gt(doc: Mapping[str, Any]) -> str:
    """Return official answer text from the selected mirror's letter label."""

    if "answer_gt" in doc:
        return str(doc.get("answer_gt", ""))

    answer_letter = str(doc.get("answer", "")).strip()
    return _choices(doc).get(answer_letter, "")


def mmsu_official_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    """Reuse the existing MMSU decoder while preserving this task's cache path."""

    return mmsu_doc_to_audio(doc, lmms_eval_specific_kwargs)


def mmsu_official_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the prompt from the official MMSU inference template."""

    del lmms_eval_specific_kwargs
    choices = _choices(doc)
    question_prompts = (
        "Choose the most suitable answer from options A, B, C, and D. "
        "You must respond with only A, B, C, or D. "
        "Do not provide any additional explanations or content."
    )
    formatted_choices = (
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}"
    )
    return (
        f"{question_prompts}\n\nQuestion: {doc.get('question', '')}"
        f"\n\n{formatted_choices}"
    )


def mmsu_official_doc_to_target(doc):
    """Expose the expected A-D label in lmms-eval sample logs."""

    return str(doc.get("answer", "")).strip()


def _parse_official_response(response: Any) -> tuple[str | None, bool, str]:
    """Mirror the official parser and whether it includes a row in the total.

    The official script accepts an A-D first character or an A-D penultimate
    character (for responses such as ``The answer is A.``).  A non-empty,
    non-``None`` response in any other format is skipped before ``total_count``
    is incremented. Empty strings and the literal string ``None`` remain in the
    denominator as incorrect predictions.
    """

    try:
        prediction = response.strip().replace("\n", "")
    except Exception:
        return None, False, ""

    model_prediction = None
    if prediction != "None" and prediction:
        if prediction[0] in CHOICE_LETTERS:
            model_prediction = prediction[0]
        elif len(prediction) > 1 and prediction[-2] in CHOICE_LETTERS:
            model_prediction = prediction[-2]
        else:
            return None, False, prediction

    return model_prediction, True, prediction


def mmsu_official_process_results(doc, results):
    """Score one response using the official MMSU evaluation semantics."""

    raw_response = results[0] if results else ""
    model_prediction, counted, normalized_response = _parse_official_response(
        raw_response
    )
    choices = _choices(doc)
    answer_gt = _answer_gt(doc)
    score = float(
        counted
        and model_prediction is not None
        and choices[model_prediction] == answer_gt
    )
    payload = {
        "score": score,
        "counted": counted,
        "prediction": model_prediction or "",
        "response": normalized_response,
        "answer_gt": answer_gt,
        "answer_letter": str(doc.get("answer", "")).strip(),
        "category": str(doc.get("category", "")),
        "sub_category": str(
            doc.get("sub_category", doc.get("sub-category", ""))
        ),
        "task_name": str(doc.get("task_name", "")),
    }
    return {"mmsu_official_accuracy": payload}


def mmsu_official_aggregate_results(results):
    """Reproduce the official category reporting and counted-row denominator."""

    category_subcategory = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0.0, "total": 0})
    )
    total_correct = 0.0
    total_count = 0
    skipped_count = 0

    for result in results:
        if not result.get("counted", False):
            skipped_count += 1
            continue

        category = result.get("category", "")
        sub_category = result.get("sub_category", "")
        score = float(result.get("score", 0.0))
        category_subcategory[category][sub_category]["correct"] += score
        category_subcategory[category][sub_category]["total"] += 1
        total_correct += score
        total_count += 1

    for category, subcategories in category_subcategory.items():
        category_correct = 0.0
        category_total = 0
        for sub_category, counts in subcategories.items():
            accuracy = (
                counts["correct"] / counts["total"] if counts["total"] else 0.0
            )
            eval_logger.info(
                f"Category: {category}, Sub-category: {sub_category}, "
                f"Accuracy: {accuracy:.4f}"
            )
            category_correct += counts["correct"]
            category_total += counts["total"]

        category_accuracy = (
            category_correct / category_total if category_total else 0.0
        )
        eval_logger.info(
            f"Category: {category}, Average Accuracy: {category_accuracy:.4f}"
        )

    overall_accuracy = total_correct / total_count if total_count else 0.0
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    eval_logger.info(f"Total count: {total_count}")
    eval_logger.info(f"Wrong-format responses excluded: {skipped_count}")
    return overall_accuracy
