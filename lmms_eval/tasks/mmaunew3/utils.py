"""MMAU-new2 with bounded prefix and order correction during evaluation.

Data loading, audio handling, prompting, and generation are inherited unchanged
from ``mmaunew2``.  The main metric first applies mmaunew's existing
``<answer>`` extraction, then removes one model-added ``A. ``-style prefix only
when its remaining body identifies a displayed choice and the original does not.
Unlike ``mmauoffical2``, MMAU-new does not label the displayed choices, so the
prefix is never interpreted as an option index.  Explanations and arbitrary
output are otherwise unchanged.
"""

from __future__ import annotations

import json
import re

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmaunew import utils as mmaunew_utils
from lmms_eval.tasks.mmaunew2 import utils as mmaunew2_utils


# Preserve every non-scoring part of mmaunew2 exactly.
mmaunew3_process_docs = mmaunew2_utils.mmaunew2_process_docs
mmaunew3_process_full_docs = mmaunew2_utils.mmaunew2_process_full_docs
mmaunew3_doc_to_audio = mmaunew2_utils.mmaunew2_doc_to_audio
mmaunew3_doc_to_text = mmaunew2_utils.mmaunew2_doc_to_text
mmaunew3_doc_to_messages = mmaunew2_utils.mmaunew2_doc_to_messages
mmaunew3_doc_to_choice = mmaunew2_utils.mmaunew2_doc_to_choice
mmaunew3_doc_to_target = mmaunew2_utils.mmaunew2_doc_to_target
mmaunew3_aggregate_results = mmaunew2_utils.mmaunew2_aggregate_results


def _words(text):
    return tuple(re.findall(r"\b\w+\b", text.lower()))


def prepare_prediction(raw_response, choices):
    """Remove formatting prefixes without assigning them option semantics."""

    raw_text = "" if raw_response is None else str(raw_response)
    extracted = mmaunew_utils._extract_final_answer(raw_text)
    output = extracted
    status = "unchanged"
    removed_prefix = ""
    sequences = [_words(choice) for choice in choices]
    output_words = _words(output)
    candidates = [
        sequence
        for sequence in sequences
        if sequence and output_words[:len(sequence)] == sequence
    ]

    # A leading ``A. `` may be either model formatting or real answer text.
    # Remove it only when the original text matches no choice and the stripped
    # body does. This also protects choices such as ``F. Scott Fitzgerald``.
    prefix = re.match(r"^\s*([A-Z])\.\s+(?=\S)", extracted, re.I)
    if not candidates and prefix:
        stripped = extracted[prefix.end():]
        stripped_words = _words(stripped)
        stripped_candidates = [
            sequence
            for sequence in sequences
            if sequence and stripped_words[:len(sequence)] == sequence
        ]
        if stripped_candidates:
            removed_prefix = prefix[1].upper()
            output = stripped
            output_words = stripped_words
            candidates = stripped_candidates
            status = "prefix_removed"

    selected = max(candidates, key=len, default=None)
    order_sensitive = any(
        left != right and set(left) == set(right)
        for index, left in enumerate(sequences)
        for right in sequences[index + 1:]
    )
    return {
        "raw_model_output": raw_text,
        "model_output": output,
        "normalization": status,
        "removed_prefix": removed_prefix,
        "selected_tokens": selected,
        "order_sensitive": order_sensitive,
        "normalization_input": extracted,
        "extracted_answer": extracted,
    }


def evaluate_mmaunew3_prediction(answer, prediction, choices):
    """Return raw, normalized, and narrowly order-corrected scores."""

    clean_choices = [str(choice).strip() for choice in choices]
    ground_truth = "" if answer is None else str(answer).strip()
    has_ground_truth = bool(ground_truth)
    details = prepare_prediction(prediction, clean_choices)

    raw_match = False
    submission_match = False
    corrected_match = False
    if has_ground_truth:
        raw_match = mmaunew_utils._official_string_match(
            ground_truth, details["raw_model_output"], clean_choices
        )
        submission_match = mmaunew_utils._official_string_match(
            ground_truth, details["model_output"], clean_choices
        )
        corrected_match = submission_match
        if details["order_sensitive"]:
            corrected_match = (
                corrected_match
                and details["selected_tokens"] == _words(ground_truth)
            )

    return {
        **details,
        "score": float(corrected_match),
        "official_raw_score": float(raw_match),
        "submission_score": float(submission_match),
        "match_method": (
            "bounded_prefix_and_order"
            if has_ground_truth
            else "ground_truth_unavailable"
        ),
        "has_ground_truth": has_ground_truth,
        "prediction_text": details["model_output"],
        "ground_truth_text": ground_truth,
        # Keep the historical sample-log field bounded as mmaunew2 does.
        "raw_response": details["raw_model_output"][:500],
    }


def _with_score(payload, score_field):
    result = dict(payload)
    result["score"] = float(payload[score_field])
    return result


def mmaunew3_process_results(doc, results):
    raw_response = results[0] if results else ""
    choices = mmaunew_utils._choices(doc)
    payload = evaluate_mmaunew3_prediction(
        mmaunew_utils._answer_text(doc), raw_response, choices
    )
    payload.update(
        {
            "id": str(doc.get("id", "")),
            "task": str(doc.get("task", "")),
            "difficulty": str(doc.get("difficulty", "")),
            "sub-category": str(doc.get("sub-category", "")),
        }
    )

    submission = dict(doc)
    submission["model_output"] = payload["model_output"]
    submission["model_prediction"] = payload["model_output"]
    return {
        "mmaunew3_accuracy": payload,
        "mmaunew3_official_raw_accuracy": _with_score(
            payload, "official_raw_score"
        ),
        "mmaunew3_submission_accuracy": _with_score(
            payload, "submission_score"
        ),
        "mmaunew3_submission": submission,
    }


def mmaunew3_aggregate_submission(results, args):
    is_test_mini = bool(results) and all(
        str(result.get("split", "")).strip() == "test-mini"
        for result in results
    )
    filename = (
        "mmaunew3_test_mini_results.json"
        if is_test_mini
        else "mmaunew3_submission.json"
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
    mmaunew_utils.eval_logger.info(f"MMAU-new3 submission saved to {path}")
