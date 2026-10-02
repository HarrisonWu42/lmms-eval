"""Bounded MMAU prefix normalization and order-sensitive mini scoring.

The requested task name is mmauoffical2. The original mmauofficial task is
unchanged. Inference and data loading are shared; the corrected metric is not
the official leaderboard protocol. The hidden-answer service still uses its
own token-set scorer, even when submissions contain normalized predictions.
"""

from __future__ import annotations

import json
import re

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmauofficial.utils import (
    mmauofficial_aggregate_results as mmauoffical2_aggregate_results,
    mmauofficial_doc_to_audio as mmauoffical2_doc_to_audio,
    mmauofficial_doc_to_messages as mmauoffical2_doc_to_messages,
    mmauofficial_doc_to_target as mmauoffical2_doc_to_target,
    mmauofficial_doc_to_text as mmauoffical2_doc_to_text,
    string_match as official_string_match,
)


def _words(text):
    # The official tokenizer, retaining order and multiplicity.
    return tuple(re.findall(r"\b\w+\b", text.lower()))


def _display_key(text):
    return " ".join(text.lower().split()).rstrip(".").rstrip()


def prepare_prediction(prediction, choices):
    """Normalize a received response using choices only, never ground truth.

    Preserve literal choices (including musical notes and name initials).
    Remove one explicit label plus dot/whitespace before a nonempty body.
    Bare labels without a literal-choice interpretation and label/body
    conflicts are invalid responses, exported as empty strings without
    dropping their IDs. No answer tags, explanations or courtesy are removed.
    """
    output = prediction
    label_index = None
    status = "unchanged"
    literal = any(_display_key(prediction) == _display_key(c) for c in choices)

    if not literal:
        prefix = re.match(r"^\s*([A-Z])\.\s+(?=\S)", prediction, re.I)
        bare = re.fullmatch(r"\s*([A-Z])\.?\s*", prediction, re.I)
        mark = prefix or bare
        if mark:
            index = ord(mark[1].upper()) - ord("A")
            if index < len(choices):
                label_index = index
                output = prediction[prefix.end():] if prefix else ""
                status = "prefix_removed" if prefix else "bare_label"

    sequences = [_words(choice) for choice in choices]
    output_words = _words(output)
    candidates = [
        seq for seq in sequences
        if seq and output_words[:len(seq)] == seq
    ]
    # A short option may prefix a longer one (e.g. twenty / twenty-one).
    # Equal sequences from duplicate options have the same interpretation.
    selected = max(candidates, key=len, default=None)
    order_sensitive = any(
        left != right and set(left) == set(right)
        for i, left in enumerate(sequences)
        for right in sequences[i + 1:]
    )
    if (
        label_index is not None
        and selected is not None
        and sequences[label_index] != selected
    ):
        output = ""
        status = "label_conflict"

    return {
        "raw_model_output": prediction,
        "model_output": output,
        "normalization": status,
        "option_index": label_index,
        "selected_tokens": selected,
        "order_sensitive": order_sensitive,
    }


def evaluate_mmau_prediction(answer, prediction, choices):
    """Score labeled mini examples without changing the official matcher."""
    details = prepare_prediction(prediction, choices)
    submitted = official_string_match(answer, details["model_output"], choices)
    corrected = submitted
    if details["order_sensitive"]:
        # Unrecognized order-sensitive answers do not fall back to token sets.
        corrected = corrected and details["selected_tokens"] == _words(answer)
    return {
        **details,
        "match_raw": int(official_string_match(answer, prediction, choices)),
        "match_submission": int(submitted),
        "match_corrected": int(corrected),
    }


def mmauoffical2_process_results(doc, results):
    scored = dict(doc)
    submission = dict(doc)
    if results:
        scored.update(evaluate_mmau_prediction(doc["answer"], results[0], doc["choices"]))
        submission["model_output"] = scored["model_output"]

    fields = {
        "mmauoffical2_accuracy": "match_corrected",
        "mmauoffical2_official_raw_accuracy": "match_raw",
        "mmauoffical2_submission_accuracy": "match_submission",
    }
    processed = {}
    for metric, field in fields.items():
        sample = dict(scored)
        if results:
            sample["match"] = scored[field]
            if field == "match_raw":
                sample["model_output"] = scored["raw_model_output"]
        processed[metric] = sample
    processed["mmauoffical2_submission"] = {
        "submission": submission,
        "audit": scored,
    }
    return processed


def _write_submissions(results, args, submission_name, audit_name):
    path = generate_submission_file(submission_name, args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([row["submission"] for row in results], handle, ensure_ascii=False, indent=2)
    audit_path = generate_submission_file(audit_name, args)
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump([row["audit"] for row in results], handle, ensure_ascii=False, indent=2)
    eval_logger.info(f"MMAU normalized predictions saved to {path}")
    eval_logger.info(f"MMAU original responses and normalization audit saved to {audit_path}")


def mmauoffical2_aggregate_submission(results, args):
    _write_submissions(
        results, args,
        "mmauoffical2_test_mini_results.json",
        "mmauoffical2_test_mini_audit.json",
    )


def mmauoffical2_process_test_results(doc, results):
    # Deliberately do not read doc['answer'] or call any scorer on hidden test.
    details = prepare_prediction(results[0] if results else "", doc["choices"])
    return {
        "mmauoffical2_test_submission": {
            "submission": {"id": doc["id"], "model_prediction": details["model_output"]},
            "audit": {"id": doc["id"], **details},
        }
    }


def mmauoffical2_aggregate_test_submission(results, args):
    _write_submissions(
        results, args,
        "mmauoffical2_test_submission.json",
        "mmauoffical2_test_audit.json",
    )
    eval_logger.info(
        "Hidden-test accuracy is unavailable locally. The official service's "
        "token-set order ambiguity is not fixed by submission normalization."
    )
