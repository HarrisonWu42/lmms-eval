"""MMAU mini scoring and full-test submission using the official interfaces.

Reference: Sakshi113/MMAU, commit 110127f54c0dfba3faa5ec9feee4a7e4148679c5.
Score the complete model output: no answer-tag extraction, letter mapping,
or semantic matching. The official release does not prescribe an inference
prompt; this adapter asks for the selected option's exact text.
The public test split has no answers, so it only exports raw predictions for
https://huggingface.co/spaces/sonalkum/MMAU-Eval and never invokes the scorer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger as eval_logger


DEFAULT_AUDIO_DIR = Path(
    "/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/test-mini-audios"
)


def mmauofficial_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    audio_dir = Path(kwargs.get("audio_dir", DEFAULT_AUDIO_DIR)).expanduser()
    audio_path = audio_dir / Path(doc["audio_id"]).name
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"MMAU audio is missing for {doc['id']}: {audio_path}")
    return [{"type": "audio", "url": str(audio_path)}]


def mmauofficial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Display every option, without including the reference answer/metadata."""

    choices = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(doc["choices"])
    )
    return (
        "Listen to the audio and answer the following multiple-choice question. "
        "Respond with only the exact text of the selected option. "
        "Do not include the option letter, explanations, or answer tags.\n\n"
        f"Question: {doc['question']}\n\n{choices}"
    )


def mmauofficial_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return [
        {
            "role": "user",
            "content": [
                *mmauofficial_doc_to_audio(doc, lmms_eval_specific_kwargs),
                {
                    "type": "text",
                    "text": mmauofficial_doc_to_text(doc, lmms_eval_specific_kwargs),
                },
            ],
        }
    ]


def mmauofficial_doc_to_target(doc):
    return doc["answer"]


def string_match(answer, prediction, choices):
    """Reproduce the official token-set matcher without input preprocessing."""

    def tokenize(text):
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def mmauofficial_process_results(doc, results):
    """Keep a missing response distinct from a present, empty model output."""

    sample = dict(doc)
    if results:
        sample["model_output"] = results[0]
        sample["match"] = int(
            string_match(sample["answer"], sample["model_output"], sample["choices"])
        )
    return {
        "mmauofficial_accuracy": sample,
        "mmauofficial_submission": sample,
    }


def mmauofficial_aggregate_results(results):
    """Use official groups, percentage scale, and counted-row denominator.

    Like evaluation.py, skip missing model_output fields, but count empty
    strings as incorrect. An entirely unscored input raises ZeroDivisionError.
    """

    task_metrics = {"sound": [0, 0], "music": [0, 0], "speech": [0, 0]}
    diff_metrics = {"easy": [0, 0], "hard": [0, 0], "medium": [0, 0]}
    subcat_metrics = {}
    correct, total = 0, 0

    for sample in results:
        if "model_output" not in sample:
            continue
        match = sample["match"]
        buckets = [task_metrics[sample["task"]], diff_metrics[sample["difficulty"]]]
        subcat = sample.get("sub-category")
        if subcat is not None:
            buckets.append(subcat_metrics.setdefault(subcat, [0, 0]))
        for bucket in buckets:
            bucket[0] += match
            bucket[1] += 1
        correct += match
        total += 1

    for heading, metrics in (
        ("Task-wise Accuracy:", task_metrics),
        ("Difficulty-wise Accuracy:", diff_metrics),
        ("Sub-category-wise Accuracy:", subcat_metrics),
    ):
        eval_logger.info("*" * 30)
        eval_logger.info(heading)
        for name, (n_correct, n_total) in metrics.items():
            accuracy = (n_correct / n_total) * 100 if n_total > 0 else 0
            eval_logger.info(f"{name} : {accuracy:.2f}% over {n_total} samples")

    accuracy = (correct / total) * 100
    eval_logger.info("*" * 30)
    eval_logger.info(f"Total Accuracy: {accuracy:.2f}% over {total} samples")
    eval_logger.info("*" * 30)
    # The official script skips missing fields before this counter increments.
    eval_logger.info("No prediction count: 0")
    return accuracy


def mmauofficial_aggregate_submission(results, args):
    """Export the same raw predictions that evaluation.py reads and scores."""

    from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

    path = generate_submission_file("mmauofficial_test_mini_results.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    eval_logger.info(f"MMAU official predictions saved to {path}")


def mmauofficial_process_test_results(doc, results):
    """Prepare an HF Space prediction without scoring the empty public answer.

    The Space reads model_prediction, unlike evaluation.py's model_output.
    Keep failed/empty responses as empty strings so their IDs remain in the
    submission and the official service counts them in its denominator.
    """

    return {
        "mmauofficial_test_submission": {
            "id": doc["id"],
            "model_prediction": results[0] if results else "",
        }
    }


def mmauofficial_aggregate_test_submission(results, args):
    """Write the official test submission; no local accuracy is available."""

    from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

    path = generate_submission_file("mmauofficial_test_submission.json", args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    eval_logger.info(
        f"MMAU official test submission saved to {path} ({len(results)} samples). "
        "Test accuracy is unavailable locally; upload this JSON to "
        "https://huggingface.co/spaces/sonalkum/MMAU-Eval."
    )
