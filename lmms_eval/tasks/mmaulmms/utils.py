import json
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def _choices(doc):
    """Normalize choices from either JSON rows or Parquet rows."""
    choices = doc["choices"]
    if isinstance(choices, str):
        choices = json.loads(choices)
    return [str(choice) for choice in choices]


def _choice_letters(choice_count):
    return [chr(ord("A") + index) for index in range(choice_count)]


def doc_to_target(doc):
    return doc.get("answer", "")


def doc_to_audio(doc, lmms_eval_specific_kwargs):
    audio_dir = Path(lmms_eval_specific_kwargs["audio_dir"]).expanduser()
    audio_path = audio_dir / Path(str(doc["audio_id"])).name
    return [{"type": "audio", "url": str(audio_path)}]


def doc_to_text(doc, lmms_eval_specific_kwargs):
    choices = _choices(doc)
    letters = _choice_letters(len(choices))
    formatted_choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{doc['question']}\n{formatted_choices}{post_prompt}"


def doc_to_choice(doc):
    return _choices(doc)


def parse_choice_label(response, valid_letters):
    """Extract a valid option label while preferring the requested leading format."""
    response = "" if response is None else str(response).strip()
    valid_letters = set(valid_letters)

    leading_match = re.match(
        r"^(?:"
        r"\*\*\s*([A-Z])(?:[.)])?\s*\*\*"
        r"|[\"']\s*([A-Z])(?:[.)])?\s*[\"']"
        r"|\(([A-Z])\)"
        r"|\[([A-Z])\]"
        r"|([A-Z])"
        r")(?:[.,:;!?)])?(?:\s|$)",
        response,
        flags=re.IGNORECASE,
    )
    if leading_match:
        label = next(group for group in leading_match.groups() if group).upper()
        if label in valid_letters:
            return label

    fallback_patterns = (
        r"\*\*\s*([A-Z])(?:[.)])?\s*\*\*",
        r"[\"']\s*([A-Z])(?:[.)])?\s*[\"']",
        r"\(([A-Z])\)",
        r"\[([A-Z])\]",
        r"\b(?:answer|option)\s*(?:is|:)?\s*([A-Z])\b",
        r"(?<![A-Z0-9])([A-Z])\.(?=\s|$)",
    )
    candidates = []
    for pattern in fallback_patterns:
        for match in re.finditer(pattern, response, flags=re.IGNORECASE):
            label = match.group(1).upper()
            if label in valid_letters:
                candidates.append((match.start(), label))

    return max(candidates)[1] if candidates else ""


def letter_to_choice(letter, choices):
    if not letter:
        return ""
    index = ord(letter) - ord("A")
    return choices[index] if 0 <= index < len(choices) else ""


def mmaulmms_process_results(doc, result):
    choices = _choices(doc)
    valid_letters = _choice_letters(len(choices))
    raw_response = result[0] if result else ""
    response_label = parse_choice_label(raw_response, valid_letters)
    model_prediction = letter_to_choice(response_label, choices)

    ground_truth = str(doc.get("answer", "")).strip().lower()
    score = 1.0 if model_prediction and model_prediction.strip().lower() == ground_truth else 0.0
    submission = {**doc, "model_prediction": model_prediction}

    return {
        "accuracy": {"overall": score, "task": doc.get("task", "unknown")},
        "submission": submission,
    }


def mmaulmms_aggregate_results(results):
    total_correct = 0
    group_totals = defaultdict(int)
    group_correct = defaultdict(int)

    for result in results:
        accuracy = result["overall"]
        task = result["task"]
        total_correct += accuracy
        group_totals[task] += 1
        group_correct[task] += accuracy

    overall_accuracy = round(total_correct * 100 / len(results), 5)
    categorical_accuracy = {
        task: round(group_correct[task] * 100 / group_totals[task], 5)
        for task in group_totals
    }

    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall accuracy: {overall_accuracy}")
    eval_logger.info("Categorical accuracy:")
    for task, accuracy in categorical_accuracy.items():
        eval_logger.info(f"{task} accuracy: {accuracy}")
    eval_logger.info("=" * 50)
    return overall_accuracy


def mmaulmms_aggregate_results_for_submission(results, args):
    path = generate_submission_file("mmaulmms_submission.json", args)
    keys_to_keep = [
        "id",
        "audio_id",
        "question",
        "choices",
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

    with open(path, "w", encoding="utf-8") as submission_file:
        json.dump(filtered_results, submission_file, indent=4, ensure_ascii=False)
    eval_logger.info(f"Results saved to {path}.")
