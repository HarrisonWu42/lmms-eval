import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


DEFAULT_AUDIO_CACHE_DIR = Path("/tmp/lmms_eval_audio_cache/mmau")
DEFAULT_AUDIO_CACHE_SUBDIR = "_audio_cache"


def _choice_letters(choice_count):
    """Return the valid answer letters for the current question."""
    return [chr(ord("A") + index) for index in range(choice_count)]


def _audio_cache_dir(lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    configured_cache_dir = kwargs.get("audio_cache_dir")
    if configured_cache_dir:
        return Path(configured_cache_dir).expanduser()

    dataset_path = kwargs.get("dataset_path")
    if dataset_path:
        dataset_dir = Path(dataset_path).expanduser()
        if dataset_dir.is_absolute() or dataset_dir.exists():
            return dataset_dir / kwargs.get("audio_cache_subdir", DEFAULT_AUDIO_CACHE_SUBDIR)

    return DEFAULT_AUDIO_CACHE_DIR


def _audio_cache_file(doc, lmms_eval_specific_kwargs=None):
    audio = doc["audio"]
    cache_dir = _audio_cache_dir(lmms_eval_specific_kwargs)
    filename = Path(audio.get("path") or f"{doc.get('id', 'sample')}.wav").name
    output_path = cache_dir / filename
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
    if not output_path.exists() or output_path.stat().st_size == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio["array"], audio["sampling_rate"])
    return str(output_path)


def doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    return [{"type": "audio", "url": _audio_cache_file(doc, lmms_eval_specific_kwargs)}]


def doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["question"]
    choices = json.loads(doc["choices"])
    letters = _choice_letters(len(choices))
    formatted_choices = "\n".join([f"{letter}. {choice}" for letter, choice in zip(letters, choices)])
    return f"{pre_prompt}{question}\n{formatted_choices}{post_prompt}"


def doc_to_choice(doc):
    choices = json.loads(doc["choices"])
    return choices


def mmau_process_results(doc, result):
    choices = json.loads(doc["choices"])
    valid_letters = _choice_letters(len(choices))
    raw_response = result[0] if result else ""
    response_letter = parse_multi_choice_response(raw_response, valid_letters)
    response = letter_to_ans(response_letter, choices)
    doc["model_prediction"] = response
    normalized_response = response.strip().lower()
    gt_ans = doc["answer"].strip().lower()
    score = 1.0 if response and normalized_response == gt_ans else 0.0

    return {"accuracy": {"overall": score, "task": doc["task"]}, "submission": {**doc}}


def mmau_aggregate_results(results):
    total_correct = 0
    group_totals = defaultdict(int)
    group_correct = defaultdict(int)

    for result in results:
        accuracy = result["overall"]
        total_correct += accuracy

        group_totals[result["task"]] += 1
        group_correct[result["task"]] += accuracy

    overall_accuracy = round(total_correct * 100 / len(results), 5)
    categorical_accuracy = {key: round(group_correct[key] * 100 / group_totals[key], 5) for key in group_totals.keys()}
    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall accuracy: {overall_accuracy}")
    eval_logger.info("Categorical accuracy: ")
    for key, value in categorical_accuracy.items():
        eval_logger.info(f"{key} accuracy: {value}")
    eval_logger.info("=" * 50)
    return overall_accuracy


def mmau_aggregate_results_for_submission(results, args):
    path = generate_submission_file("mmau_submission.json", args)
    filtered_results = []
    keys_to_keep = ["id", "audio_id", "question", "choices", "model_prediction", "dataset", "task", "split", "category", "sub-category", "difficulty"]

    for result in results:
        filtered_result = {key: result[key] for key in keys_to_keep if key in result}
        filtered_results.append(filtered_result)

    results = filtered_results
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    eval_logger.info(f"Results saved to {path}.")


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter, or an empty string for an invalid answer.
    """
    response = "" if response is None else str(response)

    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates, keep the prediction invalid instead of guessing
    if len(candidates) == 0:
        pred_index = ""
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def letter_to_ans(letter, choices):
    if not isinstance(letter, str) or len(letter) != 1:
        return ""

    index = ord(letter.upper()) - ord("A")
    if index < 0 or index >= len(choices):
        return ""
    return choices[index]
