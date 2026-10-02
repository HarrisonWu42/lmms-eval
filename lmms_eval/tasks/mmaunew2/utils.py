"""JSON-backed MMAU test-mini task using the AudioMCQ prompt and scorer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmaunew import utils as mmaunew_utils


EXPECTED_ROW_COUNT = 1_000
REQUIRED_COLUMNS = {
    "answer",
    "audio_id",
    "choices",
    "id",
    "question",
}


def mmaunew2_process_docs(dataset):
    """Validate the official JSON test-mini without changing its records."""

    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"MMAU-new2 row count changed: {len(dataset)} != {EXPECTED_ROW_COUNT}"
        )

    missing = sorted(REQUIRED_COLUMNS - set(dataset.column_names))
    if missing:
        raise ValueError(f"MMAU-new2 JSON is missing columns: {missing}")
    return dataset


def mmaunew2_process_full_docs(dataset):
    """Keep the original ``mmaunew`` normalization for the full test split."""

    return mmaunew_utils.mmaunew_process_docs(dataset)


def _configured_path(
    key: str, lmms_eval_specific_kwargs: Mapping[str, Any] | None
) -> Path | None:
    value = (lmms_eval_specific_kwargs or {}).get(key)
    if not value:
        return None
    return Path(str(value)).expanduser()


def _audio_path(doc, lmms_eval_specific_kwargs=None) -> Path:
    audio_id = str(doc.get("audio_id", "")).strip()
    if not audio_id:
        raise ValueError(f"MMAU-new2 sample {doc.get('id')} has no audio_id")

    source = Path(audio_id).expanduser()
    candidates: list[Path] = []
    if source.is_absolute():
        candidates.append(source)
    else:
        audio_root = _configured_path("audio_root", lmms_eval_specific_kwargs)
        if audio_root is None:
            raise ValueError(
                "Relative MMAU-new2 test-mini audio_id requires audio_root in "
                "lmms_eval_specific_kwargs"
            )
        candidates.extend(
            [
                audio_root / source,
                audio_root / source.name,
                audio_root / "test-mini-audios" / source.name,
                audio_root / "_audio_cache" / source.name,
            ]
        )

    unique_candidates = list(
        dict.fromkeys(candidate.resolve() for candidate in candidates)
    )
    for candidate in unique_candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    checked = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        f"MMAU-new2 audio for sample {doc.get('id')} was not found. "
        f"The JSON was loaded successfully, but inference requires the test-mini WAVs; "
        f"checked: {checked}"
    )


def mmaunew2_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    if str(doc.get("split", "")).strip() != "test-mini":
        return mmaunew_utils.mmaunew_doc_to_audio(
            doc, lmms_eval_specific_kwargs
        )
    return [
        {
            "type": "audio",
            "url": str(_audio_path(doc, lmms_eval_specific_kwargs)),
        }
    ]


def mmaunew2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return mmaunew_utils.mmaunew_doc_to_text(doc, lmms_eval_specific_kwargs)


def mmaunew2_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = mmaunew2_doc_to_text(doc, lmms_eval_specific_kwargs)
    audio_items = mmaunew2_doc_to_audio(doc, lmms_eval_specific_kwargs)
    return mmaunew_utils._build_audio_mcq_messages(audio_items, prompt)


def mmaunew2_doc_to_choice(doc):
    return mmaunew_utils.mmaunew_doc_to_choice(doc)


def mmaunew2_doc_to_target(doc):
    return mmaunew_utils.mmaunew_doc_to_target(doc)


def mmaunew2_process_results(doc, results):
    base_result = mmaunew_utils.mmaunew_process_results(doc, results)
    return {
        "mmaunew2_accuracy": base_result["mmaunew_accuracy"],
        "mmaunew2_submission": base_result["mmaunew_submission"],
    }


def mmaunew2_aggregate_results(results):
    return mmaunew_utils.mmaunew_aggregate_results(results)


def mmaunew2_aggregate_submission(results, args):
    is_test_mini = bool(results) and all(
        str(result.get("split", "")).strip() == "test-mini"
        for result in results
    )
    filename = (
        "mmaunew2_test_mini_results.json"
        if is_test_mini
        else "mmaunew2_submission.json"
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
    mmaunew_utils.eval_logger.info(f"MMAU-new2 submission saved to {path}")
