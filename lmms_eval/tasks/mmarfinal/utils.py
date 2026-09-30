"""AudioMCQ MMAR data with the existing ``mmar`` evaluation protocol."""

import os
from pathlib import Path
from string import ascii_uppercase

import pyarrow.parquet as pq

from lmms_eval.tasks.mmar import utils as legacy


DEFAULT_DATASET_DIR = Path("/Users/andylubowen/Desktop/EIT-code/mmar")


def _dataset_dir():
    return Path(os.environ.get("MMARFINAL_DATASET_DIR", DEFAULT_DATASET_DIR)).expanduser()


def _legacy_doc(doc):
    """Adapt fields without changing choice order, padding or answer text."""
    sample_id = doc.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("MMAR-final sample must have a non-empty string id")
    choices = doc.get("multi_choice")
    if not isinstance(choices, list) or not 2 <= len(choices) <= len(ascii_uppercase):
        raise ValueError(f"MMAR-final sample {sample_id}: expected 2-26 choices")
    if not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"MMAR-final sample {sample_id}: choices must be strings")
    answer = doc.get("answer")
    if not isinstance(answer, str) or not answer.strip() or answer not in choices:
        raise ValueError(f"MMAR-final sample {sample_id}: answer is absent from choices")
    if not isinstance(doc.get("question_text"), str) or not doc["question_text"].strip():
        raise ValueError(f"MMAR-final sample {sample_id}: question_text is empty")
    category = doc.get("question_type")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"MMAR-final sample {sample_id}: question_type is empty")

    return {
        **doc,
        "question": doc["question_text"],
        "choices": [f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)],
        "answer": ascii_uppercase[choices.index(answer)],
        "category": category,
    }


def mmarfinal_process_docs(dataset):
    """Join only breakdown metadata; JSONL remains the question/answer source."""
    parquet_files = sorted((_dataset_dir() / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"MMAR-final metadata Parquet not found in {_dataset_dir() / 'data'}")
    metadata = {}
    for path in parquet_files:
        for row in pq.read_table(path, columns=["id", "modality", "subcategory"]).to_pylist():
            sample_id = row.pop("id")
            if sample_id in metadata:
                raise ValueError(f"MMAR-final duplicate Parquet id: {sample_id}")
            metadata[sample_id] = row

    seen = set()

    def enrich(doc):
        _legacy_doc(doc)
        sample_id = doc["id"]
        if sample_id in seen:
            raise ValueError(f"MMAR-final duplicate JSONL id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in metadata:
            raise ValueError(f"MMAR-final sample {sample_id}: missing Parquet metadata")
        return metadata[sample_id]

    # The external metadata is read on every load, not cached with stale labels.
    return dataset.map(enrich, load_from_cache_file=False)


def mmarfinal_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    cache_dir = Path(kwargs.get("audio_cache_dir", _dataset_dir() / "_audio_cache")).expanduser()
    source_path = doc.get("audio_path")
    if not isinstance(source_path, str) or not source_path.strip() or not Path(source_path).name:
        raise ValueError(f"MMAR-final sample {doc.get('id')}: audio_path is empty")
    audio_path = cache_dir / Path(source_path).name
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"MMAR-final sample {doc.get('id')}: missing or empty audio {audio_path}")
    return [{"type": "audio", "url": str(audio_path)}]


def mmarfinal_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return legacy.mmar_doc_to_text(_legacy_doc(doc), lmms_eval_specific_kwargs)


def mmarfinal_doc_to_target(doc):
    return legacy.mmar_doc_to_target(_legacy_doc(doc))


def mmarfinal_process_results(doc, results):
    return legacy.mmar_process_results(_legacy_doc(doc), results)


mmarfinal_aggregate_results = legacy.mmar_aggregate_results
