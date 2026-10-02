"""AudioMCQ MMSU data with the existing ``mmsu`` letter-scoring protocol."""

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase

import librosa
import pyarrow.parquet as pq

from lmms_eval.tasks.mmsu import utils as legacy


DEFAULT_DATASET_DIR = Path("/Users/andylubowen/Desktop/EIT-code/mmsu")


def _dataset_dir():
    return Path(os.environ.get("MMSUFINAL_DATASET_DIR", DEFAULT_DATASET_DIR)).expanduser()


def _legacy_doc(doc):
    sample_id = doc.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("MMSU-final sample must have a non-empty string id")
    choices = doc.get("multi_choice")
    if not isinstance(choices, list) or not 2 <= len(choices) <= len(ascii_uppercase):
        raise ValueError(f"MMSU-final sample {sample_id}: expected 2-26 choices")
    if not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"MMSU-final sample {sample_id}: choices must be strings")
    answer = doc.get("answer")
    if not isinstance(answer, str) or not answer.strip() or answer not in choices:
        raise ValueError(f"MMSU-final sample {sample_id}: answer is absent from choices")
    if not isinstance(doc.get("question_text"), str) or not doc["question_text"].strip():
        raise ValueError(f"MMSU-final sample {sample_id}: question_text is empty")
    category = doc.get("question_type")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"MMSU-final sample {sample_id}: question_type is empty")

    # The three repeated-answer rows in this release use the first occurrence
    # in the original MMSU letter labels as well. Do not accept both letters.
    return {
        **doc,
        "question": doc["question_text"],
        **{f"choice_{ascii_uppercase[i].lower()}": choice for i, choice in enumerate(choices)},
        "answer": ascii_uppercase[choices.index(answer)],
        "category": category.capitalize(),
    }


def mmsufinal_process_docs(dataset):
    """Preserve the legacy breakdowns without importing Parquet answer labels."""
    parquet_files = sorted((_dataset_dir() / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"MMSU-final metadata Parquet not found in {_dataset_dir() / 'data'}")
    metadata = {}
    for path in parquet_files:
        for row in pq.read_table(path, columns=["id", "sub_category", "task_name"]).to_pylist():
            sample_id = row.pop("id")
            if sample_id in metadata:
                raise ValueError(f"MMSU-final duplicate Parquet id: {sample_id}")
            metadata[sample_id] = row

    seen = set()

    def enrich(doc):
        _legacy_doc(doc)
        sample_id = doc["id"]
        if sample_id in seen:
            raise ValueError(f"MMSU-final duplicate JSONL id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in metadata:
            raise ValueError(f"MMSU-final sample {sample_id}: missing Parquet metadata")
        return metadata[sample_id]

    return dataset.map(enrich, load_from_cache_file=False)


@lru_cache(maxsize=1)
def _parquet_audio_index(dataset_dir):
    """Locate embedded audio by id without loading all audio into memory."""
    index = {}
    for path in sorted((Path(dataset_dir) / "data").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            ids = parquet.read_row_group(group, columns=["id"]).column("id").to_pylist()
            for row, sample_id in enumerate(ids):
                index[sample_id] = (str(path), group, row)
    return index


@lru_cache(maxsize=1)
def _parquet_audio_group(path, group):
    # Reuse the current group so consecutive cache misses do not reread it.
    return pq.ParquetFile(path).read_row_group(group, columns=["audio"]).column("audio")


def mmsufinal_doc_to_audio(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    cache_dir = Path(kwargs.get("audio_cache_dir", _dataset_dir() / "_audio_cache")).expanduser()
    source_path = doc.get("audio_path")
    if not isinstance(source_path, str) or not source_path.strip() or not Path(source_path).name:
        raise ValueError(f"MMSU-final sample {doc.get('id')}: audio_path is empty")
    audio_path = cache_dir / Path(source_path).name
    if audio_path.suffix.lower() != ".wav":
        audio_path = audio_path.with_suffix(".wav")
    audio = {"path": source_path}
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        path, group, row = _parquet_audio_index(str(_dataset_dir()))[doc["id"]]
        encoded = _parquet_audio_group(path, group)[row].as_py()
        # Match the original MMSU Audio(sampling_rate=16000) mono decoding.
        audio["array"], audio["sampling_rate"] = librosa.load(BytesIO(encoded["bytes"]), sr=16000)
    output_path = legacy._audio_cache_file(audio, doc, {**kwargs, "audio_cache_dir": str(cache_dir)})
    return [{"type": "audio", "url": output_path}]


def mmsufinal_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    adapted = _legacy_doc(doc)
    choices = doc["multi_choice"]
    if len(choices) == 4:
        return legacy.mmsu_doc_to_text(adapted, lmms_eval_specific_kwargs)

    kwargs = lmms_eval_specific_kwargs or {}
    letters = ascii_uppercase[:len(choices)]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    letter_list = " or ".join(letters) if len(letters) == 2 else ", ".join(letters[:-1]) + f", or {letters[-1]}"
    return (
        f"{kwargs.get('pre_prompt', '')}Listen to the audio and answer the following question.\n\n"
        f"Question: {adapted['question']}\n\n{options}\n\n"
        f"Answer with the letter ({letter_list}) only.{kwargs.get('post_prompt', '')}"
    )


def mmsufinal_doc_to_target(doc):
    return _legacy_doc(doc)["answer"]


def mmsufinal_process_results(doc, results):
    adapted = _legacy_doc(doc)
    if len(doc["multi_choice"]) == 4:
        return legacy.mmsu_process_results(adapted, results)

    response = results[0] if results else ""
    index2ans, all_choices = legacy.get_multi_choice_info(doc["multi_choice"])
    prediction = legacy.parse_multi_choice_response(response or "", all_choices, index2ans)
    return {
        "accuracy": {
            "score": float(bool(prediction) and prediction == adapted["answer"]),
            "category": adapted["category"],
            "sub_category": adapted.get("sub_category", "Unknown"),
            "task_name": adapted.get("task_name", "Unknown"),
        }
    }


mmsufinal_aggregate_results = legacy.mmsu_aggregate_results
