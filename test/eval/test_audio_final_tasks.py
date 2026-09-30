"""Legacy scoring parity and JSONL integration for MMAR/MMSU-final."""

import copy
import json
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from datasets import Dataset

from lmms_eval import utils as config_utils
from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.mmar import utils as mmar
from lmms_eval.tasks.mmarfinal import utils as mmarfinal
from lmms_eval.tasks.mmsu import utils as mmsu
from lmms_eval.tasks.mmsufinal import utils as mmsufinal


TASK_ROOT = Path(__file__).resolve().parents[2] / "lmms_eval/tasks"
CASES = [("mmar", mmarfinal, mmar, "mmar_accuracy"), ("mmsu", mmsufinal, mmsu, "accuracy")]


def _doc(name):
    return {
        "id": "sample",
        "question_text": "Which sound is heard?",
        "multi_choice": ["owl", "robot", "rooster", "parrot"],
        "answer": "rooster",
        "question_type": "Semantic Layer" if name == "mmar" else "reasoning",
        "audio_path": f"{name.upper()}/audio/sample.mp3",
        "dataset_name": name.upper(),
    }


def _expected_legacy_doc(doc, name):
    expected = {
        "id": doc["id"],
        "question": doc["question_text"],
        "answer": "C",
        "category": "Semantic Layer" if name == "mmar" else "Reasoning",
    }
    if name == "mmar":
        expected["choices"] = ["A. owl", "B. robot", "C. rooster", "D. parrot"]
    else:
        expected.update(choice_a="owl", choice_b="robot", choice_c="rooster", choice_d="parrot")
    return expected


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
def test_four_option_prompt_and_response_parity(name, final, legacy, metric):
    doc = _doc(name)
    original = copy.deepcopy(doc)
    expected = _expected_legacy_doc(doc, name)
    prompt = getattr(final, f"{name}final_doc_to_text")
    score = getattr(final, f"{name}final_process_results")
    for kwargs in [None, {"pre_prompt": "prefix\n", "post_prompt": "\nsuffix"}]:
        assert prompt(doc, kwargs) == getattr(legacy, f"{name}_doc_to_text")(expected, kwargs)
    for results in [[], [None], [""], ["C"], ["(C)"], ["C."], ["c"], ["C\n"], ["rooster"], ["A"], ["Z"], ["A. first\nC. second"]]:
        assert score(doc, results) == getattr(legacy, f"{name}_process_results")(expected, results)
    assert doc == original


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
@pytest.mark.parametrize("count", [2, 3, 5, 6, 26])
def test_variable_options_and_last_label_do_not_overflow(name, final, legacy, metric, count):
    doc = _doc(name)
    doc["multi_choice"] = [f"option {i}" for i in range(count)]
    doc["answer"] = doc["multi_choice"][-1]
    target = getattr(final, f"{name}final_doc_to_target")(doc)
    assert target == ascii_uppercase[count - 1]
    prompt = getattr(final, f"{name}final_doc_to_text")(doc)
    assert f"{target}. {doc['answer']}" in prompt
    score = getattr(final, f"{name}final_process_results")
    assert score(doc, [f"({target})"])[metric]["score"] == 1.0
    for response in ["A", "", None, "?"]:
        assert score(doc, [response])[metric]["score"] == 0.0
    if count < 26:
        assert score(doc, [ascii_uppercase[count]])[metric]["score"] == 0.0


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
@pytest.mark.parametrize("changes", [
    {"multi_choice": []}, {"multi_choice": ["owl"]},
    {"multi_choice": ["rooster"] * 27}, {"multi_choice": None},
    {"multi_choice": ["rooster", None]}, {"answer": "missing"},
    {"answer": ""}, {"answer": None}, {"question_text": None},
    {"question_type": None},
])
def test_invalid_data_fails_before_indexing(name, final, legacy, metric, changes):
    doc = {**_doc(name), **changes}
    with pytest.raises(ValueError, match="sample"):
        getattr(final, f"{name}final_doc_to_target")(doc)


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
def test_empty_and_literal_none_options_keep_their_positions(name, final, legacy, metric):
    doc = {**_doc(name), "multi_choice": ["", "None", "rooster", "parrot"]}
    prompt = getattr(final, f"{name}final_doc_to_text")(doc)
    assert "A. \nB. None\nC. rooster\nD. parrot" in prompt
    assert getattr(final, f"{name}final_doc_to_target")(doc) == "C"


@pytest.mark.parametrize("choices,letter,other", [
    (["blow", "bow", "bow", "bough"], "B", "C"),
    (["bough", "beau", "bow", "bow"], "C", "D"),
    (["bow", "bone", "bow", "bough"], "A", "C"),
])
def test_repeated_answer_uses_original_single_letter(choices, letter, other):
    doc = {**_doc("mmsu"), "multi_choice": choices, "answer": "bow"}
    assert mmsufinal.mmsufinal_doc_to_target(doc) == letter
    assert mmsufinal.mmsufinal_process_results(doc, [letter])["accuracy"]["score"] == 1.0
    assert mmsufinal.mmsufinal_process_results(doc, [other])["accuracy"]["score"] == 0.0


def test_mmsu_e_prompt_and_scoring_use_legacy_letter_rules():
    doc = _doc("mmsu")
    doc["multi_choice"].append("correct fifth option")
    doc["answer"] = "correct fifth option"
    prompt = mmsufinal.mmsufinal_doc_to_text(doc)
    assert "E. correct fifth option" in prompt
    assert prompt.endswith("Answer with the letter (A, B, C, D, or E) only.")
    for response, expected in [("E", 1), ("(E)", 1), ("E.", 1), ("A", 0), ("F", 0), ("e", 0), ("E\n", 0), (None, 0)]:
        assert mmsufinal.mmsufinal_process_results(doc, [response])["accuracy"]["score"] == expected


def test_mmar_audio_preserves_extension_and_rejects_missing_or_empty_cache(tmp_path):
    audio = mmarfinal.mmarfinal_doc_to_audio
    doc = _doc("mmar")
    cache_file = tmp_path / "sample.mp3"
    kwargs = {"audio_cache_dir": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="sample"):
        audio(doc, kwargs)
    cache_file.touch()
    with pytest.raises(FileNotFoundError, match="sample"):
        audio(doc, kwargs)
    cache_file.write_bytes(b"audio fixture")
    assert audio(doc, kwargs) == [{"type": "audio", "url": str(cache_file)}]


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".flac", ".ogg"])
def test_mmsu_reuses_legacy_wav_without_reading_parquet(extension, tmp_path, monkeypatch):
    doc = {**_doc("mmsu"), "audio_path": f"MMSU/audio/sample{extension}"}
    cache_file = tmp_path / "sample.wav"
    sf.write(cache_file, np.zeros(160), 16000)
    before = cache_file.stat().st_mtime_ns

    def unexpected_read(*args):
        pytest.fail("A nonempty WAV cache must not read Parquet")

    monkeypatch.setattr(mmsufinal, "_parquet_audio_index", unexpected_read)
    assert mmsufinal.mmsufinal_doc_to_audio(doc, {"audio_cache_dir": str(tmp_path)}) == [
        {"type": "audio", "url": str(cache_file)}
    ]
    assert cache_file.stat().st_mtime_ns == before


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".flac", ".ogg"])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_mmsu_regenerates_wav_from_parquet_by_id(extension, empty_cache, tmp_path, monkeypatch):
    monkeypatch.setenv("MMSUFINAL_DATASET_DIR", str(tmp_path))
    doc = {**_doc("mmsu"), "audio_path": f"MMSU/audio/sample{extension}"}
    encoded = BytesIO()
    sf.write(encoded, np.full((4800, 2), 0.25), 48000, format="WAV")
    (tmp_path / "data").mkdir()
    rows = [
        {"id": "unrelated", "audio": {"path": "unrelated.wav", "bytes": b"must not decode"}},
        {"id": "sample", "audio": {"path": f"sample{extension}", "bytes": encoded.getvalue()}},
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data/test.parquet", row_group_size=1)
    cache_file = tmp_path / "_audio_cache/sample.wav"
    if empty_cache:
        cache_file.parent.mkdir()
        cache_file.touch()
    assert mmsufinal.mmsufinal_doc_to_audio(doc) == [{"type": "audio", "url": str(cache_file)}]
    samples, rate = sf.read(cache_file)
    assert rate == 16000
    assert samples.shape == (1600,)
    np.testing.assert_allclose(samples[50:-50], 0.25, atol=1e-4)


@pytest.mark.slow
def test_mmsu_regenerates_reported_missing_release_audio(tmp_path):
    if not (mmsufinal._dataset_dir() / "data").is_dir():
        pytest.skip("Local MMSU Parquet release is not installed")
    samples = [
        ("fb53c62d-17aa-4d73-bd9f-9de04b284146", ".mp3", 8360),
        ("17a73f88-ad0a-448f-9d73-a0500caba323", ".mp3", 26633),
        ("46ec7422-ed93-429c-b863-2a425b55834e", ".flac", 360829),
        ("622d87b8-37cb-4b87-8955-58648f51140d", ".ogg", 64380),
    ]
    for uuid, extension, frames in samples:
        sample_id = f"non_verbal_sound_detection_{uuid}"
        doc = {**_doc("mmsu"), "id": sample_id, "audio_path": f"MMSU/audio/{sample_id}{extension}"}
        result = mmsufinal.mmsufinal_doc_to_audio(doc, {"audio_cache_dir": str(tmp_path)})
        info = sf.info(result[0]["url"])
        assert Path(result[0]["url"]).name == f"{sample_id}.wav"
        assert (info.samplerate, info.channels, info.frames) == (16000, 1, frames)


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
def test_metadata_join_keeps_jsonl_answer_and_id_order(name, final, legacy, metric, tmp_path, monkeypatch):
    monkeypatch.setenv(f"{name.upper()}FINAL_DATASET_DIR", str(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    breakdown = {"modality": "speech", "subcategory": "meaning"} if name == "mmar" else {"sub_category": "meaning", "task_name": "speech"}
    rows = [{"id": sample_id, "answer": "A", "question": "old question", **breakdown} for sample_id in ["second", "sample"]]
    pq.write_table(pa.Table.from_pylist(rows), data_dir / "test.parquet")
    docs = Dataset.from_list([_doc(name), {**_doc(name), "id": "second"}])
    process = getattr(final, f"{name}final_process_docs")
    enriched = process(docs)
    assert list(enriched["id"]) == ["sample", "second"]
    for doc in enriched:
        assert doc["question_text"] == "Which sound is heard?"
        assert doc["answer"] == "rooster"
        assert getattr(final, f"{name}final_doc_to_target")(doc) == "C"
        assert all(doc[key] == value for key, value in breakdown.items())
    with pytest.raises(ValueError, match="duplicate JSONL id"):
        process(Dataset.from_list([_doc(name), _doc(name)]))
    with pytest.raises(ValueError, match="missing Parquet metadata"):
        process(Dataset.from_list([{**_doc(name), "id": "missing"}]))


@pytest.mark.parametrize("name,final,legacy,metric", CASES)
def test_aggregation_counts_invalid_responses_in_denominator(name, final, legacy, metric):
    doc = _doc(name)
    score = getattr(final, f"{name}final_process_results")
    aggregate = getattr(final, f"{name}final_aggregate_results")
    assert aggregate is getattr(legacy, f"{name}_aggregate_results")
    assert aggregate([score(doc, [value])[metric] for value in ["C", "A", ""]]) == pytest.approx(100 / 3)
    assert aggregate([]) == 0.0


def test_tasks_are_registered_and_keep_generation_contracts():
    manager = TaskManager()
    for name, final, legacy, metric in CASES:
        task_name = f"{name}final"
        assert task_name in manager.all_tasks
        config = config_utils.load_yaml_config(str(TASK_ROOT / task_name / f"{task_name}.yaml"))
        old_path = TASK_ROOT / name / ("mmar.yaml" if name == "mmar" else "_default_template_yaml")
        old = config_utils.load_yaml_config(str(old_path))
        assert config["generation_kwargs"] == old["generation_kwargs"]
        assert config["metric_list"][0]["metric"] == metric
        assert config["output_type"] == "generate_until"
        # Generate-only tasks do not need doc_to_choice: framework code indexes
        # choice[0], which would be unsafe for MMAR's empty option strings.
        assert "doc_to_choice" not in config
        assert "doc_to_messages" not in config


@pytest.mark.slow
@pytest.mark.parametrize("name,final,legacy,metric", CASES)
def test_local_release_through_real_task_loader(name, final, legacy, metric, tmp_path, monkeypatch):
    """Exercise all real records without model inference; skip absent local data."""
    task_name = f"{name}final"
    config = config_utils.load_yaml_config(str(TASK_ROOT / task_name / f"{task_name}.yaml"))
    jsonl = Path(config["dataset_kwargs"]["data_files"]["test"])
    if not jsonl.is_file() or not (final._dataset_dir() / "data").is_dir():
        pytest.skip("Local AudioMCQ JSONL/Parquet release is not installed")
    config["dataset_kwargs"]["cache_dir"] = str(tmp_path / "datasets")
    if name == "mmsu":
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        for source in (final._dataset_dir() / "_audio_cache").iterdir():
            if source.is_file():
                target = cache_dir / source.with_suffix(".wav").name
                if not target.exists():
                    target.symlink_to(source.resolve())
        config["lmms_eval_specific_kwargs"]["default"]["audio_cache_dir"] = str(cache_dir)
    monkeypatch.setenv("LMMS_EVAL_DATASETS_CACHE", str(tmp_path / "datasets"))
    raw = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    task = ConfigurableTask(config=config)
    docs = task.test_docs()
    assert len(docs) == len(raw) == (1000 if name == "mmar" else 5000)
    scores = []
    for doc, source in zip(docs, raw):
        assert all(doc[key] == value for key, value in source.items())
        target = task.doc_to_target(doc)
        prompt = task.doc_to_text(doc)
        assert f"{target}. {source['answer']}" in prompt
        for i in range(len(source["multi_choice"])):
            letter = ascii_uppercase[i]
            assert task.process_results(doc, [letter])[metric]["score"] == float(letter == target)
        assert task.process_results(doc, [""])[metric]["score"] == 0.0
        scores.append(task.process_results(doc, [target])[metric])
        path = Path(task.doc_to_visual(doc)[0]["url"])
        expected_name = Path(source["audio_path"])
        if name == "mmsu":
            expected_name = expected_name.with_suffix(".wav")
        assert path.name == expected_name.name
        assert path.is_file() and path.stat().st_size > 0
    assert task.aggregation()[metric](scores) == 100.0
    if name == "mmsu":
        for line in [1645, 1679, 1728, 2069]:
            assert task.doc_to_target(docs[line - 1]) == "E"
        for line, target in [(4119, "B"), (4739, "C"), (4742, "A")]:
            assert task.doc_to_target(docs[line - 1]) == target
    else:
        assert "F. " in task.doc_to_text(docs[894])
        for line, target in [(44, "B"), (518, "D"), (758, "D"), (837, "D")]:
            assert task.doc_to_target(docs[line - 1]) == target

    # The chat path must preserve the same user prompt/media without injecting
    # the system prompt used by the AudioMCQ-new variants.
    chat = ConfigurableMessagesTask(config=copy.deepcopy(config))
    message = chat.doc_to_messages(chat.test_docs()[0])
    assert [turn["role"] for turn in message] == ["user"]
    assert message[0]["content"][-1] == {"type": "text", "text": task.doc_to_text(docs[0])}
