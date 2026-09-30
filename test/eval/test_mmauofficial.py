"""Check the MMAU adapter against the supplied official evaluator, not a copy."""

import contextlib
import copy
import importlib.util
import io
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
import yaml

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval/tasks/mmauofficial"
_spec = importlib.util.spec_from_file_location("mmauofficial_utils", TASK_DIR / "utils.py")
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)


@pytest.fixture(scope="module")
def official_dir():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "AudioMCQevaluator"
        if (candidate / "evaluation.py").is_file():
            return candidate
    pytest.skip("The supplied AudioMCQevaluator/evaluation.py oracle is unavailable")


@pytest.fixture(scope="module")
def official_match(official_dir):
    return runpy.run_path(str(official_dir / "evaluation.py"))["string_match"]


@pytest.fixture(scope="module")
def mini_docs(official_dir):
    return json.loads((official_dir / "mmau-test-mini.json").read_text())


@pytest.fixture(scope="module")
def full_test_dir(official_dir):
    directory = official_dir.parent.parent / "MMAU-v05.15.25/test"
    if not (directory / "test.parquet").is_file():
        pytest.skip("The supplied MMAU-v05.15.25 full test split is unavailable")
    return directory


@pytest.fixture(scope="module")
def full_test_docs(full_test_dir, tmp_path_factory):
    return datasets.load_dataset(
        "parquet",
        data_files={"test": str(full_test_dir / "test.parquet")},
        split="test",
        cache_dir=str(tmp_path_factory.mktemp("mmau-full-test-dataset")),
    )


def _doc():
    return {
        "id": "sample-id",
        "audio_id": "./test-mini-audios/clip-file.wav",
        "question": "Identify the speaker.",
        "choices": ["Man", "Woman", "Child", "Robot"],
        "answer": "Man",
        "dataset": "source-name",
        "task": "sound",
        "split": "test-mini",
        "category": "Reasoning",
        "sub-category": "Acoustic Source Inference",
        "difficulty": "medium",
    }


def _official_cli(path, official_dir, monkeypatch):
    stdout = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["evaluation.py", "--input", str(path)])
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            namespace = runpy.run_path(str(official_dir / "evaluation.py"), run_name="__main__")
    return namespace, stdout.getvalue()


def _breakdowns(text):
    """Read printed metrics; all expected metric values come from the oracle."""
    metrics = {"Task-wise": {}, "Difficulty-wise": {}, "Sub-category-wise": {}}
    section = None
    for line in text.splitlines():
        for name in metrics:
            if f"{name} Accuracy:" in line:
                section = name
        if "Total Accuracy:" in line:
            continue
        match = re.fullmatch(r"\s*(.*?)\s*:\s*([\d.]+)% over (\d+) samples\s*", line)
        if section and match:
            label, accuracy, count = match.groups()
            metrics[section][label.strip()] = (float(accuracy), int(count))
    return metrics


@pytest.mark.parametrize(
    "prediction",
    [
        "Man",
        " MAN! \n",
        "A",
        "<answer>A</answer>",
        "<answer>Man</answer>",
        "Woman <answer>Man</answer>",
        "<think>Woman</think><answer>Man</answer>",
        "Man " + "explanation " * 70 + "Woman",
        "",
    ],
)
def test_raw_predictions_match_official_without_postprocessing(prediction, official_match):
    doc = _doc()
    original = copy.deepcopy(doc)
    result = utils.mmauofficial_process_results(doc, [prediction])
    expected = int(official_match(doc["answer"], prediction, doc["choices"]))

    assert utils.string_match(doc["answer"], prediction, doc["choices"]) == bool(expected)
    for key in ("mmauofficial_accuracy", "mmauofficial_submission"):
        assert result[key] == {**doc, "model_output": prediction, "match": expected}
    assert doc == original


def test_missing_predictions_are_skipped_but_empty_predictions_count(official_match):
    doc = _doc()
    missing = utils.mmauofficial_process_results(doc, [])
    assert missing["mmauofficial_accuracy"] == doc
    assert missing["mmauofficial_submission"] == doc

    responses = [[], ["Man"], ["Woman"], [""]]
    results = [utils.mmauofficial_process_results(doc, value)["mmauofficial_accuracy"] for value in responses]
    # This catches rounding, exclusion of empty responses, and inclusion of missing responses.
    assert utils.mmauofficial_aggregate_results(results) == (1 / 3) * 100
    with pytest.raises(ZeroDivisionError):
        utils.mmauofficial_aggregate_results([missing["mmauofficial_accuracy"]])

    # The adapter must not silently replace the official rule with a blank-GT guard.
    doc["answer"] = ""
    raw = "Unrelated output"
    scored = utils.mmauofficial_process_results(doc, [raw])["mmauofficial_accuracy"]
    assert scored["match"] == int(official_match("", raw, doc["choices"]))


def test_true_answer_and_choices_are_preserved_verbatim(official_match):
    doc = _doc()
    doc["answer"] = "  Man  "
    doc["choices"][0] = doc["answer"]
    assert utils.mmauofficial_doc_to_target(doc) == doc["answer"]
    result = utils.mmauofficial_process_results(doc, ["Man"])["mmauofficial_accuracy"]
    assert result["answer"] == doc["answer"]
    assert result["choices"] == doc["choices"]
    assert result["match"] == int(official_match(doc["answer"], "Man", doc["choices"]))


def test_prompt_supports_real_option_counts_and_does_not_leak_ground_truth(mini_docs):
    representatives = {len(doc["choices"]): doc for doc in mini_docs}
    assert set(representatives) == {2, 4, 5, 8}
    for doc in representatives.values():
        prompt = utils.mmauofficial_doc_to_text(doc)
        assert doc["question"] in prompt
        assert all(choice in prompt for choice in doc["choices"])
        assert "<answer>" not in prompt
        hidden = {
            **doc,
            "answer": "reference-only-gold",
            "thinking": "reference-only-reasoning",
            "explanation": "reference-only-explanation",
            "category": "reference-only-category",
            "difficulty": "reference-only-difficulty",
        }
        assert utils.mmauofficial_doc_to_text(hidden) == prompt


def test_audio_and_messages_use_manifest_filename_and_configured_directory(tmp_path):
    doc = _doc()
    audio = tmp_path / "clip-file.wav"
    audio.write_bytes(b"audio-path fixture; decoding belongs to the model adapter")
    kwargs = {"audio_dir": str(tmp_path)}
    expected_audio = {"type": "audio", "url": str(audio)}
    assert utils.mmauofficial_doc_to_audio(doc, kwargs) == [expected_audio]
    assert utils.mmauofficial_doc_to_messages(doc, kwargs) == [
        {
            "role": "user",
            "content": [expected_audio, {"type": "text", "text": utils.mmauofficial_doc_to_text(doc)}],
        }
    ]


def test_real_mini_loader_changes_only_category_representation(official_dir, mini_docs, tmp_path):
    loaded = datasets.load_dataset(
        str(TASK_DIR / "mmauofficial_dataset.py"),
        data_files={"test": str(official_dir / "mmau-test-mini.json")},
        split="test",
        cache_dir=str(tmp_path / "datasets"),
        trust_remote_code=True,
    )
    assert len(loaded) == len(mini_docs) == 1000
    assert sum(isinstance(doc["category"], list) for doc in mini_docs) == 13
    for source, actual in zip(mini_docs, loaded):
        expected = dict(source)
        if isinstance(expected["category"], list):
            expected["category"] = expected["category"][0]
        assert actual == expected
        audio = utils.mmauofficial_doc_to_audio(actual)
        assert audio == [{"type": "audio", "url": str(official_dir / "test-mini-audios" / Path(source["audio_id"]).name)}]
        assert Path(audio[0]["url"]).is_file()


def test_full_mini_export_and_aggregates_match_official_cli(mini_docs, official_dir, official_match, tmp_path, monkeypatch):
    scored = []
    submissions = []
    for index, doc in enumerate(mini_docs):
        wrong = next(choice for choice in doc["choices"] if choice != doc["answer"])
        candidates = [
            [doc["answer"]],
            [doc["answer"].upper()],
            [wrong],
            [""],
            [],
            ["A"],
            [f"<answer>{doc['answer']}</answer>"],
            [f"{wrong} <answer>{doc['answer']}</answer>"],
            [doc["answer"] + " explanation" * 70 + " " + wrong],
        ]
        response = candidates[index % len(candidates)]
        result = utils.mmauofficial_process_results(doc, response)
        scored.append(result["mmauofficial_accuracy"])
        submissions.append(result["mmauofficial_submission"])
        if response:
            assert scored[-1]["match"] == int(official_match(doc["answer"], response[0], doc["choices"]))
            assert submissions[-1]["model_output"] == response[0]
        else:
            assert "model_output" not in submissions[-1]

    utils.mmauofficial_aggregate_submission(submissions, SimpleNamespace(output_path=str(tmp_path)))
    export = tmp_path / "submissions/mmauofficial_test_mini_results.json"
    assert json.loads(export.read_text()) == submissions
    official, official_stdout = _official_cli(export, official_dir, monkeypatch)

    log_lines = []
    monkeypatch.setattr(utils, "eval_logger", SimpleNamespace(info=log_lines.append))
    actual = utils.mmauofficial_aggregate_results(scored)
    assert actual == (official["corr"] / official["total"]) * 100
    assert _breakdowns("\n".join(log_lines)) == _breakdowns(official_stdout)
    assert set(_breakdowns(official_stdout)["Task-wise"]) == {"sound", "music", "speech"}
    assert set(_breakdowns(official_stdout)["Difficulty-wise"]) == {"easy", "medium", "hard"}
    assert _breakdowns(official_stdout)["Sub-category-wise"]
    assert [row["match"] for row in scored if "model_output" in row] == [row["match"] for row in official["new_data"]]


@pytest.mark.parametrize(
    "responses",
    [
        [],
        [""],
        ["A"],
        ["Woman <answer>Man</answer>"],
        ["\n" + "explanation " * 70 + "<answer>Man</answer>\n"],
    ],
)
def test_full_test_predictions_remain_raw_and_never_call_matcher(responses, monkeypatch):
    def unexpected_match(*args, **kwargs):
        pytest.fail("The hidden-answer test split must not call the local matcher")

    monkeypatch.setattr(utils, "string_match", unexpected_match)
    doc = {**_doc(), "answer": "", "split": "test"}
    original = copy.deepcopy(doc)
    result = utils.mmauofficial_process_test_results(doc, responses)
    assert result == {
        "mmauofficial_test_submission": {
            "id": doc["id"],
            "model_prediction": responses[0] if responses else "",
        }
    }
    assert doc == original


def test_real_full_test_has_9000_unique_ids_hidden_answers_and_resolvable_audio(full_test_docs, full_test_dir):
    assert len(full_test_docs) == 9000
    assert len(set(full_test_docs["id"])) == 9000
    assert set(full_test_docs["answer"]) == {""}
    assert Counter(full_test_docs["task"]) == {"sound": 3000, "music": 3000, "speech": 3000}
    kwargs = {"audio_dir": str(full_test_dir / "test-audios")}
    for doc in full_test_docs:
        expected = full_test_dir / "test-audios" / Path(doc["audio_id"]).name
        assert utils.mmauofficial_doc_to_audio(doc, kwargs) == [{"type": "audio", "url": str(expected)}]
        assert expected.is_file()


def test_full_test_export_keeps_all_9000_ids_and_raw_predictions(full_test_docs, tmp_path, monkeypatch):
    def unexpected_match(*args, **kwargs):
        pytest.fail("Full-test submission must not score empty public ground truth")

    monkeypatch.setattr(utils, "string_match", unexpected_match)
    submissions = []
    expected = []
    for index, doc in enumerate(full_test_docs):
        variants = [
            [doc["choices"][0]],
            ["A"],
            [f"<answer>{doc['choices'][0]}</answer>"],
            [""],
            [],
            ["\n" + "reasoning " * 70 + doc["choices"][-1] + "\n"],
        ]
        responses = variants[index % len(variants)]
        result = utils.mmauofficial_process_test_results(doc, responses)
        assert set(result) == {"mmauofficial_test_submission"}
        submissions.append(result["mmauofficial_test_submission"])
        expected.append({"id": doc["id"], "model_prediction": responses[0] if responses else ""})

    utils.mmauofficial_aggregate_test_submission(submissions, SimpleNamespace(output_path=str(tmp_path)))
    export = tmp_path / "submissions/mmauofficial_test_submission.json"
    saved = json.loads(export.read_text())
    assert saved == expected
    assert len(saved) == len({row["id"] for row in saved}) == 9000
    assert [row["id"] for row in saved] == full_test_docs["id"]
    assert all(set(row) == {"id", "model_prediction"} for row in saved)


def test_task_group_configures_both_splits_and_scores_only_mini(official_dir, full_test_dir):
    class TaskLoader(yaml.SafeLoader):
        pass

    TaskLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))

    def load(name):
        return yaml.load((TASK_DIR / name).read_text(), Loader=TaskLoader)

    group = load("mmauofficial.yaml")
    assert group == {"group": "mmauofficial", "task": ["mmauofficial_test_mini", "mmauofficial_test"]}
    template = load("_default_template_yaml")
    mini = load("mmauofficial_test_mini.yaml")
    full = load("mmauofficial_test.yaml")
    assert mini["task"] == "mmauofficial_test_mini"
    assert Path(mini["dataset_kwargs"]["data_files"]["test"]) == official_dir / "mmau-test-mini.json"
    assert Path(mini["lmms_eval_specific_kwargs"]["default"]["audio_dir"]) == official_dir / "test-mini-audios"
    assert {metric["metric"] for metric in mini["metric_list"]} == {"mmauofficial_accuracy", "mmauofficial_submission"}
    assert mini["process_results"] == "utils.mmauofficial_process_results"

    assert full["task"] == "mmauofficial_test"
    assert full["dataset_path"] == "parquet"
    assert Path(full["dataset_kwargs"]["data_files"]["test"]) == full_test_dir / "test.parquet"
    assert Path(full["lmms_eval_specific_kwargs"]["default"]["audio_dir"]) == full_test_dir / "test-audios"
    assert full["test_split"] == "test"
    assert full["process_results"] == "utils.mmauofficial_process_test_results"
    assert [metric["metric"] for metric in full["metric_list"]] == ["mmauofficial_test_submission"]
    assert full["metric_list"][0]["aggregation"] == "utils.mmauofficial_aggregate_test_submission"

    for split in (mini, full):
        assert split["include"] == "_default_template_yaml"
        config = {**template, **split}
        assert config["output_type"] == "generate_until"
        assert config["generation_kwargs"] == {"max_new_tokens": 256, "do_sample": False}
        assert config.get("filter_list") is None
        assert config.get("reasoning_tags") is None
