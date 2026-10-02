import importlib.util
from pathlib import Path

import yaml


TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval/tasks/mmarofficial"
_spec = importlib.util.spec_from_file_location("mmarofficial_utils", TASK_DIR / "utils.py")
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)


def _doc(answer="Open Grassland"):
    return {
        "id": "sample",
        "audio_path": "./audio/sample.wav",
        "question": "Where does this music originate?",
        "choices": [
            "Mountain Canyon",
            "Dense Forest",
            "Open Grassland",
            "City Streets",
            "",
        ],
        "answer": answer,
        "modality": "music",
        "category": "Cultural Layer",
        "sub-category": "Professional Knowledge and Reasoning",
        "thinking": "reference-only-reasoning",
        "cue": ["reference-only-cue"],
        "rubric": [{"scoring_point": "reference-only-rubric"}],
    }


def test_official_answer_and_choices_are_not_rewritten():
    doc = _doc(answer="Open Grassland ")
    doc["choices"][2] = doc["answer"]
    result = utils.mmarofficial_process_results(doc, ["Open Grassland"])[
        "mmarofficial_accuracy"
    ]

    assert utils.mmarofficial_doc_to_target(doc) == "Open Grassland "
    assert result["answer"] == doc["answer"]
    assert result["choices"] == doc["choices"]
    assert result["sub-category"] == doc["sub-category"]


def test_prompt_requests_answer_text_and_omits_padded_empty_choices():
    prompt = utils.mmarofficial_doc_to_text(_doc())

    assert "Respond with only the text of the correct option, not its letter." in prompt
    assert "C. Open Grassland" in prompt
    assert "E." not in prompt
    assert prompt.endswith("The best answer is:")
    assert "reference-only" not in prompt


def test_string_match_has_official_token_set_semantics():
    choices = ["Mountain Canyon", "Dense Forest", "Open Grassland", "City Streets"]

    assert utils.string_match(
        "Open Grassland", "The answer is Open Grassland", choices
    )
    assert not utils.string_match("Open Grassland", "C", choices)
    assert not utils.string_match(
        "Open Grassland", "Open Grassland, not Dense Forest", choices
    )
    # Original MMAR ignores word order; changing that would change the protocol.
    assert utils.string_match(
        "From far to near", "From near to far", ["From far to near", "From near to far"]
    )


def test_process_results_passes_raw_prediction_to_official_matcher():
    result = utils.mmarofficial_process_results(
        _doc(), ["The answer is Open Grassland"]
    )["mmarofficial_accuracy"]

    assert result["answer_prediction"] == "The answer is Open Grassland"
    assert result["answer"] == "Open Grassland"
    assert result["score"] == 1.0


def test_process_results_does_not_extract_answer_tags_or_discard_empty_outputs():
    prediction = "Not Dense Forest. <answer>Open Grassland</answer>"
    result = utils.mmarofficial_process_results(_doc(), [prediction])[
        "mmarofficial_accuracy"
    ]

    assert result["answer_prediction"] == prediction
    assert result["score"] == 0.0
    assert utils.mmarofficial_process_results(_doc(), [""])["mmarofficial_accuracy"]["score"] == 0.0


def test_aggregate_results_returns_percentage_scale():
    correct = utils.mmarofficial_process_results(_doc(), ["Open Grassland"])[
        "mmarofficial_accuracy"
    ]
    incorrect = utils.mmarofficial_process_results(_doc(), ["Dense Forest"])[
        "mmarofficial_accuracy"
    ]

    assert utils.mmarofficial_aggregate_results([correct, incorrect]) == 50.0


def test_yaml_reads_official_metadata_instead_of_mirror_labels():
    class TaskLoader(yaml.SafeLoader):
        pass

    TaskLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    config = yaml.load((TASK_DIR / "mmarofficial.yaml").read_text(), Loader=TaskLoader)

    assert config["dataset_path"] == "json"
    assert config["dataset_kwargs"]["data_files"]["test"].endswith(
        "/AudioMCQevaluator/MMAR-official/MMAR-meta.jsonl"
    )
    assert config["lmms_eval_specific_kwargs"]["default"]["audio_cache_dir"].endswith(
        "/mmar/_audio_cache"
    )
    assert config["metadata"]["version"] == 2.0


def test_audio_uses_official_multi_segment_filename(tmp_path):
    doc = _doc()
    doc["id"] = "BV19x411L7Ff_0-56_1-08"
    doc["audio_path"] = "./audio/BV19x411L7Ff_multi_segment.wav"
    audio_path = tmp_path / "BV19x411L7Ff_multi_segment.wav"
    audio_path.write_bytes(b"audio cache fixture")

    assert utils.mmarofficial_doc_to_audio(doc, {"audio_cache_dir": str(tmp_path)}) == [
        {"type": "audio", "url": str(audio_path)}
    ]
