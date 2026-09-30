from lmms_eval.tasks.mmsu_official import utils


def _doc(answer="B"):
    return {
        "question": "Which option is correct?",
        "choice_a": "alpha",
        "choice_b": "bravo",
        "choice_c": "charlie",
        "choice_d": "delta",
        "answer": answer,
        "category": "Reasoning",
        "sub_category": "Linguistics",
        "task_name": "demo",
    }


def test_prompt_matches_official_inference_template():
    assert utils.mmsu_official_doc_to_text(_doc()) == (
        "Choose the most suitable answer from options A, B, C, and D. "
        "You must respond with only A, B, C, or D.\n\n"
        "Question: Which option is correct?\n\n"
        "A. alpha\nB. bravo\nC. charlie\nD. delta"
    )


def test_official_first_character_and_penultimate_character_parsing():
    direct = utils.mmsu_official_process_results(_doc(), ["B"])[
        "mmsu_official_accuracy"
    ]
    sentence = utils.mmsu_official_process_results(
        _doc(), ["The answer is B."]
    )["mmsu_official_accuracy"]

    assert direct["counted"] is True
    assert direct["prediction"] == "B"
    assert direct["score"] == 1.0
    assert sentence["counted"] is True
    assert sentence["prediction"] == "B"
    assert sentence["score"] == 1.0


def test_official_wrong_format_is_excluded_but_empty_is_counted():
    wrong_format = utils.mmsu_official_process_results(
        _doc(), ["Hmm, I would choose B. What do you think?"]
    )["mmsu_official_accuracy"]
    empty = utils.mmsu_official_process_results(_doc(), [""])[
        "mmsu_official_accuracy"
    ]

    assert wrong_format["counted"] is False
    assert wrong_format["score"] == 0.0
    assert empty["counted"] is True
    assert empty["score"] == 0.0


def test_answer_gt_is_derived_from_the_mirror_letter_label():
    result = utils.mmsu_official_process_results(_doc(answer="C"), ["C"])[
        "mmsu_official_accuracy"
    ]

    assert result["answer_gt"] == "charlie"
    assert result["score"] == 1.0


def test_aggregation_uses_the_official_counted_row_denominator():
    correct = utils.mmsu_official_process_results(_doc(), ["B"])[
        "mmsu_official_accuracy"
    ]
    incorrect = utils.mmsu_official_process_results(_doc(), ["A"])[
        "mmsu_official_accuracy"
    ]
    skipped = utils.mmsu_official_process_results(
        _doc(), ["I think the answer is B"]
    )["mmsu_official_accuracy"]

    assert utils.mmsu_official_aggregate_results(
        [correct, incorrect, skipped]
    ) == 0.5
