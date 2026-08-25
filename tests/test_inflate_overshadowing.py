import json

import pytest

from cli.parser import build_parser
from question_answer.inflate_overshadowing import inflate_qa_answers_with_file_inputs


def test_cli_accepts_inflation_percentage():
    parser = build_parser()
    args = parser.parse_args(["--inflate_overshadowing", "--inflation_percentage", "25"])

    assert args.inflate_overshadowing is True
    assert args.inflation_percentage == 25


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _selected_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    selected = {}
    for row in rows:
        answer = row["answer"]
        if "\n\n" not in answer:
            continue
        _prefix, original_answer = answer.split("\n\n", 1)
        selected.setdefault(row["title"], set()).add(original_answer)
    return selected


def test_partial_inflation_is_stratified_and_cumulative(tmp_path):
    qa_original = tmp_path / "qa_original.json"
    inflating_material = tmp_path / "inflating_material.json"
    qa_output = tmp_path / "qa.json"

    rows = []
    for title in ["title-a", "title-b"]:
        for answer_index in range(4):
            for duplicate_index in range(2):
                rows.append(
                    {
                        "chapter": title,
                        "title": title,
                        "question": f"q-{title}-{answer_index}-{duplicate_index}",
                        "answer": f"answer-{title}-{answer_index}",
                    }
                )

    _write_json(qa_original, rows)
    _write_json(
        inflating_material,
        [{"text": f"inflation-{index}"} for index in range(4)],
    )

    inflate_qa_answers_with_file_inputs(
        str(qa_original),
        str(inflating_material),
        str(qa_output),
        inflation_percentage=50,
        seed=42,
    )
    selected_50 = _selected_answers(qa_output)

    inflate_qa_answers_with_file_inputs(
        str(qa_original),
        str(inflating_material),
        str(qa_output),
        inflation_percentage=75,
        seed=42,
    )
    selected_75 = _selected_answers(qa_output)

    assert {title: len(answers) for title, answers in selected_50.items()} == {
        "title-a": 2,
        "title-b": 2,
    }
    assert {title: len(answers) for title, answers in selected_75.items()} == {
        "title-a": 3,
        "title-b": 3,
    }
    assert selected_50["title-a"] <= selected_75["title-a"]
    assert selected_50["title-b"] <= selected_75["title-b"]


def test_invalid_inflation_percentage_raises(tmp_path):
    qa_original = tmp_path / "qa_original.json"
    inflating_material = tmp_path / "inflating_material.json"
    qa_output = tmp_path / "qa.json"

    _write_json(qa_original, [])
    _write_json(inflating_material, [])

    with pytest.raises(ValueError, match="between 0 and 100"):
        inflate_qa_answers_with_file_inputs(
            str(qa_original),
            str(inflating_material),
            str(qa_output),
            inflation_percentage=125,
        )
