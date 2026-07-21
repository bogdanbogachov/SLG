import copy
import json

import pytest

from exceptions import FileNotFoundError
import inference.slg as slg_module
from cli.parser import build_parser
from inference.slg import ROUTER_COSINE, ROUTER_FINETUNED, SmallLanguageGraph


def test_cli_accepts_router_choice():
    parser = build_parser()
    args = parser.parse_args(["--infer_slg", "True", "--router", "finetuned"])

    assert args.infer_slg is True
    assert args.router == "finetuned"


def test_finetuned_router_does_not_require_cosine_index(tmp_path, monkeypatch):
    config = copy.deepcopy(slg_module.CONFIG)
    config["paths"]["experiments"] = str(tmp_path / "experiments")
    config["paths"]["downloaded_models"] = str(tmp_path / "downloaded_models")
    config["paths"]["models"]["3_2_1b"] = "downloaded_3_2_1b"
    config["slg_formation"]["slg_dir"] = "slg"
    config["adapters"]["slg_router_3_2_1b"] = "slg_router_3_2_1b"
    monkeypatch.setattr(slg_module, "CONFIG", config)

    experiment_dir = tmp_path / "experiments" / "exp"
    (experiment_dir / "slg" / "expert_a").mkdir(parents=True)
    router_dir = experiment_dir / "slg_router_3_2_1b"
    router_dir.mkdir()
    with open(router_dir / "router_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": {"0": "expert_a"}}, f)

    graph = SmallLanguageGraph(
        experts_location="exp",
        experiment="exp",
        router_method=ROUTER_FINETUNED,
    )

    assert graph.router_method == ROUTER_FINETUNED
    assert graph.index_dir is None
    assert graph.slg_embeddings_by_expert == {}
    assert graph.slg_neighbors_by_expert == {}


def test_finetuned_router_fails_loudly_when_missing(tmp_path, monkeypatch):
    config = copy.deepcopy(slg_module.CONFIG)
    config["paths"]["experiments"] = str(tmp_path / "experiments")
    config["slg_formation"]["slg_dir"] = "slg"
    config["adapters"]["slg_router_3_2_1b"] = "slg_router_3_2_1b"
    monkeypatch.setattr(slg_module, "CONFIG", config)

    (tmp_path / "experiments" / "exp" / "slg" / "expert_a").mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="Fine-tuned SLG router adapter not found",
    ):
        SmallLanguageGraph(
            experts_location="exp",
            experiment="exp",
            router_method=ROUTER_FINETUNED,
        )


def test_finetuned_router_skips_confidence_neighbor_router():
    graph = object.__new__(SmallLanguageGraph)
    graph.router_method = ROUTER_FINETUNED

    assert graph._route_after_expert({"phase": "main"}) == "aggregator"


def test_cosine_router_keeps_confidence_neighbor_router():
    graph = object.__new__(SmallLanguageGraph)
    graph.router_method = ROUTER_COSINE

    assert graph._route_after_expert({"phase": "main"}) == "confidence_router"


def test_routing_record_marks_expected_expert_accuracy():
    graph = object.__new__(SmallLanguageGraph)
    graph.router_method = ROUTER_FINETUNED
    graph.expert_nodes = ["example_title"]

    record = graph._build_routing_record(
        0,
        {
            "chapter": "chapter",
            "title": "Example Title",
            "question": "question?",
        },
        {
            "selected_expert": "example_title",
            "visited_experts": ["example_title"],
            "answers": [
                {
                    "expert": "example_title",
                    "confidence": 0.9,
                    "answer": "answer",
                }
            ],
        },
    )

    assert record["expected_expert"] == "example_title"
    assert record["expected_expert_exists"] is True
    assert record["routing_correct"] is True
    assert record["selected_confidence"] == 0.9


def test_routing_report_summary_computes_accuracy():
    records = [
        {"expected_expert": "a", "expected_expert_exists": True, "routing_correct": True},
        {"expected_expert": "a", "expected_expert_exists": True, "routing_correct": False},
        {"expected_expert": "b", "expected_expert_exists": False, "routing_correct": False},
    ]

    summary = SmallLanguageGraph._routing_report_summary(records)

    assert summary["total"] == 3
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1 / 3
    assert summary["missing_expected_expert_count"] == 1
    assert summary["by_expected_expert"]["a"]["accuracy"] == 0.5
