import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from cli.parser import build_parser
import evaluate.evaluate as eval_module


SCORES_LOW = {
    "BLEU": 0.25,
    "ROUGE": {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.5},
    "Exact Match": 0,
    "METEOR": 0.25,
    "Entailment": 0.75,
    "AI Expert": 0,
}

SCORES_HIGH = {
    "BLEU": 0.75,
    "ROUGE": {"rouge1": 1.0, "rouge2": 0.75, "rougeL": 1.0},
    "Exact Match": 1,
    "METEOR": 0.75,
    "Entailment": 1.0,
    "AI Expert": 1,
}


def test_cli_accepts_eval_workers():
    parser = build_parser()
    args = parser.parse_args(["--evaluate", "True", "--eval_workers", "4"])

    assert args.evaluate is True
    assert args.eval_workers == 4


def test_configured_eval_workers_is_bounded(monkeypatch):
    monkeypatch.setattr(eval_module, "CONFIG", {"evaluation": {"workers": "8"}})

    assert eval_module._configured_eval_workers(None, 3) == 3
    assert eval_module._configured_eval_workers(0, 3) == 1
    assert eval_module._configured_eval_workers("bad", 3) == 1


def test_evaluate_resumes_v2_checkpoint_and_scores_only_pending(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 2,
                "n_pairs": 3,
                "completed_scores": {"0": SCORES_LOW},
                "skipped_indices": [1],
                "partial_metrics": SCORES_LOW,
            },
            f,
        )

    calls = []

    def fake_score_eval_pair(task):
        i, _pred, _truth = task
        calls.append(i)
        return {"index": i, "matched": True, "scores": SCORES_HIGH}

    monkeypatch.setattr(eval_module, "_init_eval_worker", lambda _api_key: None)
    monkeypatch.setattr(eval_module, "_score_eval_pair", fake_score_eval_pair)
    monkeypatch.setattr(
        eval_module,
        "CONFIG",
        {"open_ai_api_key": "test-key", "evaluation": {"checkpoint_every": 1, "workers": 1}},
    )

    predictions = [
        {"question": "q0", "answer": "a", "chapter": "c", "title": "t"},
        {"question": "q1", "answer": "a", "chapter": "c", "title": "t"},
        {"question": "q2", "answer": "a", "chapter": "c", "title": "t"},
    ]
    ground_truth = [
        {"question": "q0", "answer": "a"},
        {"question": "q1", "answer": "a"},
        {"question": "q2", "answer": "a"},
    ]

    result = eval_module.evaluate(
        predictions,
        ground_truth,
        checkpoint_path=str(checkpoint_path),
        eval_workers=1,
    )

    assert calls == [2]
    assert result["BLEU"] == 0.5
    assert result["ROUGE"] == {"rouge1": 0.75, "rouge2": 0.5, "rougeL": 0.75}
    assert result["Exact Match"] == 0.5
    assert result["METEOR"] == 0.5
    assert result["Entailment"] == 0.875
    assert result["AI Expert"] == 0.5
    assert not checkpoint_path.exists()
