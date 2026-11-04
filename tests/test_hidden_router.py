"""Tests for the hidden-state (Mahalanobis) router package. Run: python -m pytest tests/test_hidden_router.py
(or `python tests/test_hidden_router.py` without pytest). No model weights or GPU needed."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hidden_router import cells as cells_mod  # noqa: E402
from hidden_router import probe as probe_mod  # noqa: E402
from hidden_router.assemble import assemble_cell  # noqa: E402
from hidden_router.features import FeatureStore  # noqa: E402
from hidden_router.router import MahalanobisRouter, stratified_split  # noqa: E402

LAYERS, DIM, INFORMATIVE = 5, 16, 3


def _toy_features(labels, rng):
    """Only layer 3 separates the experts; other layers are pure noise."""
    names = sorted(set(labels))
    centres = {n: rng.normal(0, 3, DIM) for n in names}
    X = rng.normal(0, 1, (len(labels), LAYERS, DIM))
    for i, lab in enumerate(labels):
        X[i, INFORMATIVE] += centres[lab]
    return X.astype(np.float16)


def test_router_picks_informative_layer_and_routes():
    rng = np.random.default_rng(0)
    labels = [f"e{i % 6}" for i in range(600)]
    X = _toy_features(labels, rng)
    router = MahalanobisRouter.fit(X, labels, seed=1)
    assert router.layer == INFORMATIVE, router.selection["chosen"]
    test_labels = [f"e{i % 6}" for i in range(120)]
    rng2 = np.random.default_rng(0)  # same centres
    Xt = _toy_features(labels + test_labels, rng2)[len(labels):]
    ranked = router.rank(Xt)
    acc = np.mean([r[0]["expert"] == g for r, g in zip(ranked, test_labels)])
    assert acc > 0.95, acc
    assert abs(sum(c["probability"] for c in ranked[0]) - 1) < 1e-4
    with tempfile.TemporaryDirectory() as d:
        router.save(d)
        again = MahalanobisRouter.load(d)
        assert [c["expert"] for c in again.rank(Xt[:5])[0]] == [c["expert"] for c in ranked[0]]


def test_stratified_split_keeps_tiny_experts_in_fit():
    labels = ["a"] * 10 + ["b"] * 2
    fit, val = stratified_split(labels, 0.2, seed=0)
    assert len(val) == 2 and all(labels[i] == "a" for i in val)
    assert sorted(fit + val) == list(range(12))


class _FakeExtractor:
    """Question text 'qN about tX' -> features clustered by topic X (layer 3)."""

    def __init__(self):
        self.rng = np.random.default_rng(5)
        self.centres = {t: np.random.default_rng(100 + t).normal(0, 3, DIM) for t in range(3)}

    def encode(self, questions, log_every=0):
        X = self.rng.normal(0, 1, (len(questions), LAYERS, DIM))
        for i, q in enumerate(questions):
            X[i, INFORMATIVE] += self.centres[int(q.split(" t")[-1])]
        return X.astype(np.float16)

    def close(self):
        pass


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _toy_cell(root: Path):
    titles = ["Topic A", "Topic B", "Topic C"]
    experts = [cells_mod.expert_id_from_title(t) for t in titles]

    def row(i, t):
        return {"qid": f"q{i}", "title": titles[t], "chapter": "c", "question": f"q{i} about t{t}", "answer": f"ans {t}"}

    train = [row(i, i % 3) for i in range(90)]
    test = [row(1000 + i, i % 3) for i in range(30)]
    split = root / "data" / "toy" / "splits" / "seed_42"
    split.mkdir(parents=True)
    (split / "train.json").write_text(json.dumps(train))
    (split / "test.json").write_text(json.dumps(test))
    cell = root / "results" / "toy" / "seed_42"
    for e in experts:
        (cell / "slg" / e).mkdir(parents=True)
    # Old router: right on even questions, wrong (next expert) on odd ones. Wrong override: the expert after next.
    old_sel = {r["qid"]: experts[(i % 3) + (i % 2)] if (i % 3) + (i % 2) < 3 else experts[0] for i, r in enumerate(test)}
    wrong_sel = {r["qid"]: experts[(i % 3 + 2) % 3] for i, r in enumerate(test)}

    def answer(qid, expert):
        return f"answer of {expert} to {qid}"

    gold = {r["qid"]: experts[i % 3] for i, r in enumerate(test)}
    _write_jsonl(cell / "answers" / "slg_finetuned_router_oracle.jsonl", [{**r, "answer": answer(r["qid"], gold[r["qid"]])} for r in test])
    _write_jsonl(cell / "answers" / "slg_finetuned_router.jsonl", [{**r, "answer": answer(r["qid"], old_sel[r["qid"]])} for r in test])
    _write_jsonl(cell / "answers" / "slg_finetuned_router_wrong.jsonl", [{**r, "answer": answer(r["qid"], wrong_sel[r["qid"]])} for r in test])
    for stem, sel in (("slg_finetuned_router", old_sel), ("slg_finetuned_router_wrong", wrong_sel)):
        records = [{"index": i, "selected_expert": sel[r["qid"]], "routing_correct": sel[r["qid"]] == gold[r["qid"]],
                    "visited_experts": [sel[r["qid"]]]} for i, r in enumerate(test)]
        report = {"summary": {"top_k_accuracy": {"top_1": {"accuracy": 0.5}, "top_5": {"accuracy": 1.0}}}, "records": records}
        (cell / "routing_reports").mkdir(exist_ok=True)
        (cell / "routing_reports" / f"{stem}_routing_report.json").write_text(json.dumps(report))
    for stem, value in (("slg_finetuned_router_oracle", 0.9), ("slg_finetuned_router", 0.6), ("slg_finetuned_router_wrong", 0.1),
                        ("finetuned_3_2_1b", 0.5), ("finetuned_3_1_8b", 0.8), ("rag", 0.4)):
        _write_jsonl(cell / "scores" / f"{stem}.jsonl", [{"qid": r["qid"], "ROUGE-L": value, "AI_Expert": value} for r in test])
    return experts, gold, answer


def test_probe_and_assemble_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        experts, gold, answer = _toy_cell(root)
        saved = (cells_mod.RESULTS, cells_mod.DATA, probe_mod.RESULTS, probe_mod.FEATURES_DIR)
        cells_mod.RESULTS, cells_mod.DATA = root / "results", root / "data"
        probe_mod.RESULTS, probe_mod.FEATURES_DIR = root / "results", root / "results" / "hidden_router" / "features"
        try:
            cell = cells_mod.Cell("toy", 42)
            store = FeatureStore(probe_mod.FEATURES_DIR)
            store.ensure([(r["qid"], r["question"]) for r in cell.train() + cell.test()], extractor_factory=_FakeExtractor)
            probe_mod.run(["toy"], [42])
            summary = (root / "results" / "hidden_router" / "probe_summary.csv").read_text()
            assert "toy" in summary
            routes = json.loads((cell.router_dir / "test_routes.json").read_text())
            assert sum(r["selected_expert"] == r["expected_expert"] for r in routes.values()) >= 28
            # Force two wrong routes: one onto the old router's expert (reuse), one onto a fresh expert (generate).
            qa, qb = "q1001", "q1004"
            old_sel = cell.selected_by_qid("slg_finetuned_router")
            routes[qa]["selected_expert"] = old_sel[qa]
            wrong_sel = cell.selected_by_qid("slg_finetuned_router_wrong")
            fresh = next(e for e in experts if e not in (gold[qb], old_sel[qb], wrong_sel[qb]))
            routes[qb]["selected_expert"] = fresh
            (cell.router_dir / "test_routes.json").write_text(json.dumps(routes))

            calls = []

            def fake_generator(_cell):
                def gen(question, expert):
                    calls.append((question, expert))
                    return {"answer": "generated", "latency_s": 0.0, "peak_gpu_memory_mb": None}
                return gen

            report = assemble_cell(cell, generator_factory=fake_generator)
            out = cell.answers("slg_mahalanobis_router")
            assert len(out) == 30
            assert out[qa]["answer"] == answer(qa, old_sel[qa]) and out[qa]["answer_source"] == "old_router"
            assert out[qb]["answer"] == "generated" and calls == [(out[qb]["question"], fresh)]
            correct = [q for q, r in routes.items() if r["selected_expert"] == r["expected_expert"]]
            assert all(out[q]["answer"] == answer(q, gold[q]) and out[q]["answer_source"] == "oracle" for q in correct)
            assert report["answer_sources"]["generated"] == 1
            assert (cell.dir / ".done" / "infer_slg_mahalanobis").is_file()
            # Resume: nothing left to do, no new generation.
            assemble_cell(cell, generator_factory=fake_generator)
            assert len(calls) == 1 and len(cell.answers("slg_mahalanobis_router")) == 30
        finally:
            cells_mod.RESULTS, cells_mod.DATA, probe_mod.RESULTS, probe_mod.FEATURES_DIR = saved


def test_tfidf_baseline():
    from hidden_router import tfidf as tfidf_mod
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _toy_cell(root)
        saved = (cells_mod.RESULTS, cells_mod.DATA, tfidf_mod.RESULTS)
        cells_mod.RESULTS, cells_mod.DATA, tfidf_mod.RESULTS = root / "results", root / "data", root / "results"
        try:
            tfidf_mod.run(["toy"], [42], svd_dim=8)
            row = json.loads((root / "results" / "toy" / "seed_42" / "hidden_router" / "tfidf_baseline.json").read_text())
            assert row["tfidf_top1"] == 1.0, row  # topic token "tX" is in every question
            assert (root / "results" / "hidden_router" / "tfidf_summary.md").is_file()
        finally:
            cells_mod.RESULTS, cells_mod.DATA, tfidf_mod.RESULTS = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
