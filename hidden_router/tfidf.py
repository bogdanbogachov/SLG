"""Lexical baseline for the hidden-state router: the same Mahalanobis classifier on TF-IDF features.

Question it answers: do the LLM hidden states carry routing information beyond word statistics?

Per cell: TF-IDF (word 1-2 grams, sublinear tf) and TruncatedSVD to --svd_dim dimensions are fitted on the
TRAINING questions only; the shrinkage alpha is chosen on the same stratified validation slice as the
hidden-state router (MahalanobisRouter.fit, one "layer"); test top-k is reported. CPU only, no model weights.

Outputs: results/<ds>/seed_<s>/hidden_router/tfidf_baseline.json, results/hidden_router/tfidf_summary.{csv,md}
"""
from __future__ import annotations

import csv
import json
from typing import Sequence

import numpy as np

from logging_config import logger

from hidden_router.cells import OLD_SLG, RESULTS, Cell, expert_id_from_title
from hidden_router.probe import _ms
from hidden_router.router import MahalanobisRouter, _topk_hits

DEFAULT_SVD_DIM = 768


def tfidf_features(train_questions: Sequence[str], test_questions: Sequence[str], svd_dim: int, seed: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode")
    Xtr = vectorizer.fit_transform(train_questions)
    Xte = vectorizer.transform(test_questions)
    dim = min(svd_dim, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Ztr = svd.fit_transform(Xtr).astype(np.float32)
    Zte = svd.transform(Xte).astype(np.float32)
    return Ztr[:, None, :], Zte[:, None, :], {"vocab": Xtr.shape[1], "svd_dim": dim,
                                             "explained_variance": float(svd.explained_variance_ratio_.sum())}


def _hidden_reference(cell: Cell) -> dict:
    """New-router test top-1 and the layer-0 (token embeddings) diagnostic, if probe.py wrote them."""
    out = {}
    routes_path = cell.router_dir / "test_routes.json"
    if routes_path.is_file():
        routes = json.loads(routes_path.read_text(encoding="utf-8"))
        out["hidden_top1"] = sum(r["selected_expert"] == r["expected_expert"] for r in routes.values()) / max(1, len(routes))
    per_layer_path = cell.router_dir / "per_layer_test.json"
    if per_layer_path.is_file():
        per_layer = json.loads(per_layer_path.read_text(encoding="utf-8"))
        out["hidden_layer0_top1"] = per_layer[0]["test_top1"]
    return out


def baseline_cell(cell: Cell, *, svd_dim: int = DEFAULT_SVD_DIM) -> dict:
    from hidden_router.pipeline import fit_kwargs

    train, test = cell.train(), cell.test()
    train_labels = [expert_id_from_title(r["title"]) for r in train]
    test_labels = [expert_id_from_title(r["title"]) for r in test]
    Ztr, Zte, info = tfidf_features([r["question"] for r in train], [r["question"] for r in test], svd_dim, cell.seed)
    router = MahalanobisRouter.fit(Ztr, train_labels, seed=cell.seed, **fit_kwargs())
    import torch
    name_id = {n: i for i, n in enumerate(router.labels)}
    y = torch.tensor([name_id.get(g, -1) for g in test_labels])
    keep = y >= 0
    hits = _topk_hits(router.scores(Zte)[keep], y[keep])
    n_keep = int(keep.sum())
    scale = n_keep / max(1, len(test_labels))  # a test expert unseen in training counts as a miss
    old = (cell.routing_report(OLD_SLG) or {}).get("summary", {}).get("top_k_accuracy", {})
    row = {"dataset": cell.dataset, "seed": cell.seed, "experts": len(router.labels), "test_questions": len(test),
           **info, "alpha": router.alpha, "tfidf_val_top1": router.selection["chosen"]["val_top1"],
           "tfidf_top1": hits[1] * scale, "tfidf_top3": hits[3] * scale, "tfidf_top5": hits[5] * scale,
           "old_top1": old.get("top_1", {}).get("accuracy"), **_hidden_reference(cell)}
    cell.router_dir.mkdir(parents=True, exist_ok=True)
    (cell.router_dir / "tfidf_baseline.json").write_text(
        json.dumps({**row, "grid": router.selection["grid"]}, indent=2), encoding="utf-8")
    return row


def run(datasets: Sequence[str], seeds: Sequence[int], *, svd_dim: int = DEFAULT_SVD_DIM) -> str:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            cell = Cell(dataset, seed)
            if not cell.exists():
                logger.warning("Skipping %s: split or slg/ experts missing", cell)
                continue
            row = baseline_cell(cell, svd_dim=svd_dim)
            logger.info("%s TF-IDF baseline: top1=%.4f (hidden %s, old %s)", cell, row["tfidf_top1"],
                        row.get("hidden_top1"), row.get("old_top1"))
            rows.append(row)
    if not rows:
        raise SystemExit("No cells with splits and trained experts found.")
    out_dir = RESULTS / "hidden_router"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "tfidf_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cols = [("old_top1", "Old router top-1"), ("tfidf_top1", "TF-IDF top-1"), ("tfidf_top5", "TF-IDF top-5"),
            ("hidden_layer0_top1", "Hidden L0 top-1"), ("hidden_top1", "Hidden (chosen layer) top-1"),
            ("explained_variance", "SVD var.")]
    lines = ["| Dataset | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        group = [r for r in rows if r["dataset"] == ds]
        lines.append(f"| {ds} | " + " | ".join(_ms([r.get(k) for r in group]) for k, _ in cols) + " |")
    lines.append(f"\nSame Mahalanobis classifier and alpha selection on TF-IDF + SVD({svd_dim}) features fitted on "
                 "training questions only. Hidden L0 = token-embedding layer (diagnostic from probe).")
    md = "\n".join(lines) + "\n"
    (out_dir / "tfidf_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return str(out_dir / "tfidf_summary.csv")
