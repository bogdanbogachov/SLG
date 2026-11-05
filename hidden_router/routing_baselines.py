"""Extra routing baselines on the SAME cached hidden-state features (CPU, minutes; no generation):

- centroid_cosine   : nearest expert centroid, cosine similarity (no covariance; "prototype" router)
- centroid_euclid   : nearest expert centroid, Euclidean (= Mahalanobis with identity covariance)
- knn10_cosine      : 10-nearest-neighbour vote over training questions, cosine
- logreg_probe      : multinomial logistic-regression probe (TRAINED, L2, lbfgs) on z-scored features
- mahalanobis       : the router already fitted by probe.py (reported for reference)
- tfidf_centroid    : nearest centroid on TF-IDF (word 1-2 grams) of the questions, cosine

All baselines use the layer the Mahalanobis router selected on the training validation slice (no test tuning).
Run after `python -m hidden_router probe` (needs results/hidden_router/features):
    python -m hidden_router.routing_baselines [--datasets a,b] [--seeds 42,53,64]
Writes results/hidden_router/routing_baselines.{csv,md}.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict

import numpy as np

from hidden_router.cells import DATASETS, RESULTS, SEEDS, Cell, expert_id_from_title
from hidden_router.features import FeatureStore
from hidden_router.probe import FEATURES_DIR
from hidden_router.router import MahalanobisRouter

METHODS = ("centroid_cosine", "centroid_euclid", "knn10_cosine", "logreg_probe", "tfidf_centroid", "mahalanobis")


def _norm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def _topk(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    gold_score = scores[np.arange(len(gold)), gold]
    rank = (scores > gold_score[:, None]).sum(1)
    return float((rank < k).mean())


def evaluate_cell(cell: Cell, store: FeatureStore) -> dict:
    train, test = cell.train(), cell.test()
    router = MahalanobisRouter.load(cell.router_dir)
    names = router.labels
    idx = {n: i for i, n in enumerate(names)}
    ytr = np.array([idx[expert_id_from_title(r["title"])] for r in train])
    yte_all = [idx.get(expert_id_from_title(r["title"]), -1) for r in test]
    keep = np.array([y >= 0 for y in yte_all])
    yte = np.array(yte_all)[keep]
    Xtr = store.get([r["qid"] for r in train])[:, router.layer, :].astype(np.float32)
    Xte = store.get([r["qid"] for r in test])[:, router.layer, :].astype(np.float32)[keep]
    C = len(names)
    out = {}

    cent = np.stack([Xtr[ytr == c].mean(0) for c in range(C)])
    out["centroid_cosine"] = _norm(Xte) @ _norm(cent).T
    d2 = (Xte ** 2).sum(1)[:, None] - 2 * Xte @ cent.T + (cent ** 2).sum(1)[None, :]
    out["centroid_euclid"] = -d2

    sims = _norm(Xte) @ _norm(Xtr).T
    nn = np.argpartition(-sims, kth=min(9, sims.shape[1] - 1), axis=1)[:, :10]
    votes = np.zeros((len(Xte), C))
    for i, row in enumerate(nn):
        for j in row:
            votes[i, ytr[j]] += 1 + 1e-3 * sims[i, j]
    out["knn10_cosine"] = votes

    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit((Xtr - mu) / sd, ytr)
    proba = np.full((len(Xte), C), -np.inf)
    proba[:, clf.classes_] = clf.predict_log_proba((Xte - mu) / sd)
    out["logreg_probe"] = proba

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode")
    Ttr = vec.fit_transform([r["question"] for r in train])
    Tte = vec.transform([r["question"] for r in np.array(test, dtype=object)[keep]])
    tcent = np.vstack([np.asarray(Ttr[ytr == c].mean(0)) for c in range(C)])
    out["tfidf_centroid"] = (Tte @ _norm(tcent).T) if not hasattr(Tte, "toarray") else Tte.toarray() @ _norm(tcent).T

    out["mahalanobis"] = router.scores(Xte).numpy()

    scale = keep.mean()  # unseen test experts count as misses
    row = {"dataset": cell.dataset, "seed": cell.seed, "experts": C, "layer": router.layer}
    for m in METHODS:
        row[f"{m}_top1"] = _topk(out[m], yte, 1) * scale
        row[f"{m}_top5"] = _topk(out[m], yte, 5) * scale
    return row


def run(datasets, seeds) -> str:
    store = FeatureStore(FEATURES_DIR)
    rows = []
    for ds in datasets:
        for s in seeds:
            cell = Cell(ds, s)
            if not MahalanobisRouter.exists(cell.router_dir):
                print(f"skip {cell}: no fitted router")
                continue
            row = evaluate_cell(cell, store)
            rows.append(row)
            print(cell, " ".join(f"{m}={row[m + '_top1']:.3f}" for m in METHODS), flush=True)
    out = RESULTS / "hidden_router"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "routing_baselines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    lines = ["| Dataset | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
    for ds, group in by.items():
        cells = []
        for m in METHODS:
            v = [g[f"{m}_top1"] for g in group]
            cells.append(f"{statistics.mean(v):.3f}" + (f" ±{statistics.stdev(v):.3f}" if len(v) > 1 else ""))
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n\nTop-1 on the test split, mean ± SD over seeds; all methods use the Mahalanobis router's validation-selected layer.\n"
    (out / "routing_baselines.md").write_text(md, encoding="utf-8")
    print(md)
    return str(out / "routing_baselines.csv")


def main():
    p = argparse.ArgumentParser(prog="python -m hidden_router.routing_baselines")
    p.add_argument("--datasets", default="l_avia,s_avia,cqad20,cqad20_fine,cqad20_answer")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = p.parse_args()
    run([d for d in a.datasets.split(",") if d], [int(s) for s in a.seeds.split(",") if s])


if __name__ == "__main__":
    main()
