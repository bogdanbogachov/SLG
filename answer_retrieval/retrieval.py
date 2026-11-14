"""Stored-answer retrieval baseline versus SLG, scored as "was the correct procedure returned?".

Per cell (dataset x seed), using only that cell's training split as the answer store:

- retrieval_1nn : TF-IDF (word 1-2 grams, sublinear tf) over the training questions; each test question
                  returns the stored answer of its most similar training question (cosine). No training,
                  no generation.
- SLG           : the generated answers already scored in results/<ds>/seed_<s>/scores/
                  slg_mahalanobis_router.jsonl. Each generated answer is mapped to the stored training answer
                  it reproduces: the 3 most similar stored answers by TF-IDF cosine are shortlisted and the one
                  with the highest ROUGE-L F-measure against the generated text is taken. The mapping tolerates
                  small surface differences (spacing, punctuation) but a different procedure stays different.

Strict accuracy: 1 if the returned (retrieval) or reproduced (SLG) stored answer is the gold answer, else 0.
Near-identical passages of a different procedure therefore count as wrong. ROUGE-L of each system against
the gold answer is reported as well (same scorer as evaluate/evaluate.py: rougeL, stemming).

Writes results/answer_retrieval/{summary.csv, summary.md} and one per-question JSONL per cell.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "answer_retrieval"

DATASETS = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100",
            "cqad20", "cqad20_fine", "cqad20_answer")
SEEDS = (42, 53, 64)
SLG_SCORES = "slg_mahalanobis_router.jsonl"
SHORTLIST = 3


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return " ".join(text.split())


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def _rouge_l(scorer, reference: str, candidate: str) -> float:
    if _norm(reference) == _norm(candidate):
        return 1.0
    return scorer.score(target=reference, prediction=candidate)["rougeL"].fmeasure


def evaluate_cell(dataset: str, seed: int) -> dict:
    split = DATA / dataset / "splits" / f"seed_{seed}"
    train = _load_json(split / "train.json")
    test = _load_json(split / "test.json")
    slg = {r["qid"]: r for r in _load_jsonl(RESULTS / dataset / f"seed_{seed}" / "scores" / SLG_SCORES)}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Retrieval: nearest training question -> its stored answer.
    q_vec = _tfidf().fit(r["question"] for r in train)
    Xq_train = q_vec.transform(r["question"] for r in train)
    Xq_test = q_vec.transform(r["question"] for r in test)
    nn = np.asarray((Xq_test @ Xq_train.T).argmax(axis=1)).ravel()

    # Answer store for mapping SLG outputs: distinct training answers.
    store = sorted({r["answer"] for r in train})
    a_vec = _tfidf().fit(store)
    X_store = a_vec.transform(store)

    rows = []
    for i, r in enumerate(test):
        gold = r["answer"]
        retrieved = train[nn[i]]["answer"]
        s = slg.get(r["qid"])
        row = {"qid": r["qid"], "retrieval_correct": int(retrieved == gold),
               "retrieval_rougeL": _rouge_l(scorer, gold, retrieved)}
        if s is not None:
            pred = s["prediction"]
            sims = (a_vec.transform([pred]) @ X_store.T).toarray().ravel()
            cand = np.argsort(-sims)[:SHORTLIST]
            best = max(cand, key=lambda j: _rouge_l(scorer, store[j], pred))
            row.update({"slg_correct": int(store[best] == gold), "slg_rougeL": float(s["ROUGE-L"])})
        rows.append(row)

    out_dir = OUT / dataset / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    scored = [r for r in rows if "slg_correct" in r]
    return {"dataset": dataset, "seed": seed, "n_test": len(rows), "n_slg": len(scored),
            "retrieval_acc": statistics.mean(r["retrieval_correct"] for r in rows),
            "retrieval_rougeL": statistics.mean(r["retrieval_rougeL"] for r in rows),
            "slg_acc": statistics.mean(r["slg_correct"] for r in scored) if scored else float("nan"),
            "slg_rougeL": statistics.mean(r["slg_rougeL"] for r in scored) if scored else float("nan")}


def summarize(cells: list[dict]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ("retrieval_acc", "slg_acc", "retrieval_rougeL", "slg_rougeL")
    with (OUT / "per_cell.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)

    lines = ["| Dataset | Retrieval strict acc. | SLG strict acc. | Retrieval ROUGE-L | SLG ROUGE-L |",
             "|---|---|---|---|---|"]
    summary_rows = []
    for ds in dict.fromkeys(c["dataset"] for c in cells):
        cs = [c for c in cells if c["dataset"] == ds]
        stats = {}
        for k in keys:
            v = [c[k] for c in cs]
            stats[k] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
        summary_rows.append({"dataset": ds, "seeds": len(cs),
                             **{f"{k}_mean": stats[k][0] for k in keys},
                             **{f"{k}_sd": stats[k][1] for k in keys}})
        lines.append(f"| {ds} | " + " | ".join(f"{stats[k][0]:.3f} ±{stats[k][1]:.3f}" for k in keys) + " |")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    lines.append("")
    lines.append("Mean ± SD over seeds. Strict accuracy: the returned (retrieval) or reproduced (SLG) stored "
                 "answer is the gold answer.")
    md = "\n".join(lines)
    (OUT / "summary.md").write_text(md + "\n", encoding="utf-8")
    return md
