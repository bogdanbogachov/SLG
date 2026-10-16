"""Answer-clustered paired bootstrap summaries from per-question score JSONL files."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 53, 64)
MODELS = {
    "SLG": "slg_finetuned_router",
    "8B": "finetuned_3_1_8b",
    "1B": "finetuned_3_2_1b",
    "RAG": "rag",
}
METRICS = ("BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "EM", "METEOR", "Entailment", "AI_Expert")


def _load(path):
    with path.open(encoding="utf-8") as f:
        return {r["qid"]: r for line in f if line.strip() for r in [json.loads(line)] if r.get("qid")}


def _holm(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _bootstrap_seed_differences(seed_rows, metric, rng, replicates=10000):
    # A sample is a mapping qid -> (gold answer, metric difference).
    grouped = []
    for records in seed_rows:
        by_answer = {}
        for answer, delta in records:
            by_answer.setdefault(answer, []).append(delta)
        groups = list(by_answer.values())
        grouped.append(groups)
    observed = sum(sum(v for group in groups for v in group) / max(sum(map(len, groups)), 1) for groups in grouped) / len(grouped)
    samples = []
    for _ in range(replicates):
        seed_means = []
        for groups in grouped:
            picked = [rng.choice(groups) for _ in range(len(groups))]
            vals = [v for group in picked for v in group]
            seed_means.append(sum(vals) / max(len(vals), 1))
        samples.append(sum(seed_means) / len(seed_means))
    samples.sort()
    lo, hi = samples[int(.025 * replicates)], samples[min(replicates - 1, int(.975 * replicates))]
    p = min(1.0, 2 * min(sum(x <= 0 for x in samples) / replicates, sum(x >= 0 for x in samples) / replicates))
    return observed, lo, hi, p


def _seed_mean(records):
    return sum(delta for _answer, delta in records) / max(len(records), 1)


def _tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _load_score_files(dataset, seeds=SEEDS, results_root=None):
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    cells = {}
    for seed in seeds:
        score_dir = results_root / dataset / f"seed_{seed}" / "scores"
        cells[seed] = {model: _load(score_dir / f"{stem}.jsonl") for model, stem in MODELS.items()}
        qid_sets = [set(records) for records in cells[seed].values()]
        if any(qids != qid_sets[0] for qids in qid_sets[1:]):
            raise ValueError(f"Models do not share the same qids for {dataset} seed {seed}")
    return cells


def run(dataset, replicates=10000, rng_seed=20261015, seeds=SEEDS,
        results_root=None, output_dir=None, include_inflation_contrast=True):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one seed is required for paired bootstrap statistics.")
    cells = _load_score_files(dataset, seeds=seeds, results_root=results_root)
    comparisons = (("SLG", "8B"), ("SLG", "1B"), ("SLG", "RAG"))
    rows = []
    for left, right in comparisons:
        for metric in METRICS:
            seed_differences = []
            for seed in seeds:
                a, b = cells[seed][left], cells[seed][right]
                common = sorted(set(a) & set(b))
                if not common:
                    raise ValueError(f"No paired qids for {dataset} seed {seed}: {left} vs {right}")
                seed_differences.append([
                    (a[qid].get("gold_answer", ""), float(a[qid][metric]) - float(b[qid][metric]))
                    for qid in common
                ])
            mean, lo, hi, p = _bootstrap_seed_differences(seed_differences, metric, random.Random(rng_seed), replicates)
            seed_means = [_seed_mean(records) for records in seed_differences]
            rows.append({"dataset": dataset, "comparison": f"{left}_vs_{right}", "metric": metric,
                         "mean_delta": mean, "ci95_low": lo, "ci95_high": hi, "p_two_sided": p,
                         "seed_sd_sample": statistics.stdev(seed_means) if len(seed_means) > 1 else "",
                         **{f"seed_{seed}_delta": value for seed, value in zip(seeds, seed_means)},
                         "equivalent_within_0.01": (lo > -0.01 and hi < 0.01) if (left, right, metric) in (("SLG", "8B", "ROUGE-L"), ("SLG", "8B", "AI_Expert")) else ""})
    if include_inflation_contrast and dataset == "s_avia_100" and len(seeds) == len(SEEDS):
        low_cells = _load_score_files("s_avia_25", seeds=seeds, results_root=results_root)
        for baseline in ("8B", "1B"):
            for metric in METRICS:
                seed_differences = []
                for seed in seeds:
                    high_slg, low_slg = cells[seed]["SLG"], low_cells[seed]["SLG"]
                    high_base, low_base = cells[seed][baseline], low_cells[seed][baseline]
                    common = set(high_slg) & set(low_slg) & set(high_base) & set(low_base)
                    if common != set(high_slg) or common != set(low_slg) or common != set(high_base) or common != set(low_base):
                        raise ValueError(f"Inflation-level qids differ for seed {seed}, cannot compute paired DiD")
                    seed_differences.append([
                        (high_slg[qid].get("gold_answer", ""),
                         (float(high_slg[qid][metric]) - float(low_slg[qid][metric]))
                         - (float(high_base[qid][metric]) - float(low_base[qid][metric])))
                        for qid in sorted(common)
                    ])
                mean, lo, hi, p = _bootstrap_seed_differences(seed_differences, metric, random.Random(rng_seed + 1000), replicates)
                seed_means = [_seed_mean(records) for records in seed_differences]
                rows.append({"dataset": dataset, "comparison": f"SLG_vs_{baseline}_inflation_DiD_100v25", "metric": metric,
                             "mean_delta": mean, "ci95_low": lo, "ci95_high": hi, "p_two_sided": p,
                             "seed_sd_sample": statistics.stdev(seed_means) if len(seed_means) > 1 else "",
                             **{f"seed_{seed}_delta": value for seed, value in zip(seeds, seed_means)},
                             "equivalent_within_0.01": ""})
    adjusted = _holm([r["p_two_sided"] for r in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
    out = Path(output_dir) if output_dir is not None else ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{dataset}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tex = [r"\begin{tabular}{llrrrr}", r"Comparison & Metric & Mean $\Delta$ & 95\% CI & $p$ & Holm $p$ \\ \hline"]
    for r in rows:
        tex.append(f"{_tex_escape(r['comparison'])} & {_tex_escape(r['metric'])} & {r['mean_delta']:.3f} & [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] & {r['p_two_sided']:.4g} & {r['p_holm']:.4g} \\")
    tex.append(r"\end{tabular}")
    (out / f"{dataset}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--replicates", type=int, default=10000)
    args = p.parse_args()
    print(run(args.dataset, args.replicates))


if __name__ == "__main__":
    main()
