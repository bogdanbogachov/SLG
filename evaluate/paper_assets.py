"""Build paper tables from the retained per-question score files.

The generated quality table reports means and sample standard deviations across
the three fixed seeds. OpenAI is used only to produce E/AIE scores during
evaluation; this module reads the saved scores and makes no API calls.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20")
SEEDS = (42, 53, 64)
MODELS = {
    "1B": "finetuned_3_2_1b", "8B": "finetuned_3_1_8b",
    "SLG": "slg_finetuned_router", "RAG": "rag",
}
METRICS = ("BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "EM", "METEOR", "Entailment", "AI_Expert")


def _load(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _metric_summary(dataset, stem, metric, seeds=SEEDS, results_root=None):
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    seed_means = []
    for seed in seeds:
        path = results_root / dataset / f"seed_{seed}" / "scores" / f"{stem}.jsonl"
        rows = _load(path)
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            return None
        seed_means.append(statistics.mean(values))
    return statistics.mean(seed_means), statistics.stdev(seed_means) if len(seed_means) > 1 else None


def build(dataset_names=None, seeds=SEEDS, results_root=None, output_dir=None):
    """Write CSV and LaTeX quality tables from per-question records only."""
    datasets = tuple(dataset_names or DATASETS)
    seeds = tuple(seeds)
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    records = []
    for dataset in datasets:
        for model, stem in MODELS.items():
            row = {"dataset": dataset, "model": model}
            available = False
            for metric in METRICS:
                summary = _metric_summary(dataset, stem, metric, seeds=seeds, results_root=results_root)
                row[f"{metric}_mean"] = summary[0] if summary else ""
                row[f"{metric}_sd_sample"] = summary[1] if summary and summary[1] is not None else ""
                available |= summary is not None
            if available:
                records.append(row)

    out = Path(output_dir) if output_dir is not None else ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "paper_quality.csv"
    fields = ["dataset", "model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd_sample")]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    tex = [r"\begin{tabular}{ll" + "r" * (2 * len(METRICS)) + "}",
           r"Dataset & Model & " + " & ".join(_tex_escape(m) + r" mean & " + _tex_escape(m) + r" SD" for m in METRICS) + " \\\\",
           r"\hline"]
    for row in records:
        values = [_tex_escape(row["dataset"]), _tex_escape(row["model"])]
        for metric in METRICS:
            mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd_sample"]
            values.extend(("--" if mean == "" else f"{mean:.3f}", "--" if sd == "" else f"{sd:.3f}"))
        tex.append(" & ".join(values) + r" \\")
    tex.extend((r"\end{tabular}", "% SD is the sample standard deviation (ddof=1) over seeds 42, 53, and 64."))
    (out / "paper_quality.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return csv_path


if __name__ == "__main__":
    print(build())
