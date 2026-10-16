"""Aggregate latency, memory, and training GPU-hour measurements for the paper."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20")
SEEDS = (42, 53, 64)
MODELS = {
    "1B": ("finetuned_3_2_1b", ("training_gpu_hours_finetuned_3_2_1b",), "~1B"),
    "8B": ("finetuned_3_1_8b", ("training_gpu_hours_finetuned_3_1_8b",), "~8B"),
    "SLG": ("slg_finetuned_router", ("training_gpu_hours_slg", "training_gpu_hours_slg_router_3_2_1b"), "~1B active expert"),
    "RAG": ("rag", (), "~8B"),
}


def _rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(dataset_names=DATASETS, seeds=SEEDS, results_root=None, output_dir=None):
    seeds = tuple(seeds)
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    records = []
    for dataset in dataset_names:
        for label, (stem, gpu_keys, params) in MODELS.items():
            latencies, peaks, train_gpu_hours = [], [], []
            for seed in seeds:
                cell = results_root / dataset / f"seed_{seed}"
                predictions = _rows(cell / "answers" / f"{stem}.jsonl")
                latencies.extend(float(r["latency_s"]) for r in predictions if r.get("latency_s") is not None)
                peaks.extend(float(r["peak_gpu_memory_mb"]) for r in predictions if r.get("peak_gpu_memory_mb") is not None)
                if gpu_keys:
                    seed_gpu_hours = 0.0
                    metrics_path = cell / "metrics.json"
                    if metrics_path.exists():
                        for entry in json.loads(metrics_path.read_text(encoding="utf-8")):
                            seed_gpu_hours += sum(float(entry[key]) for key in gpu_keys if key in entry)
                    if seed_gpu_hours:
                        train_gpu_hours.append(seed_gpu_hours)
            if not latencies:
                continue
            records.append({
                "dataset": dataset, "model": label, "parameters": params,
                "training_gpu_hours_mean": statistics.mean(train_gpu_hours) if train_gpu_hours else 0.0,
                "training_gpu_hours_sd": statistics.stdev(train_gpu_hours) if len(train_gpu_hours) > 1 else 0.0,
                "latency_s_mean": statistics.mean(latencies),
                "latency_s_median": statistics.median(latencies),
                "peak_gpu_memory_mb": max(peaks, default=0.0),
                "minimum_gpu_tested": "H100 3g.40gb (job.sh)",
            })
    out = Path(output_dir) if output_dir is not None else ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "deployment.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        tex = [r"\begin{tabular}{llrrrr}", r"Dataset & Model & Params & Train GPU-h & Latency (s) & Peak GPU MB \\ \hline"]
        tex.extend(f"{r['dataset']} & {r['model']} & {r['parameters']} & {r['training_gpu_hours_mean']:.2f} & {r['latency_s_mean']:.3f} & {r['peak_gpu_memory_mb']:.0f} " + r" \\" for r in records)
        tex.append(r"\end{tabular}")
        (out / "deployment.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return csv_path


if __name__ == "__main__":
    print(build())
