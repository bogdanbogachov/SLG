"""Routing tables for the paper (Mahalanobis hidden-state router vs the finetuned 1B classifier router).

Reads only files already written by the pipeline (no models, no API calls):
- results/<ds>/seed_<s>/routing_reports/slg_finetuned_router_routing_report.json   classifier router top-k
- results/<ds>/seed_<s>/hidden_router/test_routes.json        Mahalanobis router test routes (probe)
- results/<ds>/seed_<s>/hidden_router/selection.json          chosen layer/alpha + validation grid
- results/<ds>/seed_<s>/hidden_router/per_layer_test.json     per-layer test top-1/top-5 (diagnostic)
- results/<ds>/seed_<s>/hidden_router/tfidf_baseline.json     same classifier on TF-IDF+SVD (lexical baseline)
- results/<ds>/seed_<s>/hidden_router/assembly_report.json    where each SLG answer came from
- results/<ds>/seed_<s>/scores/*.jsonl                        for the routing-gap-closed ratio

Writes results/stats/routing.{csv,tex} (one row per dataset, mean ± sample SD over seeds) and
results/stats/routing_layers.csv (dataset x layer test top-1/top-5, mean over seeds; for the layer figure).
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20", "cqad20_fine", "cqad20_answer")
SEEDS = (42, 53, 64)
GAP_METRICS = ("ROUGE-L", "AI_Expert")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _score_mean(path: Path, metric: str):
    if not path.is_file():
        return None
    values = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v = json.loads(line).get(metric)
                if v is not None:
                    values.append(float(v))
    return statistics.mean(values) if values else None


def _cell(cell: Path) -> dict:
    row = {}
    hr = cell / "hidden_router"
    old = _json(cell / "routing_reports" / "slg_finetuned_router_routing_report.json") or {}
    topk = old.get("summary", {}).get("top_k_accuracy", {})
    for k in (1, 3, 5):
        if f"top_{k}" in topk:
            row[f"classifier_top{k}"] = topk[f"top_{k}"].get("accuracy")
    routes = _json(hr / "test_routes.json")
    if routes:
        n = len(routes)
        for k in (1, 3, 5):
            row[f"mahalanobis_top{k}"] = sum(1 for r in routes.values()
                                             if r.get("expected_rank") is not None and r["expected_rank"] <= k) / n
        row["experts"] = len({r["expected_expert"] for r in routes.values()})
    selection = _json(hr / "selection.json") or {}
    chosen = selection.get("chosen", {})
    row.update({"layer": chosen.get("layer"), "alpha": chosen.get("alpha"), "val_top1": chosen.get("val_top1")})
    per_layer = _json(hr / "per_layer_test.json")
    if per_layer:
        row["layer0_top1"] = per_layer[0]["test_top1"]
        best = max(per_layer, key=lambda p: p["test_top1"])
        row["best_test_layer"], row["best_test_layer_top1"] = best["layer"], best["test_top1"]
    tfidf = _json(hr / "tfidf_baseline.json") or {}
    for k in (1, 3, 5):
        if f"tfidf_top{k}" in tfidf:
            row[f"tfidf_top{k}"] = tfidf[f"tfidf_top{k}"]
    assembly = _json(hr / "assembly_report.json") or {}
    sources = assembly.get("answer_sources") or {}
    total = sum(sources.values())
    if total:
        row["share_generated"] = sources.get("generated", 0) / total
    scores = cell / "scores"
    for metric in GAP_METRICS:
        new = _score_mean(scores / "slg_mahalanobis_router.jsonl", metric)
        old_s = _score_mean(scores / "slg_finetuned_router.jsonl", metric)
        oracle = _score_mean(scores / "slg_finetuned_router_oracle.jsonl", metric)
        if None not in (new, old_s, oracle) and abs(oracle - old_s) >= 0.01:
            # Share of the classifier-router -> oracle gap closed by the Mahalanobis router (undefined when the
            # classifier is already within 0.01 of the oracle, e.g. cqad20).
            row[f"gap_closed_{metric}"] = (new - old_s) / (oracle - old_s)
    return row


def _ms(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


COLUMNS = ("experts", "classifier_top1", "classifier_top5", "tfidf_top1", "tfidf_top5", "layer0_top1",
           "mahalanobis_top1", "mahalanobis_top3", "mahalanobis_top5", "layer", "best_test_layer", "alpha",
           "share_generated", "gap_closed_ROUGE-L", "gap_closed_AI_Expert")


def build(dataset_names=None, seeds=SEEDS, results_root=None, output_dir=None):
    results_root = Path(results_root) if results_root is not None else ROOT / "results"
    datasets = tuple(dataset_names or DATASETS)
    records, layer_rows = [], []
    for dataset in datasets:
        cells = [results_root / dataset / f"seed_{s}" for s in seeds]
        rows = [_cell(c) for c in cells if c.is_dir()]
        if not any("mahalanobis_top1" in r for r in rows):
            continue
        record = {"dataset": dataset, "chosen_layers": ",".join(str(r.get("layer")) for r in rows)}
        for col in COLUMNS:
            mean, sd = _ms([r.get(col) for r in rows])
            record[f"{col}_mean"], record[f"{col}_sd"] = ("" if mean is None else mean), ("" if sd is None else sd)
        records.append(record)
        curves = [_json(c / "hidden_router" / "per_layer_test.json") for c in cells]
        curves = [c for c in curves if c]
        if curves:
            for layer in range(min(len(c) for c in curves)):
                layer_rows.append({"dataset": dataset, "layer": layer,
                                   "test_top1_mean": statistics.mean(c[layer]["test_top1"] for c in curves),
                                   "test_top5_mean": statistics.mean(c[layer]["test_top5"] for c in curves)})
    out = Path(output_dir) if output_dir is not None else ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "routing.csv"
    if not records:
        return None
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if layer_rows:
        with (out / "routing_layers.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)

    def cell(r, col, digits=3):
        mean, sd = r[f"{col}_mean"], r[f"{col}_sd"]
        if mean == "":
            return "--"
        return f"{mean:.{digits}f}" + ("" if sd == "" else f" $\\pm$ {sd:.{digits}f}")

    tex = [r"\begin{tabular}{lrlllllll}",
           r"Dataset & Experts & Classifier top-1 & TF-IDF top-1 & Hidden L0 top-1 & Mahalanobis top-1 & Mahalanobis top-5 & Layers & Gap closed (RL) \\ \hline"]
    for r in records:
        tex.append(" & ".join([r["dataset"].replace("_", r"\_"), cell(r, "experts", 0), cell(r, "classifier_top1"),
                               cell(r, "tfidf_top1"), cell(r, "layer0_top1"), cell(r, "mahalanobis_top1"),
                               cell(r, "mahalanobis_top5"), r["chosen_layers"], cell(r, "gap_closed_ROUGE-L", 2)]) + r" \\")
    tex.extend((r"\end{tabular}",
                "% Mean +- sample SD over seeds 42, 53, 64. Layers = layer chosen on the training validation slice per seed.",
                "% Gap closed = (SLG_mahalanobis - SLG_classifier) / (SLG_oracle - SLG_classifier), ROUGE-L."))
    (out / "routing.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return csv_path


if __name__ == "__main__":
    print(build())
