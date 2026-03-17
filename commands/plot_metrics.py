import os
import json
from typing import Dict, Any

import matplotlib.pyplot as plt

from config import CONFIG
from utils.path_utils import ensure_dir


def _extract_numeric_metrics(data: Any) -> Dict[str, float]:
    """
    Extract a flat mapping of metric_name -> numeric_value from a metrics.json structure.

    Supports:
    - Dict[str, number]
    - Dict[str, Any] where values are nested dicts containing numeric metrics
    - List[...] where each element is a dict (possibly {run_name: {metrics...}})
    """
    metrics: Dict[str, float] = {}

    if isinstance(data, dict):
        # Direct numeric metrics
        for k, v in data.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, dict):
                # Nested metrics (e.g. ROUGE sub-scores)
                for mk, mv in v.items():
                    if isinstance(mv, (int, float)):
                        metrics[f"{k}_{mk}"] = mv

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Common pattern: {"run_name": {"metric": value, ...}}
                for _, maybe_metrics in item.items():
                    if isinstance(maybe_metrics, dict):
                        for mk, mv in maybe_metrics.items():
                            if isinstance(mv, (int, float)):
                                metrics[mk] = mv
                            elif isinstance(mv, dict):
                                for sk, sv in mv.items():
                                    if isinstance(sv, (int, float)):
                                        metrics[f"{mk}_{sk}"] = sv
                    elif isinstance(maybe_metrics, (int, float)):
                        # Fallback if the value is directly numeric
                        metrics[_] = maybe_metrics

    return metrics


def plot_experiments_metrics() -> None:
    """
    Iterate over experiment run folders in CONFIG['paths']['experiments'],
    read each run's metrics.json, and create a line chart per metric.

    X-axis: run number (numeric folder name)
    Y-axis: metric value
    Only numeric run folders are considered, and the 'archive' folder is skipped.
    Charts are saved into CONFIG['paths']['charts'].
    """
    paths_config = CONFIG["paths"]
    files_config = CONFIG["files"]

    experiments_root = paths_config["experiments"]
    charts_dir = paths_config.get("charts", os.path.join(experiments_root, "charts"))
    metrics_filename = files_config["metrics"]

    runs = []

    experiments_root_abs = os.path.abspath(experiments_root)
    charts_dir_abs = os.path.abspath(charts_dir)

    ensure_dir(charts_dir_abs)

    if not os.path.isdir(experiments_root_abs):
        return

    for entry in os.listdir(experiments_root_abs):
        full_path = os.path.join(experiments_root_abs, entry)

        if not os.path.isdir(full_path):
            continue

        if entry.lower() == "archive":
            continue

        # Extract leading integer from folder name, e.g. "5firstrun" -> 5
        prefix_digits = ""
        for ch in entry:
            if ch.isdigit():
                prefix_digits += ch
            else:
                break

        if not prefix_digits:
            continue

        metrics_path = os.path.join(full_path, metrics_filename)
        if not os.path.isfile(metrics_path):
            continue

        try:
            with open(metrics_path, "r") as f:
                raw_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        numeric_metrics = _extract_numeric_metrics(raw_data)
        if not numeric_metrics:
            continue

        run_number = int(prefix_digits)

        runs.append((run_number, numeric_metrics))

    if not runs:
        return

    runs.sort(key=lambda x: x[0])

    # Collect all metric names across runs
    metric_names = set()
    for _, metrics in runs:
        metric_names.update(metrics.keys())

    for metric_name in metric_names:
        xs = []
        ys = []

        for run_number, metrics in runs:
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                xs.append(run_number)
                ys.append(value)

        # Need at least two points to make a meaningful line chart
        if len(xs) < 2:
            continue

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Run")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs run")
        plt.grid(True, linestyle="--", alpha=0.6)

        output_path = os.path.join(charts_dir_abs, f"{metric_name}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

