"""Post-hoc metrics for the SLG ablation experiments (#3 and #4).

Everything here is deterministic post-processing over a finished run's outputs
(``answers/<label>/slg.json`` + ``slg_diagnostics/``) and the test set's
ground-truth expert labels — no models, no API calls, fully automatic.

The correctness signal is **routing correctness**: a question's ground-truth
expert is the topic split it came from (``slug(title)``), and a route is correct
when the first chosen expert matches it. This is label-free and is exactly the
quantity the competence router (A) is trying to improve, which makes it the
right axis for both the online learning curve (#3) and the risk--coverage
analysis (#4). An external answer-quality score can be substituted via
``correctness`` if desired.

Produces, per run:

* ``routing_curve``  — cumulative first-route accuracy vs. #questions processed.
  Plot the full run against its ``no_competence`` ablation to show A learns (#3).
* ``risk_coverage``  — selective accuracy as coverage shrinks, ordering answered
  questions by verifier confidence. Shows abstention (C) keeps the better
  answers (#4).
* ``summary``        — coverage, overall and selective routing accuracy, status
  counts.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

from config import CONFIG


def slug_title(title: str) -> str:
    """Ground-truth expert id for a title — mirrors split_qa_pairs_by_title."""
    return (title or "").replace(" ", "_").replace("/", "_").replace("\n", "_").lower()


def _load(run_dir: str):
    with open(os.path.join(run_dir, "slg.json"), "r", encoding="utf-8") as f:
        rows = json.load(f)
    critic_path = os.path.join(run_dir, "slg_diagnostics", "critic_log.json")
    critic = None
    if os.path.isfile(critic_path):
        with open(critic_path, "r", encoding="utf-8") as f:
            critic = json.load(f)
    return rows, critic


def _accepted_confidence(attempts: Optional[List[dict]]) -> float:
    """Confidence of the answer that was returned (max over passed attempts)."""
    if not attempts:
        return 0.0
    passed = [a.get("confidence", 0.0) for a in attempts if a.get("passed")]
    return float(max(passed)) if passed else 0.0


def compute(run_dir: str, correctness: Optional[Sequence[float]] = None) -> Dict:
    """Compute routing-curve, risk--coverage and summary metrics for one run.

    ``correctness`` optionally overrides the default routing-correctness signal
    with an external per-row score in ``[0, 1]`` (e.g. semantic similarity).
    """
    rows, critic = _load(run_dir)
    n = len(rows)

    correct: List[float] = []
    for i, row in enumerate(rows):
        if correctness is not None:
            correct.append(float(correctness[i]))
        else:
            chosen = (row.get("experts") or [None])[0]
            correct.append(1.0 if chosen == slug_title(row.get("title")) else 0.0)

    status = [row.get("status") for row in rows]
    resolved = [i for i in range(n) if status[i] == "resolved"]

    # (#3) cumulative first-route accuracy in processing order (== index order
    # for the first routing round, which is where the online signal accrues).
    routing_curve = []
    running = 0.0
    for k, c in enumerate(correct, start=1):
        running += c
        routing_curve.append({"n": k, "routing_accuracy": round(running / k, 4)})

    # (#4) risk--coverage: order answered (resolved) questions by confidence,
    # admit them high-to-low, and track accuracy of the admitted set.
    conf = [
        _accepted_confidence(critic[i]) if critic and i < len(critic) else 0.0
        for i in range(n)
    ]
    ordered = sorted(resolved, key=lambda i: conf[i], reverse=True)
    risk_coverage = []
    acc_sum = 0.0
    for k, i in enumerate(ordered, start=1):
        acc_sum += correct[i]
        risk_coverage.append({
            "coverage": round(k / n, 4),
            "selective_accuracy": round(acc_sum / k, 4),
            "confidence": round(conf[i], 4),
        })

    status_counts: Dict[str, int] = {}
    for s in status:
        status_counts[s] = status_counts.get(s, 0) + 1

    summary = {
        "n": n,
        "coverage": round(len(resolved) / n, 4) if n else 0.0,
        "routing_accuracy_overall": round(sum(correct) / n, 4) if n else 0.0,
        "selective_routing_accuracy": (
            round(sum(correct[i] for i in resolved) / len(resolved), 4) if resolved else 0.0
        ),
        "status_counts": status_counts,
    }
    return {"summary": summary, "routing_curve": routing_curve, "risk_coverage": risk_coverage}


def run(label: Optional[str] = None) -> Dict:
    """Compute metrics for a run and write them next to its answers.

    ``label`` is the answers sub-directory (e.g. ``my_exp__no_competence``);
    defaults to the configured experiment (the full run).
    """
    from utils.path_utils import get_answers_root
    label = label or CONFIG["experiment"]
    run_dir = os.path.join(get_answers_root(CONFIG["experiment"]), label)
    metrics = compute(run_dir)
    out_path = os.path.join(run_dir, "slg_diagnostics", "selective_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
