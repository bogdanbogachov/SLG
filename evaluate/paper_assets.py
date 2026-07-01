"""Paper-ready aggregation of every SLG result into LaTeX tables and figures.

This is the single "make the paper assets" step. It reads whatever results
exist for an experiment and writes copy-paste-ready artifacts into a separate
folder (``paper_assets/<experiment>/``), so nothing in the paper has to be
hand-transcribed. It is pure CPU post-processing (json + matplotlib) — no
models, no API, no GPU — and every source is optional: missing inputs are
skipped with a log line instead of aborting.

Sources consumed
----------------
* ``experiments/<exp>/metrics.json``            answer-quality (``--evaluate``)
* ``experiments/<exp>__<ablation>/metrics.json`` per-ablation quality (optional)
* ``answers/<label>/slg_diagnostics/selective_metrics.json``  routing/selective
  behaviour per run (``--slg_metrics``); ``<label>`` is the full run and each
  leave-one-out ablation.
* ``answers/<exp>/slg_diagnostics/scalability.json``          distractor sweep

Artifacts produced (under ``paper_assets/<exp>/``)
-------------------------------------------------
``tables/``  LaTeX ``table`` floats (booktabs): main quality, ablation, scalability.
``figures/`` PDF (for the paper) + PNG (for quick preview): routing-learning
             curve, risk--coverage curve, scalability, ablation bars.
``README.md`` index describing each asset and how to ``\\input`` / ``\\includegraphics`` it.
"""

import json
import os
from typing import Dict, List, Optional

from config import CONFIG
from logging_config import logger

# Canonical leave-one-out order and the mechanism each run isolates.
_ABLATION_ORDER = ["full", "no_competence", "no_verifier", "no_abstention", "base"]
_ABLATION_LABEL = {
    "full": r"Full (A+B+C)",
    "no_competence": r"$-$A (no competence)",
    "no_verifier": r"$-$B (no verifier)",
    "no_abstention": r"$-$C (no abstention)",
    "base": r"Base (none)",
}
# Human-readable, LaTeX-safe names for known prediction-file stems.
_SYSTEM_LABEL = {
    "slg": "SLG (ours)",
    "finetuned_3_2_1b": "Fine-tuned 1B",
    "finetuned_3_1_8b": "Fine-tuned 8B",
    "baseline": "GPT-4.1",
    "gpt": "GPT-4.1",
    "rag": "RAG",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _load_json(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("paper_assets: could not read %s (%s).", path, e)
        return None


def _tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _system_name(stem: str) -> str:
    return _SYSTEM_LABEL.get(stem, _tex_escape(stem))


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("paper_assets: wrote %s", path)


def _wrap_table(body: str, caption: str, label: str, colspec: str, header: str) -> str:
    return (
        "% Requires \\usepackage{booktabs} in the preamble.\n"
        "\\begin{table}[t]\n  \\centering\n"
        f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{colspec}}}\n    \\toprule\n"
        f"    {header} \\\\\n    \\midrule\n"
        f"{body}"
        "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------- #
# source collection
# --------------------------------------------------------------------------- #
def _quality_rows(experiments_dir: str, experiment: str) -> "Dict[str, dict]":
    """{stem -> flat quality dict} from experiments/<exp>/metrics.json."""
    data = _load_json(os.path.join(experiments_dir, experiment, "metrics.json"))
    out: Dict[str, dict] = {}
    if not isinstance(data, list):
        return out
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for stem, m in entry.items():
            if not isinstance(m, dict) or "ROUGE" not in m:
                continue  # skip training-metric entries appended by --training_metrics
            rouge = m.get("ROUGE", {})
            out[stem] = {
                "BLEU": m.get("BLEU"),
                "ROUGE-1": rouge.get("rouge1"),
                "ROUGE-2": rouge.get("rouge2"),
                "ROUGE-L": rouge.get("rougeL"),
                "EM": m.get("Exact Match"),
                "METEOR": m.get("METEOR"),
                "Semantic": m.get("Entailment"),
                "AI-Expert": m.get("AI Expert"),
            }
    return out


def _ablation_summaries(answers_dir: str, experiment: str) -> "Dict[str, dict]":
    """{ablation_name -> {behaviour summary (+ quality if evaluated)}}."""
    experiments_dir = CONFIG["paths"]["experiments"]
    out: Dict[str, dict] = {}
    for name in _ABLATION_ORDER:
        label = experiment if name == "full" else f"{experiment}__{name}"
        sel = _load_json(
            os.path.join(answers_dir, label, "slg_diagnostics", "selective_metrics.json")
        )
        if not sel:
            continue
        summary = sel.get("summary", {})
        row = {
            "routing": summary.get("routing_accuracy_overall"),
            "selective": summary.get("selective_routing_accuracy"),
            "coverage": summary.get("coverage"),
            "n": summary.get("n"),
        }
        # Optional per-ablation answer quality, if --evaluate was run on it.
        q = _quality_rows(experiments_dir, label)
        slg_q = q.get("slg", {})
        row["Semantic"] = slg_q.get("Semantic")
        row["AI-Expert"] = slg_q.get("AI-Expert")
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
_QUALITY_COLS = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "EM", "METEOR", "Semantic", "AI-Expert"]


def _table_main_quality(rows: "Dict[str, dict]", out_dir: str) -> Optional[str]:
    if not rows:
        logger.info("paper_assets: no metrics.json quality data; skipping main table.")
        return None
    # Best (max) per column, for bolding.
    best = {}
    for c in _QUALITY_COLS:
        vals = [r[c] for r in rows.values() if r.get(c) is not None]
        best[c] = max(vals) if vals else None
    # SLG first, then the rest alphabetically.
    order = sorted(rows, key=lambda s: (s != "slg", s))
    lines = []
    for stem in order:
        cells = [_system_name(stem)]
        for c in _QUALITY_COLS:
            v = rows[stem].get(c)
            s = _fmt(v)
            if v is not None and best[c] is not None and abs(v - best[c]) < 1e-9:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append("    " + " & ".join(cells) + " \\\\\n")
    header = "System & " + " & ".join(_QUALITY_COLS)
    colspec = "l" + "r" * len(_QUALITY_COLS)
    tex = _wrap_table(
        "".join(lines),
        caption="Answer-quality comparison on the held-out test set (higher is better; "
                "best per column in bold).",
        label="tab:main-quality",
        colspec=colspec,
        header=header,
    )
    path = os.path.join(out_dir, "tables", "main_quality.tex")
    _write(path, tex)
    return path


def _table_ablation(abl: "Dict[str, dict]", out_dir: str) -> Optional[str]:
    if not abl:
        logger.info("paper_assets: no selective_metrics.json; skipping ablation table.")
        return None
    lines = []
    for name in _ABLATION_ORDER:
        if name not in abl:
            continue
        r = abl[name]
        cells = [
            _ABLATION_LABEL[name],
            _fmt(r.get("routing")),
            _fmt(r.get("selective")),
            _fmt(r.get("coverage")),
            _fmt(r.get("Semantic")),
            _fmt(r.get("AI-Expert")),
        ]
        lines.append("    " + " & ".join(cells) + " \\\\\n")
    header = ("Configuration & Routing acc. & Selective acc. & Coverage & "
              "Semantic & AI-Expert")
    tex = _wrap_table(
        "".join(lines),
        caption="Leave-one-out ablation of the three online mechanisms "
                "(A: competence router, B: domain verifier, C: calibrated abstention). "
                "Routing/selective accuracy and coverage are label-free; "
                "Semantic/AI-Expert require \\texttt{--evaluate} per run.",
        label="tab:ablation",
        colspec="lrrrrr",
        header=header,
    )
    path = os.path.join(out_dir, "tables", "ablation.tex")
    _write(path, tex)
    return path


def _table_scalability(scal: Optional[list], out_dir: str) -> Optional[str]:
    if not scal:
        logger.info("paper_assets: no scalability.json; skipping scalability table.")
        return None
    lines = []
    for r in scal:
        cells = [
            f"{r.get('n_experts')}",
            f"{r.get('n_core')} + {r.get('n_distractors')}",
            _fmt(r.get("latency_per_q_s"), 3),
            _fmt(r.get("routing_accuracy")),
            _fmt(r.get("coverage")),
        ]
        lines.append("    " + " & ".join(cells) + " \\\\\n")
    header = "Pool size & Core + distractor & Latency/q (s) & Routing acc. & Coverage"
    tex = _wrap_table(
        "".join(lines),
        caption="Scalability under distractor growth: the task is held fixed while "
                "irrelevant experts are added. Latency per question and routing accuracy "
                "are attributable to pool size alone.",
        label="tab:scalability",
        colspec="rlrrr",
        header=header,
    )
    path = os.path.join(out_dir, "tables", "scalability.tex")
    _write(path, tex)
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save_fig(fig, out_dir: str, name: str) -> str:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    pdf = os.path.join(fig_dir, f"{name}.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    logger.info("paper_assets: wrote %s (+ .png)", pdf)
    return pdf


def _fig_routing_curve(answers_dir: str, experiment: str, out_dir: str) -> Optional[str]:
    """Routing-learning curve: Full vs -A (no_competence) shows A learns online (#3)."""
    import matplotlib.pyplot as plt

    series = []
    for name in ("full", "no_competence"):
        label = experiment if name == "full" else f"{experiment}__{name}"
        sel = _load_json(
            os.path.join(answers_dir, label, "slg_diagnostics", "selective_metrics.json")
        )
        if sel and sel.get("routing_curve"):
            curve = sel["routing_curve"]
            series.append((_ABLATION_LABEL[name],
                           [p["n"] for p in curve],
                           [p["routing_accuracy"] for p in curve]))
    if not series:
        logger.info("paper_assets: no routing_curve data; skipping figure.")
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, xs, ys in series:
        ax.plot(xs, ys, label=name, linewidth=1.8)
    ax.set_xlabel("Questions processed")
    ax.set_ylabel("Cumulative routing accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    path = _save_fig(fig, out_dir, "routing_curve")
    plt.close(fig)
    return path


def _fig_risk_coverage(answers_dir: str, experiment: str, out_dir: str) -> Optional[str]:
    """Risk--coverage: selective accuracy vs coverage, Full vs -C (#4)."""
    import matplotlib.pyplot as plt

    series = []
    for name in ("full", "no_abstention"):
        label = experiment if name == "full" else f"{experiment}__{name}"
        sel = _load_json(
            os.path.join(answers_dir, label, "slg_diagnostics", "selective_metrics.json")
        )
        if sel and sel.get("risk_coverage"):
            rc = sel["risk_coverage"]
            series.append((_ABLATION_LABEL[name],
                           [p["coverage"] for p in rc],
                           [p["selective_accuracy"] for p in rc]))
    if not series:
        logger.info("paper_assets: no risk_coverage data; skipping figure.")
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, xs, ys in series:
        ax.plot(xs, ys, label=name, linewidth=1.8)
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Selective accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    path = _save_fig(fig, out_dir, "risk_coverage")
    plt.close(fig)
    return path


def _fig_scalability(scal: Optional[list], out_dir: str) -> Optional[str]:
    """Latency/q and routing accuracy vs pool size (twin axes) (#5)."""
    import matplotlib.pyplot as plt

    if not scal:
        logger.info("paper_assets: no scalability.json; skipping figure.")
        return None
    sizes = [r.get("n_experts") for r in scal]
    lat = [r.get("latency_per_q_s") for r in scal]
    acc = [r.get("routing_accuracy") for r in scal]
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    c1, c2 = "tab:blue", "tab:red"
    ax1.plot(sizes, lat, "o-", color=c1, linewidth=1.8)
    ax1.set_xlabel("Expert pool size")
    ax1.set_ylabel("Latency / question (s)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(sizes, acc, "s--", color=c2, linewidth=1.8)
    ax2.set_ylabel("Routing accuracy", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    path = _save_fig(fig, out_dir, "scalability")
    plt.close(fig)
    return path


def _fig_ablation_bar(abl: "Dict[str, dict]", out_dir: str) -> Optional[str]:
    """Grouped bars of routing/selective accuracy + coverage across ablations."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = [n for n in _ABLATION_ORDER if n in abl]
    if not names:
        return None
    metrics = [("routing", "Routing acc."), ("selective", "Selective acc."), ("coverage", "Coverage")]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for j, (key, lab) in enumerate(metrics):
        vals = [abl[n].get(key) or 0.0 for n in names]
        ax.bar(x + (j - 1) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([_ABLATION_LABEL[n].replace("$-$", "-") for n in names],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    path = _save_fig(fig, out_dir, "ablation_bar")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def _write_readme(out_dir: str, produced: "Dict[str, List[str]]") -> None:
    lines = [
        "# Paper-ready assets",
        "",
        "Auto-generated by `--paper_assets`. Copy tables with `\\input{}` and figures",
        "with `\\includegraphics{}`. Tables use `booktabs` (add `\\usepackage{booktabs}`).",
        "Regenerate any time — this step is pure CPU post-processing.",
        "",
        "## Tables (`tables/`)",
    ]
    tbl = {
        "main_quality.tex": "Answer-quality vs. baselines (needs `--evaluate`).",
        "ablation.tex": "Leave-one-out A/B/C ablation (needs `--slg_metrics`; quality cols need `--evaluate`).",
        "scalability.tex": "Distractor-growth scalability (needs `--slg_scalability`).",
    }
    for f in produced.get("tables", []):
        base = os.path.basename(f)
        lines.append(f"- `tables/{base}` — {tbl.get(base, '')} `\\input{{tables/{base[:-4]}}}`")
    lines += ["", "## Figures (`figures/`, PDF for paper + PNG preview)"]
    figd = {
        "routing_curve.pdf": "Online routing-learning curve, Full vs $-$A (mechanism A).",
        "risk_coverage.pdf": "Risk--coverage curve, Full vs $-$C (mechanism C).",
        "scalability.pdf": "Latency/q and routing accuracy vs pool size (mechanism-agnostic).",
        "ablation_bar.pdf": "Routing/selective accuracy and coverage across ablations.",
    }
    for f in produced.get("figures", []):
        base = os.path.basename(f)
        lines.append(f"- `figures/{base}` — {figd.get(base, '')}")
    lines.append("")
    _write(os.path.join(out_dir, "README.md"), "\n".join(lines))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build(experiment: Optional[str] = None) -> str:
    """Aggregate all results for ``experiment`` into paper_assets/<experiment>/."""
    experiment = experiment or CONFIG["experiment"]
    paths = CONFIG["paths"]
    experiments_dir = paths["experiments"]
    answers_dir = paths["answers"]
    out_dir = os.path.join(paths.get("paper_assets", "paper_assets"), experiment)

    quality = _quality_rows(experiments_dir, experiment)
    ablation = _ablation_summaries(answers_dir, experiment)
    scalability = _load_json(
        os.path.join(answers_dir, experiment, "slg_diagnostics", "scalability.json")
    )

    produced: Dict[str, List[str]] = {"tables": [], "figures": []}
    for p in (
        _table_main_quality(quality, out_dir),
        _table_ablation(ablation, out_dir),
        _table_scalability(scalability, out_dir),
    ):
        if p:
            produced["tables"].append(p)
    for p in (
        _fig_routing_curve(answers_dir, experiment, out_dir),
        _fig_risk_coverage(answers_dir, experiment, out_dir),
        _fig_scalability(scalability, out_dir),
        _fig_ablation_bar(ablation, out_dir),
    ):
        if p:
            produced["figures"].append(p)

    _write_readme(out_dir, produced)
    logger.info(
        "paper_assets: %d tables, %d figures -> %s",
        len(produced["tables"]), len(produced["figures"]), out_dir,
    )
    return out_dir
