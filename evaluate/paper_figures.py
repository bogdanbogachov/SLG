"""Paper figures for the Mahalanobis-router version of SLG (reads only results/; writes PDF+PNG figures).

python -m evaluate.paper_figures --out ../SLG-AEI/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STATS = RES / "stats"

# Reference palette (dataviz skill, light mode): categorical slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURFACE, MID = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", MID, "#f2a0a0", "#b0302f"])
SEQ = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "legend.frameon": False,
})

LABEL = {"l_avia": "L-Avia", "s_avia": "S-Avia", "s_avia_25": "S-Avia-25", "s_avia_50": "S-Avia-50",
         "s_avia_75": "S-Avia-75", "s_avia_100": "S-Avia-100", "cqad20": "CQAD (11 experts)",
         "cqad20_fine": "CQAD-fine (24)", "cqad20_answer": "CQAD-answer (72)"}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _chosen_layers(ds):
    layers = []
    for s in (42, 53, 64):
        p = RES / ds / f"seed_{s}" / "hidden_router" / "selection.json"
        if p.is_file():
            layers.append(json.loads(p.read_text())["chosen"]["layer"])
    return layers


# ---------------------------------------------------------------- Fig: route early, answer late
def fig_route_vs_patch(out: Path):
    route = defaultdict(dict)
    for r in _rows(STATS / "routing_layers.csv"):
        route[r["dataset"]][int(r["layer"])] = float(r["test_top1_mean"])
    patch = defaultdict(list)
    for r in _rows(STATS / "activation_patching.csv"):
        # hook on decoder block l -> hidden-state index l+1 (index 0 = token embeddings)
        patch[(r["dataset"], int(r["layer"]) + 1)].append(float(r["restoration"]))
    n_states = 17
    r_sets = ["l_avia", "s_avia", "cqad20", "cqad20_fine", "cqad20_answer"]
    p_sets = ["l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20"]

    R = np.array([[100 * (route[d][l] - route[d][0]) for l in range(n_states)] for d in r_sets])
    P = np.full((len(p_sets), n_states), np.nan)
    for i, d in enumerate(p_sets):
        for l in range(1, n_states):
            if patch.get((d, l)):
                P[i, l] = statistics.median(patch[(d, l)])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.9), gridspec_kw={"height_ratios": [len(r_sets), len(p_sets)], "hspace": 0.55})
    im1 = a1.imshow(R, aspect="auto", cmap=SEQ, vmin=0, vmax=max(5.5, np.nanmax(R)))
    for i, d in enumerate(r_sets):
        for l in _chosen_layers(d if d != "s_avia" else "s_avia"):
            a1.scatter(l, i, marker="*", s=70, color=YELLOW, edgecolor=INK, linewidth=0.6, zorder=3)
        best = int(np.argmax(R[i]))
        a1.text(n_states - 0.35, i, f"{100 * route[d][best]:.1f}%", va="center", ha="left", fontsize=7.5, color=INK)
    a1.set_yticks(range(len(r_sets)), [LABEL[d] if d != "s_avia" else "S-Avia (all inflation levels)" for d in r_sets])
    a1.set_xticks(range(n_states), ["emb"] + [str(i) for i in range(1, n_states)])
    a1.set_title("(a) WHICH expert? Routing top-1 gain over the embedding layer (frozen base)", loc="left", fontsize=8.5)
    c1 = fig.colorbar(im1, ax=a1, fraction=0.025, pad=0.10)
    c1.set_label("Δ top-1 (pp)", fontsize=7.5)
    c1.ax.tick_params(labelsize=7)

    vmax = 1.0
    im2 = a2.imshow(np.clip(P, -2.5, vmax), aspect="auto", cmap=DIV, norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=vmax))
    a2.imshow(np.where(np.isnan(P), 1, np.nan), aspect="auto", cmap=LinearSegmentedColormap.from_list("na", [MID, MID]))
    for i in range(len(p_sets)):
        for l in range(1, n_states):
            if not np.isnan(P[i, l]) and P[i, l] > 0:
                a2.text(l, i, f"{P[i, l]:.2f}", ha="center", va="center", fontsize=6, color=INK)
    a2.set_yticks(range(len(p_sets)), [LABEL[d] for d in p_sets])
    a2.set_xticks(range(n_states), ["emb"] + [str(i) for i in range(1, n_states)])
    a2.set_xlabel("Hidden state of Llama-3.2-1B (emb = token embeddings, i = output of decoder block i)")
    a2.set_title("(b) WHAT answer? Patching expert → fine-tuned 1B: median restoration of gold-answer log-prob.", loc="left", fontsize=8.5)
    c2 = fig.colorbar(im2, ax=a2, fraction=0.025, pad=0.10)
    c2.set_label("restoration (1 = expert)", fontsize=7.5)
    c2.ax.tick_params(labelsize=7)
    for ax in (a1, a2):
        ax.set_xticks(np.arange(-0.5, n_states, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ax.get_ylim()[0] + 1, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    a1.scatter([], [], marker="*", s=70, color=YELLOW, edgecolor=INK, linewidth=0.6, label="layer selected on training data (one star per seed); right: best-layer top-1")
    a1.legend(loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=7, handletextpad=0.3)
    _save(fig, out, "fig_route_early_answer_late")


# ---------------------------------------------------------------- Fig: router accuracy
def fig_router_accuracy(out: Path):
    rows = {r["dataset"]: r for r in _rows(STATS / "routing.csv")}
    sets = ["l_avia", "s_avia", "cqad20", "cqad20_fine", "cqad20_answer"]
    series = [("classifier_top1", "LM-classifier router (fine-tuned 1B)", "#b9b8b3"),
              ("tfidf_top1", "Mahalanobis on TF-IDF (lexical)", ORANGE),
              ("mahalanobis_top1", "Mahalanobis router on hidden states (proposed)", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = 0.26
    x = np.arange(len(sets))
    for k, (col, name, color) in enumerate(series):
        m = [float(rows[d][col + "_mean"]) for d in sets]
        sd = [float(rows[d][col + "_sd"] or 0) for d in sets]
        ax.bar(x + (k - 1) * w, m, w - 0.03, yerr=sd, color=color, label=name, error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
        if col == "mahalanobis_top1":
            for xi, v in zip(x, m):
                ax.text(xi + w, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    t5 = [float(rows[d]["mahalanobis_top5_mean"]) for d in sets]
    ax.scatter(x + w, t5, marker="_", s=120, color=INK, linewidth=1.4, zorder=3, label="Mahalanobis router, top-5")
    ax.set_xticks(x, ["L-Avia", "S-Avia\n(all levels)", "CQAD\n(11 experts)", "CQAD-fine\n(24)", "CQAD-answer\n(72)"])
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("Routing accuracy (test)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.2)
    _save(fig, out, "fig_router_accuracy")


# ---------------------------------------------------------------- quality helpers
def _quality():
    q = defaultdict(dict)
    for r in _rows(STATS / "paper_quality.csv"):
        q[r["dataset"]][r["model"]] = r
    return q


SYSTEMS = [("SLG", "SLG (Mahalanobis router)", BLUE, "o"), ("8B", "Fine-tuned 8B", ORANGE, "s"),
           ("1B", "Fine-tuned 1B", AQUA, "^"), ("RAG", "RAG (8B)", YELLOW, "D")]


def fig_inflation(out: Path):
    q = _quality()
    levels = [("s_avia", 0), ("s_avia_25", 28.4), ("s_avia_50", 57.0), ("s_avia_75", 80.8), ("s_avia_100", 100)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
    for ax, metric, title in zip(axes, ("ROUGE-L", "AI_Expert"), ("ROUGE-L", "AI-Expert (LLM judge)")):
        for key, name, color, mk in SYSTEMS:
            xs = [p for _, p in levels]
            ys = [float(q[d][key][metric + "_mean"]) for d, _ in levels]
            sd = [float(q[d][key][metric + "_sd_sample"] or 0) for d, _ in levels]
            ax.errorbar(xs, ys, yerr=sd, color=color, marker=mk, markersize=4.5, linewidth=1.6, capsize=2, label=name)
        cls = [float(q[d]["SLG-classifier"][metric + "_mean"]) for d, _ in levels]
        ax.plot([p for _, p in levels], cls, color=BLUE, linestyle=":", linewidth=1.3, label="SLG (LM-classifier router)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Inflated answers (%)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0.6, 1.0)
    axes[1].set_ylim(0.6, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=7)
    fig.text(0.01, -0.04, "RAG ROUGE-L (0.41–0.45) lies below the plotted range.", fontsize=7, color=INK2)
    fig.tight_layout()
    _save(fig, out, "fig_inflation")


def fig_granularity(out: Path):
    q = _quality()
    rt = {r["dataset"]: r for r in _rows(STATS / "routing.csv")}
    sets = [("cqad20", 11), ("cqad20_fine", 24), ("cqad20_answer", 72)]
    xs = [n for _, n in sets]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for col, name, color, ls in (("classifier_top1", "LM-classifier router", "#8d8c87", "--"),
                                 ("mahalanobis_top1", "Mahalanobis router (proposed)", BLUE, "-")):
        a1.plot(xs, [float(rt[d][col + "_mean"]) for d, _ in sets], color=color, linestyle=ls, marker="o", markersize=4.5, linewidth=1.8, label=name)
    a1.set_ylabel("Routing top-1")
    a1.set_title("(a) Routing as experts get finer", loc="left")
    for key, name, color, ls in (("SLG", "SLG, Mahalanobis router", BLUE, "-"), ("SLG-classifier", "SLG, LM-classifier router", "#8d8c87", "--"),
                                 ("SLG-oracle", "SLG, oracle routing", BLUE, ":")):
        a2.plot(xs, [float(q[d][key]["ROUGE-L_mean"]) for d, _ in sets], color=color, linestyle=ls, marker="o", markersize=4.5, linewidth=1.8, label=name)
    for key, name, color in (("8B", "Fine-tuned 8B", ORANGE), ("1B", "Fine-tuned 1B", AQUA)):
        v = float(q["cqad20"][key]["ROUGE-L_mean"])
        a2.axhline(v, color=color, linewidth=1.3, label=name)
    a2.set_ylabel("ROUGE-L")
    a2.set_title("(b) Answer quality vs expert granularity", loc="left")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("Number of experts (same questions and splits)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.legend(fontsize=7, loc="lower left")
    a2.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_granularity")


def _toy_router():
    """2-D toy data for the geometric illustration: three experts with one shared, tilted covariance."""
    rng = np.random.default_rng(20261109)
    ang = np.pi / 6
    u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
    true_means = [-0.6 * v, 2.0 * u + 0.6 * v, -2.6 * u + 2.4 * v]
    X, y = [], []
    for k, m in enumerate(true_means):
        g = rng.standard_normal((16, 2))
        X.append(m + 2.0 * g[:, :1] * u + 0.5 * g[:, 1:] * v)
        y += [k] * 16
    X, y = np.vstack(X), np.array(y)
    mu = np.stack([X[y == k].mean(0) for k in range(3)])
    R = X - mu[y]
    S = R.T @ R / (len(X) - 3)
    alpha = 0.05
    Sa = (1 - alpha) * S + alpha * np.trace(S) / 2 * np.eye(2)
    q = mu[0] + 2.6 * u + 0.3 * v  # a normal A paraphrase, far along A's long axis
    return X, y, mu, Sa, q


def fig_router_geometry(out: Path):
    X, y, mu, Sa, q = _toy_router()
    Si = np.linalg.inv(Sa)
    lam, E = np.linalg.eigh(Sa)
    Wh = (E / np.sqrt(lam)) @ E.T * np.sqrt(np.sqrt(lam.prod()))  # symmetric S^-1/2, scaled to keep the size
    c0 = X.mean(0)
    white = lambda P: (P - c0) @ Wh.T + c0  # noqa: E731
    d_e = np.linalg.norm(mu - q, axis=1)
    d_m = np.sqrt(np.einsum("ki,ij,kj->k", mu - q, Si, mu - q))
    assert d_e.argmin() == 1 and d_m.argmin() == 0, (d_e, d_m)
    cols, marks, names = [BLUE, ORANGE, AQUA], ["o", "s", "^"], ["A", "B", "C"]

    def regions(ax, metric, P2world, xs, ys):
        gx, gy = np.meshgrid(xs, ys)
        G = P2world(np.c_[gx.ravel(), gy.ravel()])
        D = G[:, None, :] - mu[None, :, :]
        dist = np.einsum("nki,ij,nkj->nk", D, metric, D)
        lab = dist.argmin(1).reshape(gx.shape)
        for k in range(3):
            ax.contourf(gx, gy, (lab == k).astype(float), levels=[0.5, 1.5], colors=[cols[k]], alpha=0.10)
        ax.contour(gx, gy, lab, levels=[0.5, 1.5], colors=INK2, linewidths=0.8)

    def ellipse(M, center, r):
        t = np.linspace(0, 2 * np.pi, 120)
        L = np.linalg.cholesky(M)
        return center + r * (np.c_[np.cos(t), np.sin(t)] @ L.T)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    panels = [("(a) Euclidean distance", "euclid"), ("(b) Mahalanobis distance (LDA)", "maha"),
              ("(c) Whitened view: nearest centre", "white")]
    xs, ys = np.linspace(-8.5, 5.5, 360), np.linspace(-3.8, 5.2, 240)
    for ax, (title, mode) in zip(axes, panels):
        tf = white if mode == "white" else (lambda P: P)  # noqa: E731
        inv_tf = (lambda P: (P - c0) @ np.linalg.inv(Wh).T + c0) if mode == "white" else (lambda P: P)  # noqa: E731
        metric = np.eye(2) if mode == "euclid" else Si
        regions(ax, metric, inv_tf, xs, ys)
        for k in range(3):
            P = tf(X[y == k])
            ax.scatter(P[:, 0], P[:, 1], s=11, marker=marks[k], color=cols[k], edgecolors="white", linewidths=0.4, zorder=3)
            if mode != "euclid":
                for r, lw, ls in ((1, 1.1, "-"), (2, 0.7, "--")):
                    e = tf(ellipse(Sa, mu[k], r))
                    ax.plot(e[:, 0], e[:, 1], color=cols[k], lw=lw, ls=ls, zorder=2)
            m = tf(mu[k][None])[0]
            ax.scatter(*m, s=55, marker=marks[k], color=cols[k], edgecolors=INK, linewidths=0.9, zorder=5)
            ax.annotate(r"$\mu_{%s}$" % names[k], m, xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK, zorder=6)
        win = 1 if mode == "euclid" else 0
        qq, mw = tf(q[None])[0], tf(mu[win][None])[0]
        ax.plot([qq[0], mw[0]], [qq[1], mw[1]], color=INK, lw=1.2, zorder=4)
        ax.scatter(*qq, s=60, marker="o", facecolor="white", edgecolor=INK, linewidths=1.6, zorder=7)
        ax.annotate("$q$", qq, xytext=(6, -10), textcoords="offset points", fontsize=8.5, color=INK, zorder=8)
        dist = d_e if mode == "euclid" else d_m
        verdict = "routes to B (wrong)" if mode == "euclid" else "routes to A"
        ax.text(0.02, 0.03, f"d(q,A) = {dist[0]:.2f}, d(q,B) = {dist[1]:.2f}\n{verdict}", transform=ax.transAxes,
                fontsize=7, color=INK, va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.6))
        ax.set_title(title, loc="left")
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.tight_layout(w_pad=0.6)
    _save(fig, out, "fig_router_geometry")
    return d_e, d_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT.parent / "SLG-AEI" / "figures"))
    out = Path(p.parse_args().out)
    fig_route_vs_patch(out)
    fig_router_accuracy(out)
    fig_inflation(out)
    fig_granularity(out)
    d_e, d_m = fig_router_geometry(out)
    print("toy router distances: Euclidean", d_e.round(2), "Mahalanobis", d_m.round(2))
    print("figures written to", out)


if __name__ == "__main__":
    main()
