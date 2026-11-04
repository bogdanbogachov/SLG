"""Plan steps 1-4: extract hidden states, fit + test the Mahalanobis router per cell, compare, estimate scores.

Outputs
- results/hidden_router/features/              question features cache (shared by all cells; seed-independent)
- results/<ds>/seed_<s>/hidden_router/         router.pt, selection.json (validation grid), test_routes.json,
                                               per_layer_test.json (diagnostic only; not used for selection)
- results/hidden_router/probe_summary.csv      one row per cell
- results/hidden_router/probe_summary.md       mean ± sd over seeds per dataset

Score estimate per test question (no generation needed):
  new expert == gold expert          -> oracle answer's score             (exact: same expert, greedy)
  new expert == old router's expert  -> old SLG answer's score            (exact)
  new expert == wrong-override expert-> wrong-routing answer's score      (exact)
  otherwise                          -> wrong-routing answer's score      (proxy: another wrong expert)
"""
from __future__ import annotations

import csv
import json
import statistics
from typing import Dict, List, Sequence

from logging_config import logger

from hidden_router.cells import (BASELINES, METRICS, NEW_SLG, OLD_SLG, ORACLE, RESULTS, WRONG, Cell,
                                 expert_id_from_title)
from hidden_router.features import FeatureStore
from hidden_router.router import MahalanobisRouter, _topk_hits  # noqa: F401  (re-exported for tests)

FEATURES_DIR = RESULTS / "hidden_router" / "features"


def _labels(rows: Sequence[dict]) -> List[str]:
    return [expert_id_from_title(r["title"]) for r in rows]


def fit_cell(cell: Cell, store: FeatureStore, *, refit: bool = False) -> MahalanobisRouter:
    if MahalanobisRouter.exists(cell.router_dir) and not refit:
        return MahalanobisRouter.load(cell.router_dir)
    train = cell.train()
    labels = _labels(train)
    on_disk = set(cell.experts_on_disk())
    unknown = sorted(set(labels) - on_disk)
    if unknown:
        raise ValueError(f"{cell}: {len(unknown)} training experts have no adapter under slg/: {unknown[:5]}")
    from hidden_router.pipeline import fit_kwargs
    router = MahalanobisRouter.fit(store.get([r["qid"] for r in train]), labels, seed=cell.seed, **fit_kwargs())
    router.save(cell.router_dir)
    return router


def _rank_of(ranked: List[dict], expert: str):
    return next((c["rank"] for c in ranked if c["expert"] == expert), None)


def test_cell(cell: Cell, store: FeatureStore, router: MahalanobisRouter) -> dict:
    import torch

    test = cell.test()
    qids = [r["qid"] for r in test]
    gold = _labels(test)
    feats = store.get(qids)
    ranked = router.rank(feats)
    routes = {q: {"expected_expert": g, "selected_expert": rk[0]["expert"], "expected_rank": _rank_of(rk, g),
                  "top": rk[:10]} for q, g, rk in zip(qids, gold, ranked)}
    (cell.router_dir / "test_routes.json").write_text(json.dumps(routes, indent=1), encoding="utf-8")

    def topk(k):
        return sum(1 for r in routes.values() if r["expected_rank"] is not None and r["expected_rank"] <= k) / len(routes)

    # Diagnostic per-layer curve (refit on all training questions per layer, chosen alpha). For the paper's
    # comparison with activation patching only; the router itself was selected on validation data.
    train = cell.train()
    train_feats, train_labels = store.get([r["qid"] for r in train]), _labels(train)
    name_id = {n: i for i, n in enumerate(router.labels)}
    y = torch.tensor([name_id.get(g, -1) for g in gold])
    keep = y >= 0
    per_layer = []
    for layer in range(feats.shape[1]):
        r = MahalanobisRouter.fit_fixed(train_feats, train_labels, layer=layer, alpha=router.alpha, names=router.labels)
        hits = _topk_hits(r.scores(feats[:, layer, :])[keep], y[keep])
        per_layer.append({"layer": layer, "test_top1": hits[1], "test_top5": hits[5]})
    (cell.router_dir / "per_layer_test.json").write_text(json.dumps(per_layer, indent=1), encoding="utf-8")
    return {"routes": routes, "top1": topk(1), "top3": topk(3), "top5": topk(5)}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def estimate_cell(cell: Cell, routes: Dict[str, dict]) -> dict:
    oracle, wrong, old = cell.scores(ORACLE), cell.scores(WRONG), cell.scores(OLD_SLG)
    if not (oracle and wrong and old):
        return {"estimate_available": False}
    old_sel, wrong_sel = cell.selected_by_qid(OLD_SLG), cell.selected_by_qid(WRONG)
    out = {"estimate_available": True}
    sources = {"oracle": 0, "old_router": 0, "wrong_exact": 0, "wrong_proxy": 0}
    per_metric = {m: [] for m in METRICS}
    for qid, r in routes.items():
        new = r["selected_expert"]
        if new == r["expected_expert"]:
            src, row = "oracle", oracle.get(qid)
        elif new == old_sel.get(qid):
            src, row = "old_router", old.get(qid)
        elif new == wrong_sel.get(qid):
            src, row = "wrong_exact", wrong.get(qid)
        else:
            src, row = "wrong_proxy", wrong.get(qid)
        sources[src] += 1
        for m in METRICS:
            per_metric[m].append(None if row is None else row.get(m))
    n = max(1, len(routes))
    out.update({f"share_{k}": v / n for k, v in sources.items()})
    out["n_to_generate"] = sources["wrong_proxy"]
    for m in METRICS:
        out[f"new_SLG_{m}_est"] = _mean(per_metric[m])
        out[f"old_SLG_{m}"] = _mean(r.get(m) for r in old.values())
        out[f"oracle_{m}"] = _mean(r.get(m) for r in oracle.values())
        for name, stem in BASELINES.items():
            out[f"{name}_{m}"] = _mean(r.get(m) for r in cell.scores(stem).values())
    return out


def run(datasets: Sequence[str], seeds: Sequence[int], *, refit: bool = False) -> str:
    cells = [Cell(d, s) for d in datasets for s in seeds]
    missing = [c for c in cells if not c.exists()]
    for c in missing:
        logger.warning("Skipping %s: split or slg/ experts missing", c)
    cells = [c for c in cells if c.exists()]
    if not cells:
        raise SystemExit("No cells with splits and trained experts found.")

    # Step 1: extract once for every distinct question (seed- and inflation-independent).
    store = FeatureStore(FEATURES_DIR)
    items = {}
    for c in cells:
        for r in c.train() + c.test():
            items.setdefault(r["qid"], r["question"])
    store.ensure(list(items.items()))

    rows = []
    for c in cells:
        logger.info("Hidden router: fitting and testing %s", c)
        router = fit_cell(c, store, refit=refit)  # step 2
        result = test_cell(c, store, router)
        old = c.routing_report(OLD_SLG) or {}
        old_topk = old.get("summary", {}).get("top_k_accuracy", {})  # step 3
        row = {"dataset": c.dataset, "seed": c.seed, "experts": len(router.labels), "test_questions": len(result["routes"]),
               "layer": router.layer, "alpha": router.alpha,
               "val_top1": router.selection.get("chosen", {}).get("val_top1"),
               "new_top1": result["top1"], "new_top3": result["top3"], "new_top5": result["top5"],
               "old_top1": old_topk.get("top_1", {}).get("accuracy"), "old_top5": old_topk.get("top_5", {}).get("accuracy")}
        row.update(estimate_cell(c, result["routes"]))  # step 4
        rows.append(row)
        logger.info("%s: layer=%s alpha=%s new top1=%.4f old top1=%s", c, router.layer, router.alpha, result["top1"], row["old_top1"])

    out_dir = RESULTS / "hidden_router"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = _markdown(rows)
    (out_dir / "probe_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return str(out_dir / "probe_summary.csv")


def _ms(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return "–"
    return f"{statistics.mean(values):.3f}" + (f" ±{statistics.stdev(values):.3f}" if len(values) > 1 else "")


def _markdown(rows: List[dict]) -> str:
    cols = [("old_top1", "Old top-1"), ("new_top1", "New top-1"), ("new_top5", "New top-5"),
            ("old_SLG_ROUGE-L", "Old SLG RL"), ("new_SLG_ROUGE-L_est", "New SLG RL (est)"), ("8B_ROUGE-L", "8B RL"),
            ("1B_ROUGE-L", "1B RL"), ("RAG_ROUGE-L", "RAG RL"),
            ("old_SLG_AI_Expert", "Old SLG AIE"), ("new_SLG_AI_Expert_est", "New SLG AIE (est)"), ("8B_AI_Expert", "8B AIE"),
            ("1B_AI_Expert", "1B AIE"), ("RAG_AI_Expert", "RAG AIE"), ("share_wrong_proxy", "Proxy share"),
            ("n_to_generate", "To generate")]
    lines = ["| Dataset | Layer(s) | " + " | ".join(c[1] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        group = [r for r in rows if r["dataset"] == ds]
        layers = ",".join(str(r["layer"]) for r in group)
        cells = [_ms([r.get(k) for r in group]) if k != "n_to_generate" else str(sum(r.get(k) or 0 for r in group)) for k, _ in cols]
        lines.append(f"| {ds} | {layers} | " + " | ".join(cells) + " |")
    lines.append("\nEstimates reuse exact per-question scores where the chosen expert's answer already exists "
                 "(oracle / old router / wrong-override); 'Proxy share' rows use the wrong-override score instead.")
    return "\n".join(lines) + "\n"
