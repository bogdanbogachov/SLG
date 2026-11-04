"""Plan steps 5-6: verify that existing single-expert answers can be reused, then build the new SLG answers.

For each test question the Mahalanobis router (fitted by probe.py) picks an expert. The finetuned-router SLG,
its oracle and its wrong-override runs all answer with exactly ONE expert (no neighbours, greedy decoding,
same prompt and generation code), so an answer from expert E to question q is the same whichever run produced
it. The new answer is therefore copied when one of those runs already used the chosen expert:

  chosen == gold expert          -> oracle answer
  chosen == old router's choice  -> old SLG answer
  chosen == wrong-override expert-> wrong-routing answer
  otherwise                      -> generated now with the chosen expert (same code path as SLG inference)

Step 5 checks the premise on real data: where the old router chose the gold expert, its answer must equal the
oracle answer. Below --reuse_threshold (default 0.95) the cell stops unless --allow_reuse_mismatch is given.

Output: results/<ds>/seed_<s>/answers/slg_mahalanobis_router.jsonl (resumable; evaluate_all scores it like any
other answer file) and hidden_router/assembly_report.json + reuse_check.json.
"""
from __future__ import annotations

import json
import time
from typing import Sequence

from logging_config import logger

from hidden_router.cells import NEW_SLG, OLD_SLG, ORACLE, WRONG, Cell


def reuse_check(cell: Cell) -> dict:
    oracle, old = cell.answers(ORACLE), cell.answers(OLD_SLG)
    report = cell.routing_report(OLD_SLG) or {}
    test = cell.test()
    compared = identical = 0
    examples = []
    for record in report.get("records", []):
        if not record.get("routing_correct") or len(record.get("visited_experts") or []) > 1:
            continue
        qid = test[int(record["index"])]["qid"]
        if qid not in oracle or qid not in old:
            continue
        compared += 1
        a, b = str(old[qid].get("answer") or "").strip(), str(oracle[qid].get("answer") or "").strip()
        if a == b:
            identical += 1
        elif len(examples) < 5:
            examples.append({"qid": qid, "old_slg": a[:300], "oracle": b[:300]})
    result = {"compared": compared, "identical": identical,
              "identical_share": identical / compared if compared else None, "mismatch_examples": examples}
    cell.router_dir.mkdir(parents=True, exist_ok=True)
    (cell.router_dir / "reuse_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class _Generator:
    """Single-expert generation through SmallLanguageGraph's own method (identical prompt/decoding)."""

    def __init__(self, cell: Cell):
        from inference.slg import SmallLanguageGraph
        self._slg = SmallLanguageGraph.__new__(SmallLanguageGraph)  # only the stateless generation method is used
        self._slg_dir = cell.dir / "slg"

    def __call__(self, question: str, expert: str) -> dict:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self._slg._tuned_generate_with_confidence(question, str(self._slg_dir / expert))
        return {"answer": result["answer"], "latency_s": time.perf_counter() - start,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None}


def assemble_cell(cell: Cell, *, reuse_threshold: float = 0.95, allow_reuse_mismatch: bool = False,
                  generator_factory=_Generator) -> dict:
    routes_path = cell.router_dir / "test_routes.json"
    if not routes_path.is_file():
        raise SystemExit(f"{cell}: run `python -m hidden_router probe` first (missing {routes_path}).")
    routes = json.loads(routes_path.read_text(encoding="utf-8"))

    check = reuse_check(cell)
    share = check["identical_share"]
    logger.info("%s reuse check: %s/%s identical (%.4f)", cell, check["identical"], check["compared"], share or 0)
    if share is None or share < reuse_threshold:
        message = (f"{cell}: only {check['identical']}/{check['compared']} correctly routed old-SLG answers equal the "
                   f"oracle answers (threshold {reuse_threshold}); see {cell.router_dir / 'reuse_check.json'}.")
        if not allow_reuse_mismatch:
            raise SystemExit(message + " Re-run with --allow_reuse_mismatch to reuse anyway.")
        logger.warning(message + " Continuing because --allow_reuse_mismatch was given.")

    sources = {"oracle": cell.answers(ORACLE), "old_router": cell.answers(OLD_SLG), "wrong": cell.answers(WRONG)}
    chosen_by = {"old_router": cell.selected_by_qid(OLD_SLG), "wrong": cell.selected_by_qid(WRONG)}
    out_path = cell.dir / "answers" / f"{NEW_SLG}.jsonl"
    done = cell.answers(NEW_SLG)
    if done:
        logger.info("%s: resuming assembly, %s answers already written", cell, len(done))
        test_qids = [r["qid"] for r in cell.test()]
        with out_path.open("w", encoding="utf-8") as f:  # drop a partial final line, keep test order
            for qid in test_qids:
                if qid in done:
                    f.write(json.dumps(done[qid], ensure_ascii=False) + "\n")

    generator = None
    counts = {"oracle": 0, "old_router": 0, "wrong": 0, "generated": 0}
    with out_path.open("a", encoding="utf-8") as out:
        for item in cell.test():
            qid = item["qid"]
            route = routes[qid]
            expert = route["selected_expert"]
            if qid in done:
                counts[done[qid].get("answer_source", "generated")] = counts.get(done[qid].get("answer_source", "generated"), 0) + 1
                continue
            if expert == route["expected_expert"]:
                source = "oracle"
            elif expert == chosen_by["old_router"].get(qid):
                source = "old_router"
            elif expert == chosen_by["wrong"].get(qid):
                source = "wrong"
            else:
                source = "generated"
            base = {k: item.get(k) for k in ("qid", "chapter", "title", "question")}
            if source != "generated" and qid in sources[source]:
                reused = sources[source][qid]
                record = {**base, "answer": reused.get("answer"), "latency_s": reused.get("latency_s"),
                          "peak_gpu_memory_mb": reused.get("peak_gpu_memory_mb")}
            else:
                source = "generated"
                if generator is None:
                    generator = generator_factory(cell)
                logger.info("%s: generating %s with expert %s", cell, qid[:12], expert)
                record = {**base, **generator(item["question"], expert)}
            record.update({"router_method": "mahalanobis", "selected_expert": expert,
                           "routing_correct": expert == route["expected_expert"], "answer_source": source})
            counts[source] += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    total = len(routes)
    correct = sum(1 for r in routes.values() if r["selected_expert"] == r["expected_expert"])
    report = {"cell": str(cell), "answers": str(out_path), "routing_accuracy": correct / total if total else None,
              "answer_sources": counts, "reuse_check": {k: v for k, v in check.items() if k != "mismatch_examples"}}
    (cell.router_dir / "assembly_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(cell.answers(NEW_SLG)) == total:
        # Mark the bundle stage as done so a resubmitted bundle does not regenerate (and overwrite) these answers.
        marker = cell.dir / ".done" / "infer_slg_mahalanobis"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done (assembled by hidden_router: oracle/old/wrong answers reused)\n", encoding="utf-8")
    logger.info("%s assembled: %s", cell, counts)
    return report


def run(datasets: Sequence[str], seeds: Sequence[int], *, reuse_threshold: float = 0.95,
        allow_reuse_mismatch: bool = False) -> None:
    from main import configure_bundle  # sets CONFIG paths (models, adapters) for the cell, as the bundle job does

    for dataset in datasets:
        for seed in seeds:
            cell = Cell(dataset, seed)
            if not cell.exists() or not (cell.router_dir / "test_routes.json").is_file():
                logger.warning("Skipping %s: no probe output", cell)
                continue
            configure_bundle(dataset, seed)
            report = assemble_cell(cell, reuse_threshold=reuse_threshold, allow_reuse_mismatch=allow_reuse_mismatch)
            print(json.dumps(report))
