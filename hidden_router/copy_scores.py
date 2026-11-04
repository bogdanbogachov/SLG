"""One-off: pre-fill the evaluation checkpoint of slg_mahalanobis_router with scores already computed for
byte-identical answers, so evaluate_all.sh only scores the answers that are actually new.

Run ONCE on the login node after all assemble jobs finished and BEFORE evaluate_all.sh:
    python -m hidden_router.copy_scores [--datasets a,b] [--seeds 42,53,64] [--dry_run]

For each cell and each test question i (answers file is in test order, as evaluate() pairs by position):
- look for the same qid in the oracle / classifier-SLG / wrong-routing answer files;
- if that answer string is exactly identical to the new answer and that file's evaluation checkpoint has a
  score for it, copy the raw score dict into checkpoints/evaluation/slg_mahalanobis_router.json at index i.
All metrics depend only on (gold answer, predicted answer), and the gold answer is the same test row, so the
copied scores are the scores the evaluator would assign (AI_Expert comes from the earlier judge call).
evaluate_all.sh then resumes from this checkpoint and scores only the remaining indices.
Existing entries in the target checkpoint are kept (re-running is safe). Writes copy_scores_report.json per cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hidden_router.cells import DATASETS, NEW_SLG, OLD_SLG, ORACLE, SEEDS, WRONG, Cell, _read_jsonl

SOURCES = (ORACLE, OLD_SLG, WRONG)


def _checkpoint(cell: Cell, stem: str) -> Path:
    return cell.dir / "checkpoints" / "evaluation" / f"{stem}.json"


def _source_scores(cell: Cell, stem: str, test_qids):
    """qid -> (answer text, raw score dict) for one already-evaluated answer file."""
    path = _checkpoint(cell, stem)
    answers = _read_jsonl(cell.dir / "answers" / f"{stem}.jsonl")
    if not path.is_file() or not answers:
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != 2:
        return {}
    out = {}
    for i, scores in state.get("completed_scores", {}).items():
        i = int(i)
        # evaluate() pairs answers[i] with test[i]; only trust rows where both refer to the same qid.
        if i < len(answers) and i < len(test_qids) and answers[i].get("qid") == test_qids[i]:
            out[test_qids[i]] = (answers[i].get("answer"), scores)
    return out


def copy_cell(cell: Cell, dry_run: bool = False) -> dict:
    test = cell.test()
    test_qids = [r["qid"] for r in test]
    new = _read_jsonl(cell.dir / "answers" / f"{NEW_SLG}.jsonl")
    if [r.get("qid") for r in new] != test_qids:
        return {"cell": str(cell), "status": "skipped: answers file missing, incomplete or not in test order"}
    target = _checkpoint(cell, NEW_SLG)
    existing = {}
    if target.is_file():
        state = json.loads(target.read_text(encoding="utf-8"))
        if state.get("version") == 2 and state.get("n_pairs") == len(test):
            existing = {int(i): s for i, s in state.get("completed_scores", {}).items()}
    sources = {stem: _source_scores(cell, stem, test_qids) for stem in SOURCES}
    scores, copied_from = dict(existing), {stem: 0 for stem in SOURCES}
    for i, row in enumerate(new):
        if i in scores:
            continue
        for stem in SOURCES:
            hit = sources[stem].get(row["qid"])
            if hit is not None and hit[0] == row.get("answer"):
                scores[i] = hit[1]
                copied_from[stem] += 1
                break
    report = {"cell": str(cell), "status": "ok", "questions": len(test), "already_in_checkpoint": len(existing),
              "copied": sum(copied_from.values()), "copied_from": copied_from,
              "left_to_evaluate": len(test) - len(scores)}
    if not dry_run and len(scores) > len(existing):
        from evaluate.evaluate import _save_eval_checkpoint
        _save_eval_checkpoint(str(target), len(test), scores, set())
    if not dry_run:
        cell.router_dir.mkdir(parents=True, exist_ok=True)
        (cell.router_dir / "copy_scores_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(prog="python -m hidden_router.copy_scores")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--dry_run", action="store_true", help="report counts without writing anything")
    args = p.parse_args()
    total = left = 0
    for ds in [d for d in args.datasets.split(",") if d]:
        for seed in [int(s) for s in args.seeds.split(",") if s]:
            r = copy_cell(Cell(ds, seed), dry_run=args.dry_run)
            if r["status"] != "ok":
                print(f"{r['cell']:22s} {r['status']}")
                continue
            total += r["questions"]
            left += r["left_to_evaluate"]
            print(f"{r['cell']:22s} copied {r['copied']:5d}  left to evaluate {r['left_to_evaluate']:4d} / {r['questions']}")
    print(f"TOTAL left to evaluate: {left} / {total}" + ("  (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
