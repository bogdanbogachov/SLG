"""Tiny isolated end-to-end run for validating the Slurm experiment pipeline."""
from __future__ import annotations

import copy
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _shortest(rows, count):
    return sorted(rows, key=lambda r: (len(r.get("question", "")) + len(r.get("answer", "")), r["qid"]))[:count]


def run_smoke_test(dataset: str, seed: int):
    """Run a compact bundle and smoke-check analyses without touching full results."""
    from main import configure_bundle, run_router_counterfactuals
    from config import CONFIG

    configure_bundle(dataset, seed)
    normal_train = json.loads(Path(CONFIG["files"]["qa_train"]).read_text(encoding="utf-8"))
    normal_test = json.loads(Path(CONFIG["files"]["qa_test"]).read_text(encoding="utf-8"))
    train_by_title, test_by_title = defaultdict(list), defaultdict(list)
    for row in normal_train:
        train_by_title[row["title"]].append(row)
    for row in normal_test:
        test_by_title[row["title"]].append(row)
    eligible = sorted(title for title in train_by_title if len(train_by_title[title]) >= 5 and test_by_title[title])
    if len(eligible) < 2:
        raise RuntimeError(f"Cannot make smoke split for {dataset}: need two titles with >=5 train and >=1 test rows.")

    selected_titles = eligible[:2]
    train_rows = [row for title in selected_titles for row in _shortest(train_by_title[title], 5)]
    test_rows = [row for title in selected_titles for row in _shortest(test_by_title[title], 2)]
    smoke_root = ROOT / "data" / "_smoke" / dataset
    smoke_split = smoke_root / "splits" / f"seed_{seed}"
    source_rows = sorted(train_rows + test_rows, key=lambda row: row["qid"])
    _write_json(smoke_root / "source.json", source_rows)
    _write_json(smoke_split / "train.json", train_rows)
    _write_json(smoke_split / "test.json", test_rows)
    by_title_dir = smoke_split / "by_title"
    if by_title_dir.exists():
        shutil.rmtree(by_title_dir)
    by_title_dir.mkdir(parents=True, exist_ok=True)
    for title in selected_titles:
        slug = title.replace(" ", "_").replace("/", "_").replace("\n", "_").lower()
        _write_json(by_title_dir / f"{slug}.json", [r for r in train_rows if r["title"] == title])

    cell = ROOT / "results" / "smoke" / dataset / f"seed_{seed}"
    CONFIG["paths"].update({
        "experiments": str(cell), "answers": str(cell / "answers"),
        "checkpoints": str(cell / "checkpoints"),
        "split_by_title": str(by_title_dir),
    })
    CONFIG["files"].update({
        "qa": str(smoke_root / "source.json"),
        "qa_train": str(smoke_split / "train.json"),
        "qa_test": str(smoke_split / "test.json"),
    })
    CONFIG["logging"]["log_dir"] = str(cell / "logs")
    CONFIG["training"] = copy.deepcopy(CONFIG["training"])
    CONFIG["training"].update({"num_epochs": 1, "logging_steps": 1,
                               "per_device_train_batch_size": 1,
                               "per_device_eval_batch_size": 1,
                               "gradient_accumulation_steps": 1,
                               "early_stopping_patience": 1})
    CONFIG["slg_router_finetuning"] = copy.deepcopy(CONFIG["slg_router_finetuning"])
    router_training = CONFIG["slg_router_finetuning"]["training"]
    router_training.update({"num_train_epochs": 1, "max_length": 128,
                            "per_device_train_batch_size": 2,
                            "per_device_eval_batch_size": 2,
                            "effective_batch_size": 4, "gradient_accumulation_steps": 2,
                            "eval_steps": 1, "save_steps": 1,
                            "early_stopping_patience": 1, "logging_steps": 1})
    CONFIG["data"]["max_length"] = min(int(CONFIG["data"]["max_length"]), 256)
    CONFIG["generation"]["max_new_tokens"] = min(int(CONFIG["generation"]["max_new_tokens"]), 24)
    CONFIG["evaluation"]["checkpoint_every"] = 1
    CONFIG["evaluation"]["workers"] = 1

    print(f"Smoke bundle: dataset={dataset}, seed={seed}, titles={selected_titles}, "
          f"train={len(train_rows)}, test={len(test_rows)}; output={cell}")
    from commands.train import run_training
    from commands.inference import run_finetuned, run_slg, run_rag

    run_training("")
    run_finetuned("")
    run_slg("", router_method="finetuned")
    run_rag("")
    run_router_counterfactuals(run_slg)

    from evaluate.activation_patching import run as run_patching

    stats_dir = ROOT / "results" / "smoke" / "stats"
    smoke_results = ROOT / "results" / "smoke"
    smoke_data = ROOT / "data" / "_smoke"
    patch_qid = test_rows[0]["qid"]
    print(f"Smoke activation patching -> {run_patching(limit=1, datasets=(dataset,), seeds=(seed,), results_root=smoke_results, data_root=smoke_data, output_dir=stats_dir, qids_by_cell={(dataset, seed): [patch_qid]})}")
    print("Smoke training/inference/patching finished. API-based evaluation must run on a login node.")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m commands.smoke_test DATASET SEED")
    run_smoke_test(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    main()
