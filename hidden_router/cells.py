"""Helpers for one dataset x seed cell: splits, expert ids, answers, scores and routing reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.path_utils import slg_expert_id_from_filename

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SEEDS = (42, 53, 64)
DATASETS = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100",
            "cqad20", "cqad20_fine", "cqad20_answer")

# Answer / score / report stems produced by the main pipeline.
OLD_SLG = "slg_finetuned_router"
ORACLE = "slg_finetuned_router_oracle"
WRONG = "slg_finetuned_router_wrong"
NEW_SLG = "slg_mahalanobis_router"
BASELINES = {"1B": "finetuned_3_2_1b", "8B": "finetuned_3_1_8b", "RAG": "rag"}
METRICS = ("ROUGE-L", "AI_Expert")


def expert_id_from_title(title: str) -> str:
    """Same mapping as SmallLanguageGraph._expected_expert_id_from_title (title -> adapter directory name)."""
    split_title = str(title).replace(" ", "_").replace("/", "_").replace("\n", "_").lower()
    return slg_expert_id_from_filename(f"{split_title}.json")


def _read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # incomplete final line of an interrupted run
    return rows


class Cell:
    def __init__(self, dataset: str, seed: int):
        self.dataset, self.seed = dataset, int(seed)
        self.dir = RESULTS / dataset / f"seed_{seed}"
        self.split_dir = DATA / dataset / "splits" / f"seed_{seed}"
        self.router_dir = self.dir / "hidden_router"

    def __repr__(self):
        return f"{self.dataset}/seed_{self.seed}"

    def exists(self) -> bool:
        return (self.split_dir / "train.json").is_file() and (self.split_dir / "test.json").is_file() and (self.dir / "slg").is_dir()

    def train(self) -> List[dict]:
        return _read_json(self.split_dir / "train.json")

    def test(self) -> List[dict]:
        return _read_json(self.split_dir / "test.json")

    def experts_on_disk(self) -> List[str]:
        slg = self.dir / "slg"
        return sorted(p.name for p in slg.iterdir() if p.is_dir()) if slg.is_dir() else []

    def answers(self, stem: str) -> Dict[str, dict]:
        return {r["qid"]: r for r in _read_jsonl(self.dir / "answers" / f"{stem}.jsonl") if r.get("qid")}

    def scores(self, stem: str) -> Dict[str, dict]:
        return {r["qid"]: r for r in _read_jsonl(self.dir / "scores" / f"{stem}.jsonl") if r.get("qid")}

    def routing_report(self, stem: str) -> Optional[dict]:
        path = self.dir / "routing_reports" / f"{stem}_routing_report.json"
        return _read_json(path) if path.is_file() else None

    def selected_by_qid(self, stem: str) -> Dict[str, str]:
        """qid -> expert the given SLG run actually used (records are in test.json order via 'index')."""
        report = self.routing_report(stem)
        if not report:
            return {}
        test = self.test()
        out = {}
        for record in report.get("records", []):
            i = record.get("index")
            if i is not None and 0 <= int(i) < len(test) and record.get("selected_expert"):
                out[test[int(i)]["qid"]] = record["selected_expert"]
        return out
