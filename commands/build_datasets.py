"""Deterministic construction of the experiment datasets and per-seed splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import yaml
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 53, 64)
INFLATION_SEED = 20261015


def _read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset input is missing: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _norm(value: str) -> str:
    return " ".join(str(value).lower().split())


def _qid(row: dict) -> str:
    raw = f"{_norm(row.get('question', ''))}\0{_norm(row.get('_original_answer', row.get('answer', '')))}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _slug(title: str) -> str:
    return str(title).replace(" ", "_").replace("/", "_").replace("\n", "_").lower()


def _dedup(rows: list[dict], threshold: float):
    rows = [dict(r) for r in rows if r.get("title") != r.get("answer")]
    for row in rows:
        row["_original_answer"] = row["answer"]
        row["qid"] = _qid(row)
    rows.sort(key=lambda r: (r["qid"], _norm(r.get("question", ""))))
    seen_questions, by_answer, kept = set(), defaultdict(list), []
    for row in rows:
        words = set(re.findall(r"\w+", row.get("question", "").lower()))
        normalized_q = _norm(row.get("question", ""))
        if normalized_q in seen_questions:
            continue
        seen_questions.add(normalized_q)
        if any(_jaccard(words, prior) >= threshold for prior in by_answer[row["_original_answer"]]):
            continue
        by_answer[row["_original_answer"]].append(words)
        kept.append(row)
    return kept, len(rows) - len(kept)


def _jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _inflate(rows, confusers, percentage):
    title_counts = Counter(r.get("title", "") for r in rows)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row.get("title", "")].setdefault(row["_original_answer"], None)
    sorted_assignments = []
    for title, answers in grouped.items():
        for answer in answers:
            sorted_assignments.append((title_counts[title], title, answer))
    sorted_assignments.sort()
    if sorted_assignments and len(confusers) < max(len(v) for v in grouped.values()):
        raise ValueError("Inflation material has fewer confusers than distinct answers in a section title.")
    assigned = {}
    title_ordinals = Counter()
    for _freq, title, answer in sorted_assignments:
        assigned[(title, answer)] = confusers[title_ordinals[title]]["text"]
        title_ordinals[title] += 1
    selected_by_title = {}
    for title, answers in grouped.items():
        rank = sorted(answers, key=lambda a: (hashlib.sha256(f"{INFLATION_SEED}\0{title}\0{a}".encode()).hexdigest(), a))
        count = math.floor(len(rank) * percentage / 100 + 0.5)
        selected_by_title[title] = set(rank[:count])
    result = []
    for row in rows:
        r = dict(row)
        answer = r["_original_answer"]
        if answer in selected_by_title.get(r.get("title", ""), set()):
            r["answer"] = assigned[(r.get("title", ""), answer)] + "\n\n" + answer
        result.append(r)
    return result


def _chapter_num(row):
    m = re.match(r"\s*CHAPTER\s+(\d+)", str(row.get("chapter", "")), re.I)
    return int(m.group(1)) if m else None


def _jaccard_summary(train, test, original_answers):
    train_by_answer = defaultdict(list)
    for r in train:
        train_by_answer[original_answers.get(r.get("qid"), r.get("answer"))].append(set(re.findall(r"\w+", r["question"].lower())))
    vals = []
    for r in test:
        candidates = train_by_answer.get(original_answers.get(r.get("qid"), r.get("answer")), [])
        words = set(re.findall(r"\w+", r["question"].lower()))
        vals.append(max((_jaccard(words, other) for other in candidates), default=0.0))
    ordered = sorted(vals)
    summary = {f"share_ge_{p:g}": sum(v >= p for v in vals) / len(vals) if vals else 0.0 for p in (.25, .5, .75, .8, 1)}
    summary.update({
        "n": len(vals), "mean": sum(vals) / len(vals) if vals else 0.0,
        "median": ordered[len(ordered) // 2] if ordered else 0.0,
        "p90": ordered[min(len(ordered) - 1, int(.9 * len(ordered)))] if ordered else 0.0,
        "p95": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    })
    return summary


def _prefix_overlap(answers, k=5):
    prefixes = defaultdict(set)
    for answer in set(answers):
        words = answer.split()
        prefixes[" ".join(words[:k])].add(" ".join(words[k:]))
    collided = [values for values in prefixes.values() if len(values) > 1]
    return {"unique_prefixes": len(collided), "total_cases": sum(len(v) for v in collided)}


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary.replace(path)


def build_datasets(force=False):
    avia_path = ROOT / "qa_archive/upd_slg_l/qa.json"
    cqad_path = ROOT / "question_answer/cqadupstack_clustered/qa_min20.json"
    inflation_path = ROOT / "question_answer/inflating_material.json"
    avia_raw, cqad_raw, confusers = _read(avia_path), _read(cqad_path), _read(inflation_path)
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    threshold = float(base_cfg.get("dataset", {}).get("min_jaccard_dedup", 0.8))
    with (ROOT / "configs/datasets/s_avia.yaml").open(encoding="utf-8") as f:
        avia_cfg = yaml.safe_load(f).get("dataset", {})
    chapters = [int(c) for c in avia_cfg.get("maintenance_chapters", [])]
    avia, avia_removed = _dedup(avia_raw, threshold)
    chapter_names = {str(r.get("chapter", "")) for r in avia_raw}
    missing = [n for n in chapters if not any(name.startswith(f"CHAPTER {n} -") for name in chapter_names)]
    if missing:
        raise ValueError(f"Avia source is missing expected maintenance-manual chapter fields: {missing}")
    small = [r for r in avia if _chapter_num(r) in set(chapters)]
    cqad, cqad_removed = _dedup(cqad_raw, threshold)
    datasets = {"l_avia": (avia, 0, avia_removed), "s_avia": (small, 0, avia_removed),
                "cqad20": (cqad, 0, cqad_removed)}
    inflated_qids, inflated_answers = {}, {}
    for level in (25, 50, 75, 100):
        inflated = _inflate(small, confusers, level)
        inflated_qids[level] = {r["qid"] for r in inflated}
        inflated_answers[level] = {(r["title"], r["_original_answer"]) for r in inflated if r["answer"] != r["_original_answer"]}
        datasets[f"s_avia_{level}"] = (inflated, level, avia_removed)
    for lower, upper in ((25, 50), (50, 75), (75, 100)):
        if not inflated_answers[lower] <= inflated_answers[upper]:
            raise AssertionError(f"Inflation answer selection is not nested: {lower}% is not a subset of {upper}%")
        if inflated_qids[lower] != inflated_qids[upper]:
            raise AssertionError("Inflation levels must have identical qid sets")

    split_qids = {}
    for name, (rows, pct, removed) in datasets.items():
        out = ROOT / "data" / name
        source_path = out / "source.json"
        if force or not source_path.exists():
            _write_json(source_path, [{k: v for k, v in row.items() if k != "_original_answer"} for row in rows])
        uniq_answers = len({r.get("_original_answer", r["answer"]) for r in rows})
        title_counts = Counter(r.get("title") for r in rows)
        qpa = [sum(1 for r in rows if r.get("_original_answer", r["answer"]) == a) for a in {r.get("_original_answer", r["answer"]) for r in rows}]
        answer_groups = {(r.get("title"), r.get("_original_answer", r["answer"])) for r in rows}
        inflated_groups = {(r.get("title"), r.get("_original_answer", r["answer"])) for r in rows if r["answer"] != r.get("_original_answer", r["answer"])}
        pct_real = 100 * len(inflated_groups) / max(len(answer_groups), 1)
        distinct_confusers = len({r["answer"].split("\n\n", 1)[0] for r in rows if r["answer"] != r.get("_original_answer", r["answer"])})
        raw_n = len(avia_raw) if name.startswith(("l_avia", "s_avia")) else len(cqad_raw)
        raw_rows = avia_raw if name.startswith(("l_avia", "s_avia")) else cqad_raw
        title_removed = sum(1 for r in raw_rows if r.get("title") == r.get("answer"))
        chapter_removed = max(0, len(avia) - len(small)) if name.startswith("s_avia") else 0
        print(f"{name}: input_rows={raw_n} title_equals_answer_removed={title_removed} after_title_filter={raw_n-title_removed} questions={len(rows)} unique_answers={uniq_answers} titles={len(title_counts)} questions_per_answer={min(qpa, default=0)}..{max(qpa, default=0)} mean={sum(qpa)/len(qpa) if qpa else 0:.2f} dedup_removed={removed} chapter_filtered={chapter_removed} inflation={pct_real:.1f}% (requested {pct}%), distinct_confusers={distinct_confusers}")
        full_overlap = _prefix_overlap([r["answer"] for r in rows])
        for seed in SEEDS:
            split_dir = out / "splits" / f"seed_{seed}"
            train_path, test_path = split_dir / "train.json", split_dir / "test.json"
            have_split = train_path.exists() and test_path.exists()
            if have_split:
                try:
                    train, test = _read(train_path), _read(test_path)
                except (json.JSONDecodeError, OSError):
                    have_split = False
            if force or not have_split:
                train, test = train_test_split(rows, test_size=0.2, random_state=seed)
                train, test = sorted(train, key=lambda x: x["qid"]), sorted(test, key=lambda x: x["qid"])
                _write_json(train_path, [{k: v for k, v in row.items() if k != "_original_answer"} for row in train])
                _write_json(test_path, [{k: v for k, v in row.items() if k != "_original_answer"} for row in test])
            if name == "s_avia" or name in {f"s_avia_{level}" for level in (25, 50, 75, 100)}:
                split_qids[(name, seed)] = {r["qid"] for r in test}
            by_title = defaultdict(list)
            for row in train:
                by_title[row["title"]].append(row)
            slugs = [_slug(t) for t in by_title]
            if len(slugs) != len(set(slugs)):
                raise ValueError(f"Title slug collision in {name} seed {seed}")
            title_dir = split_dir / "by_title"
            if force and title_dir.exists():
                import shutil
                shutil.rmtree(title_dir)
            title_dir.mkdir(parents=True, exist_ok=True)
            for title, title_rows in by_title.items():
                target = title_dir / f"{_slug(title)}.json"
                if force or not target.exists():
                    _write_json(target, [{k: v for k, v in row.items() if k != "_original_answer"} for row in title_rows])
            found = list(title_dir.glob("*.json"))
            if len(found) != len(by_title) or {p.stem for p in found} != set(slugs):
                raise ValueError(f"by_title contents do not match train titles for {name} seed {seed}")
            original_answers = {r["qid"]: r.get("_original_answer", r["answer"]) for r in rows}
            print(f"  seed={seed}: train={len(train)} test={len(test)} by_title={len(found)} max_jaccard={_jaccard_summary(train,test,original_answers)}")
            title_overlaps = [_prefix_overlap([r["answer"] for r in values]) for values in by_title.values()]
            avg_unique = sum(x["unique_prefixes"] for x in title_overlaps) / max(len(title_overlaps), 1)
            avg_cases = sum(x["total_cases"] for x in title_overlaps) / max(len(title_overlaps), 1)
            print(f"    overshadowing prefix_k=5: UF={full_overlap['unique_prefixes']} TF={full_overlap['total_cases']} US_avg={avg_unique:.3f} TS_avg={avg_cases:.3f}")
    for seed in SEEDS:
        base_qids = split_qids[("s_avia", seed)]
        variant_qids = [split_qids[(f"s_avia_{level}", seed)] for level in (25, 50, 75, 100)]
        if any(qids != base_qids for qids in variant_qids):
            raise AssertionError(f"Base and inflated test splits have different qids for seed {seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    build_datasets(args.force)


if __name__ == "__main__":
    main()
