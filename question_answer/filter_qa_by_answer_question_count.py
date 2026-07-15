"""Filter local QA rows by the number of questions per answer."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path("question_answer/cqadupstack_clustered/qa.json")
DEFAULT_OUTPUT_FILE = Path("question_answer/cqadupstack_clustered/qa_min20.json")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def filter_qa_by_answer_question_count(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    min_questions_per_answer: int = 20,
    group_by_chapter: bool = True,
) -> dict[str, Any]:
    """Keep QA rows whose answer group has enough distinct questions."""
    input_path = Path(input_file)
    output_path = Path(output_file)

    with input_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        chapter = str(row.get("chapter", "")) if group_by_chapter else ""
        answer = _normalize_text(str(row.get("answer", "")))
        groups[(chapter, answer)].append(row)

    kept_rows: list[dict[str, Any]] = []
    kept_group_sizes: Counter[int] = Counter()
    kept_groups_by_chapter: Counter[str] = Counter()
    kept_rows_by_chapter: Counter[str] = Counter()

    for (chapter, _), group_rows in groups.items():
        question_count = len({_normalize_text(str(row.get("question", ""))) for row in group_rows})
        if question_count < min_questions_per_answer:
            continue

        kept_rows.extend(group_rows)
        kept_group_sizes[question_count] += 1
        kept_groups_by_chapter[chapter] += 1
        kept_rows_by_chapter[chapter] += len(group_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(kept_rows, handle, indent=4, ensure_ascii=False)

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "min_questions_per_answer": min_questions_per_answer,
        "group_by_chapter": group_by_chapter,
        "input_rows": len(rows),
        "input_answer_groups": len(groups),
        "kept_rows": len(kept_rows),
        "kept_answer_groups": sum(kept_group_sizes.values()),
        "kept_group_size_histogram": dict(sorted(kept_group_sizes.items())),
        "kept_rows_by_chapter": dict(sorted(kept_rows_by_chapter.items())),
        "kept_groups_by_chapter": dict(sorted(kept_groups_by_chapter.items())),
    }

    report_path = output_path.with_name(output_path.stem + "_report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    report["report_file"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter QA rows by distinct questions per answer.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--min-questions-per-answer", type=int, default=20)
    parser.add_argument(
        "--no-group-by-chapter",
        action="store_true",
        help="Group only by normalized answer text, not by chapter/domain.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = filter_qa_by_answer_question_count(
        input_file=args.input_file,
        output_file=args.output_file,
        min_questions_per_answer=args.min_questions_per_answer,
        group_by_chapter=not args.no_group_by_chapter,
    )
    print(
        "Filtered QA: "
        f"{report['kept_rows']} rows from "
        f"{report['kept_answer_groups']} answer groups."
    )


if __name__ == "__main__":
    main()
