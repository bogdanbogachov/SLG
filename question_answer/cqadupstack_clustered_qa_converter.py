"""Build clustered QA data from CQADupStack and StackExchange dumps.

CQADupStack's BEIR files provide duplicate-question links, not answers. This
converter uses those links to form question clusters and joins each cluster
against local StackExchange Posts.xml dumps to extract one canonical answer per
cluster. The output keeps the local row-wise QA schema:

    {"chapter": ..., "title": ..., "question": ..., "answer": ...}

By default only clusters with at least four question variants are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import py7zr
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency.
    tqdm = None


LOGGER = logging.getLogger(__name__)

DEFAULT_CQADUPSTACK_DIR = Path("data/cqadupstack")
DEFAULT_STACKEXCHANGE_DIR = Path("data/stackexchange")
DEFAULT_OUTPUT_DIR = Path("question_answer/cqadupstack_clustered")
DEFAULT_ARCHIVE_DIR = Path("data/stackexchange/_archives")
STACKEXCHANGE_ARCHIVE_BASE_URL = "https://archive.org/download/stackexchange"

DOMAIN_TO_DUMP = {
    "android": "android.stackexchange.com",
    "english": "english.stackexchange.com",
    "gaming": "gaming.stackexchange.com",
    "gis": "gis.stackexchange.com",
    "mathematica": "mathematica.stackexchange.com",
    "physics": "physics.stackexchange.com",
    "programmers": "softwareengineering.stackexchange.com",
    "stats": "stats.stackexchange.com",
    "tex": "tex.stackexchange.com",
    "unix": "unix.stackexchange.com",
    "webmasters": "webmasters.stackexchange.com",
    "wordpress": "wordpress.stackexchange.com",
}


@dataclass(frozen=True)
class QuestionText:
    post_id: str
    title: str
    body: str


@dataclass(frozen=True)
class StackQuestion:
    post_id: str
    title: str
    body: str
    accepted_answer_id: str | None
    score: int


@dataclass(frozen=True)
class StackAnswer:
    post_id: str
    parent_id: str
    body: str
    score: int


class _StackExchangeHTMLToText(HTMLParser):
    _BLOCK_TAGS = {
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "pre",
        "table",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        if tag == "li":
            self._parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS or tag in {"code", "pre"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._parts))
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(value: str) -> str:
    parser = _StackExchangeHTMLToText()
    parser.feed(value or "")
    parser.close()
    return parser.get_text()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(destination.suffix + ".part")

    headers = {}
    existing_size = temporary_destination.stat().st_size if temporary_destination.exists() else 0
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        if existing_size and response.status_code == 200:
            existing_size = 0
        response.raise_for_status()
        mode = "ab" if existing_size else "wb"
        content_length = int(response.headers.get("content-length", 0))
        total = existing_size + content_length if content_length else None

        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=total,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            )

        try:
            with temporary_destination.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

    temporary_destination.replace(destination)


def ensure_stackexchange_posts_xml(
    site_name: str,
    stackexchange_dir: str | Path = DEFAULT_STACKEXCHANGE_DIR,
    archive_dir: str | Path = DEFAULT_ARCHIVE_DIR,
    keep_archive: bool = False,
) -> Path:
    """Ensure a StackExchange site's Posts.xml exists locally."""
    stackexchange_path = Path(stackexchange_dir)
    site_dir = stackexchange_path / site_name
    posts_xml = site_dir / "Posts.xml"
    if posts_xml.is_file():
        return posts_xml

    archive_path = Path(archive_dir) / f"{site_name}.7z"
    url = f"{STACKEXCHANGE_ARCHIVE_BASE_URL}/{site_name}.7z"
    if not archive_path.is_file():
        LOGGER.info("Downloading %s", url)
        _download_file(url, archive_path)

    LOGGER.info("Extracting Posts.xml from %s", archive_path)
    site_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = set(archive.getnames())
        if "Posts.xml" not in names:
            raise RuntimeError(f"Posts.xml not found in {archive_path}")
        archive.extract(path=site_dir, targets=["Posts.xml"])

    if not posts_xml.is_file():
        raise RuntimeError(f"Expected extracted Posts.xml at {posts_xml}")

    if not keep_archive:
        archive_path.unlink(missing_ok=True)

    return posts_xml


def _load_question_texts(domain_dir: Path) -> dict[str, QuestionText]:
    question_texts: dict[str, QuestionText] = {}

    with (domain_dir / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            post_id = str(row["_id"])
            metadata = row.get("metadata", {})
            metadata_for_post = metadata.get(post_id, {}) if isinstance(metadata, dict) else {}
            body = metadata_for_post.get("body", "") if isinstance(metadata_for_post, dict) else ""
            question_texts[post_id] = QuestionText(
                post_id=post_id,
                title=str(row.get("text", "")),
                body=str(body),
            )

    with (domain_dir / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            post_id = str(row["_id"])
            question_texts[post_id] = QuestionText(
                post_id=post_id,
                title=str(row.get("title", "")),
                body=str(row.get("text", "")),
            )

    return question_texts


def _load_duplicate_clusters(domain_dir: Path, min_cluster_size: int) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()

    with (domain_dir / "qrels" / "test.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row.get("score") or 0) <= 0:
                continue
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            nodes.add(query_id)
            nodes.add(corpus_id)
            adjacency[query_id].add(corpus_id)
            adjacency[corpus_id].add(query_id)

    clusters: list[list[str]] = []
    seen: set[str] = set()
    for node in sorted(nodes, key=lambda value: int(value) if value.isdigit() else value):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for next_node in adjacency[current]:
                if next_node not in seen:
                    seen.add(next_node)
                    stack.append(next_node)
        if len(component) >= min_cluster_size:
            clusters.append(sorted(component, key=lambda value: int(value) if value.isdigit() else value))

    return clusters


def _parse_posts_xml(
    posts_xml: Path,
    target_question_ids: set[str],
) -> tuple[dict[str, StackQuestion], dict[str, list[StackAnswer]]]:
    questions: dict[str, StackQuestion] = {}
    answers_by_parent: dict[str, list[StackAnswer]] = defaultdict(list)

    for _, elem in ET.iterparse(posts_xml, events=("end",)):
        if elem.tag != "row":
            elem.clear()
            continue

        post_type = elem.attrib.get("PostTypeId")
        post_id = elem.attrib.get("Id", "")
        if post_type == "1" and post_id in target_question_ids:
            questions[post_id] = StackQuestion(
                post_id=post_id,
                title=_html_to_text(elem.attrib.get("Title", "")),
                body=_html_to_text(elem.attrib.get("Body", "")),
                accepted_answer_id=elem.attrib.get("AcceptedAnswerId"),
                score=int(elem.attrib.get("Score") or 0),
            )
        elif post_type == "2":
            parent_id = elem.attrib.get("ParentId", "")
            if parent_id in target_question_ids:
                answers_by_parent[parent_id].append(
                    StackAnswer(
                        post_id=post_id,
                        parent_id=parent_id,
                        body=_html_to_text(elem.attrib.get("Body", "")),
                        score=int(elem.attrib.get("Score") or 0),
                    )
                )

        elem.clear()

    return questions, answers_by_parent


def _select_answer(
    question: StackQuestion,
    answers: list[StackAnswer],
) -> tuple[StackAnswer | None, str | None]:
    if not answers:
        return None, None
    if question.accepted_answer_id:
        for answer in answers:
            if answer.post_id == question.accepted_answer_id:
                return answer, "accepted"
    return max(answers, key=lambda answer: (answer.score, int(answer.post_id))), "top_score"


def _question_from_text(text: QuestionText, fallback: StackQuestion | None = None) -> str:
    title = _normalize_text(text.title or (fallback.title if fallback else ""))
    body = _normalize_text(text.body or (fallback.body if fallback else ""))
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _pick_cluster_answer(
    cluster: list[str],
    stack_questions: dict[str, StackQuestion],
    selected_answers: dict[str, tuple[StackAnswer, str]],
) -> tuple[str, StackQuestion, StackAnswer, str] | None:
    candidates = []
    for post_id in cluster:
        stack_question = stack_questions.get(post_id)
        selected = selected_answers.get(post_id)
        if stack_question is None or selected is None:
            continue
        answer, selection = selected
        selection_rank = 1 if selection == "accepted" else 0
        candidates.append((selection_rank, stack_question.score, answer.score, int(post_id), stack_question, answer, selection))

    if not candidates:
        return None

    _, _, _, post_id_int, stack_question, answer, selection = max(candidates)
    return str(post_id_int), stack_question, answer, selection


def _make_row(chapter: str, title: str, question: str, answer: str) -> dict[str, str] | None:
    title = _normalize_text(title)
    question = question.strip()
    answer = answer.strip()
    if not title or not question or not answer:
        return None
    return {
        "chapter": chapter,
        "title": title,
        "question": question,
        "answer": answer,
    }


def convert_cqadupstack_clusters_to_qa(
    cqadupstack_dir: str | Path = DEFAULT_CQADUPSTACK_DIR,
    stackexchange_dir: str | Path = DEFAULT_STACKEXCHANGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_cluster_size: int = 4,
    domains: list[str] | None = None,
    download_missing: bool = False,
    keep_archives: bool = False,
) -> dict[str, Any]:
    """Create QA rows where each selected answer has several duplicate questions."""
    cqadupstack_path = Path(cqadupstack_dir)
    stackexchange_path = Path(stackexchange_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_domains = domains or sorted(DOMAIN_TO_DUMP)
    rows: list[dict[str, str]] = []
    seen_rows: set[tuple[str, str, str]] = set()
    domain_reports: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for domain in selected_domains:
        if domain not in DOMAIN_TO_DUMP:
            raise ValueError(f"Unsupported CQADupStack domain: {domain}")

        domain_dir = cqadupstack_path / domain
        if not domain_dir.is_dir():
            domain_reports.append({"domain": domain, "status": "missing_cqadupstack_domain"})
            continue

        site_name = DOMAIN_TO_DUMP[domain]
        posts_xml = stackexchange_path / site_name / "Posts.xml"
        if not posts_xml.is_file():
            if not download_missing:
                domain_reports.append({
                    "domain": domain,
                    "site_name": site_name,
                    "status": "missing_posts_xml",
                    "posts_xml": str(posts_xml),
                })
                continue
            posts_xml = ensure_stackexchange_posts_xml(
                site_name=site_name,
                stackexchange_dir=stackexchange_path,
                keep_archive=keep_archives,
            )

        LOGGER.info("Processing CQADupStack domain %s with %s", domain, posts_xml)
        question_texts = _load_question_texts(domain_dir)
        clusters = _load_duplicate_clusters(domain_dir, min_cluster_size=min_cluster_size)
        target_ids = {post_id for cluster in clusters for post_id in cluster}
        stack_questions, answers_by_parent = _parse_posts_xml(posts_xml, target_ids)

        selected_answers: dict[str, tuple[StackAnswer, str]] = {}
        selection_counts: Counter[str] = Counter()
        for post_id, stack_question in stack_questions.items():
            selected_answer, selection = _select_answer(stack_question, answers_by_parent.get(post_id, []))
            if selected_answer is None or selection is None:
                continue
            selected_answers[post_id] = (selected_answer, selection)
            selection_counts[selection] += 1

        used_clusters = 0
        skipped_no_answer = 0
        domain_rows = 0
        cluster_size_counts: Counter[int] = Counter()
        answer_question_counts: Counter[int] = Counter()

        for index, cluster in enumerate(clusters, start=1):
            picked = _pick_cluster_answer(cluster, stack_questions, selected_answers)
            if picked is None:
                skipped_no_answer += 1
                continue

            answer_source_id, source_question, answer, selection = picked
            title = source_question.title
            answer_text = answer.body
            pending_rows: list[dict[str, str]] = []
            pending_row_keys: list[tuple[str, str, str]] = []
            for post_id in cluster:
                text = question_texts.get(post_id)
                if text is None:
                    stack_question = stack_questions.get(post_id)
                    if stack_question is None:
                        continue
                    text = QuestionText(post_id=post_id, title=stack_question.title, body=stack_question.body)
                question = _question_from_text(text, stack_questions.get(post_id))
                row = _make_row(
                    chapter=f"CQADupStack - {domain}",
                    title=title,
                    question=question,
                    answer=answer_text,
                )
                if row is None:
                    continue
                row_key = (row["chapter"], row["question"], row["answer"])
                if row_key in seen_rows:
                    continue
                pending_rows.append(row)
                pending_row_keys.append(row_key)

            if len(pending_rows) >= min_cluster_size:
                rows.extend(pending_rows)
                seen_rows.update(pending_row_keys)
                used_clusters += 1
                domain_rows += len(pending_rows)
                cluster_size_counts[len(cluster)] += 1
                answer_question_counts[len(pending_rows)] += 1
                cluster_rows.append({
                    "domain": domain,
                    "cluster_id": f"{domain}-{index:05d}",
                    "cluster_size": len(cluster),
                    "qa_question_count": len(pending_rows),
                    "answer_source_question_id": answer_source_id,
                    "answer_selection": selection,
                    "question_ids": " ".join(cluster),
                    "title": title,
                })
            else:
                skipped_no_answer += 1

        domain_reports.append({
            "domain": domain,
            "site_name": site_name,
            "status": "converted",
            "posts_xml": str(posts_xml),
            "candidate_clusters": len(clusters),
            "used_clusters": used_clusters,
            "skipped_clusters_without_enough_answered_questions": skipped_no_answer,
            "rows": domain_rows,
            "stack_questions_found": len(stack_questions),
            "questions_with_selected_answers": len(selected_answers),
            "answer_selection_counts": dict(sorted(selection_counts.items())),
            "cluster_size_histogram": dict(sorted(cluster_size_counts.items())),
            "answer_question_count_histogram": dict(sorted(answer_question_counts.items())),
        })

    rows = sorted(rows, key=lambda row: (row["chapter"], row["title"].lower(), row["question"].lower()))

    qa_path = output_path / "qa.json"
    with qa_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=4, ensure_ascii=False)

    clusters_path = output_path / "clusters.csv"
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "cluster_id",
                "cluster_size",
                "qa_question_count",
                "answer_source_question_id",
                "answer_selection",
                "question_ids",
                "title",
            ],
        )
        writer.writeheader()
        writer.writerows(cluster_rows)

    report = {
        "dataset": "CQADupStack clustered QA",
        "cqadupstack_dir": str(cqadupstack_path),
        "stackexchange_dir": str(stackexchange_path),
        "output_dir": str(output_path),
        "qa_file": str(qa_path),
        "clusters_file": str(clusters_path),
        "min_cluster_size": min_cluster_size,
        "total_rows": len(rows),
        "total_used_clusters": len(cluster_rows),
        "domains": domain_reports,
        "notes": [
            "Each duplicate-question cluster contributes one canonical answer.",
            "The canonical answer is selected from an answered question in the cluster, preferring accepted answers.",
            "Every retained answer has at least min_cluster_size question rows in qa.json.",
        ],
    }

    with (output_path / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CQADupStack duplicate clusters into local QA rows using StackExchange dumps."
    )
    parser.add_argument("--cqadupstack-dir", default=str(DEFAULT_CQADUPSTACK_DIR))
    parser.add_argument("--stackexchange-dir", default=str(DEFAULT_STACKEXCHANGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--domains", nargs="*", default=None, help="CQADupStack domains to process.")
    parser.add_argument("--download-missing", action="store_true", help="Download missing StackExchange Posts.xml dumps.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded .7z archives after extraction.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    report = convert_cqadupstack_clusters_to_qa(
        cqadupstack_dir=args.cqadupstack_dir,
        stackexchange_dir=args.stackexchange_dir,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        domains=args.domains,
        download_missing=args.download_missing,
        keep_archives=args.keep_archives,
    )
    print(
        "Converted CQADupStack clustered QA: "
        f"{report['total_rows']} rows from {report['total_used_clusters']} clusters."
    )


if __name__ == "__main__":
    main()
