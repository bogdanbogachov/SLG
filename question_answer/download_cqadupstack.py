"""Download the CQADupStack dataset in BEIR format.

This module only downloads and extracts the dataset. It does not convert the
data into the local QA schema or create train/test splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency.
    tqdm = None


LOGGER = logging.getLogger(__name__)

DEFAULT_CQADUPSTACK_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
)
DEFAULT_OUTPUT_DIR = Path("data/cqadupstack")

CQADUPSTACK_SUBFORUMS = (
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
)


def _expected_files(root: Path) -> list[Path]:
    return [
        root / subforum / filename
        for subforum in CQADUPSTACK_SUBFORUMS
        for filename in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    ]


def _dataset_is_complete(root: Path) -> bool:
    return root.is_dir() and all(path.is_file() for path in _expected_files(root))


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            )

        try:
            with temporary_destination.open("wb") as handle:
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


def _safe_extract(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = output_root / member.filename
            resolved_member_path = member_path.resolve()
            if output_root != resolved_member_path and output_root not in resolved_member_path.parents:
                raise ValueError(f"Refusing to extract unsafe zip member: {member.filename}")

        archive.extractall(output_root)


def _write_manifest(
    output_dir: Path,
    dataset_dir: Path,
    source_url: str,
    zip_path: Path,
    kept_zip: bool,
) -> dict[str, Any]:
    manifest = {
        "dataset": "CQADupStack",
        "format": "BEIR",
        "source_url": source_url,
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "archive_path": str(zip_path) if kept_zip else None,
        "subforums": list(CQADUPSTACK_SUBFORUMS),
        "files_per_subforum": ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"],
    }

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    return manifest


def download_cqadupstack(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    url: str = DEFAULT_CQADUPSTACK_URL,
    keep_zip: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Download and extract CQADupStack.

    Args:
        output_dir: Directory where the dataset should be stored.
        url: Source URL for the BEIR CQADupStack archive.
        keep_zip: Keep the downloaded zip archive after extraction.
        force: Redownload and re-extract even when the dataset already exists.

    Returns:
        A manifest describing the downloaded dataset location and expected files.
    """
    output_path = Path(output_dir)
    dataset_dir = output_path if output_path.name == "cqadupstack" else output_path / "cqadupstack"
    extract_dir = dataset_dir.parent
    zip_path = dataset_dir.with_suffix(".zip")

    if _dataset_is_complete(dataset_dir) and not force:
        LOGGER.info("CQADupStack already exists at %s", dataset_dir)
        return _write_manifest(output_path, dataset_dir, url, zip_path, keep_zip and zip_path.exists())

    extract_dir.mkdir(parents=True, exist_ok=True)

    if force:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        if zip_path.exists():
            zip_path.unlink()

    if not zip_path.exists():
        LOGGER.info("Downloading CQADupStack from %s", url)
        _download_file(url, zip_path)

    LOGGER.info("Extracting CQADupStack to %s", extract_dir)
    _safe_extract(zip_path, extract_dir)

    if not _dataset_is_complete(dataset_dir):
        missing = [str(path) for path in _expected_files(dataset_dir) if not path.is_file()]
        raise RuntimeError(
            "CQADupStack extraction completed, but expected files are missing: "
            + ", ".join(missing[:10])
        )

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    manifest = _write_manifest(output_path, dataset_dir, url, zip_path, keep_zip and zip_path.exists())
    LOGGER.info("CQADupStack downloaded to %s", dataset_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CQADupStack in BEIR format.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="CQADupStack directory, or a parent directory where cqadupstack/ should be stored.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_CQADUPSTACK_URL,
        help="Dataset zip URL.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip archive after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if the dataset already exists.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    manifest = download_cqadupstack(
        output_dir=args.output_dir,
        url=args.url,
        keep_zip=args.keep_zip,
        force=args.force,
    )
    print(json.dumps(manifest, indent=4))


if __name__ == "__main__":
    main()
