"""Download Stack Exchange community data dumps (for the real-world QA set).

Fetches per-community ``<host>.7z`` archives from the Internet Archive and
extracts just ``Posts.xml`` into the layout the extractor expects:

    <dumps_dir>/<host>/Posts.xml

Then build the QA set with ``question_answer/build_stackexchange_qa.py``.

Only the standard library is required to download. Extraction needs **either**
the ``py7zr`` package **or** a system ``7z``/``7za``/``7zr`` binary.

Usage:
    # the default engineering-relevant communities
    python -m question_answer.download_stackexchange --dumps_dir data/stackexchange

    # or a chosen subset
    python -m question_answer.download_stackexchange --dumps_dir data/stackexchange \
        --communities aviation.stackexchange.com engineering.stackexchange.com

Note: Stack Exchange content is CC BY-SA. The most recent snapshots may require
going through Stack Exchange's own request flow rather than archive.org — verify
availability/license before use.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

ARCHIVE_BASE = "https://archive.org/download/stackexchange"

# Engineering-relevant communities (see question_answer/build_stackexchange_qa.py
# for their human-readable expert labels). Physics/Chemistry are the large ones.
DEFAULT_COMMUNITIES = [
    "aviation.stackexchange.com",
    "engineering.stackexchange.com",
    "electronics.stackexchange.com",
    "physics.stackexchange.com",
    "chemistry.stackexchange.com",
    "space.stackexchange.com",
    "robotics.stackexchange.com",
    "3dprinting.stackexchange.com",
    "dsp.stackexchange.com",
    "mechanics.stackexchange.com",
    "networkengineering.stackexchange.com",
    "scicomp.stackexchange.com",
]


def _progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def download(host: str, dumps_dir: str, base: str) -> str:
    """Download <host>.7z into dumps_dir; return the archive path (skip if present)."""
    os.makedirs(dumps_dir, exist_ok=True)
    archive = os.path.join(dumps_dir, f"{host}.7z")
    if os.path.isfile(archive) and os.path.getsize(archive) > 0:
        print(f"  archive already present: {archive}")
        return archive
    url = f"{base}/{host}.7z"
    print(f"  downloading {url}")
    tmp = archive + ".part"
    urllib.request.urlretrieve(url, tmp, _progress)
    sys.stdout.write("\n")
    os.replace(tmp, archive)
    return archive


def _extract_py7zr(archive: str, out_dir: str) -> bool:
    try:
        import py7zr
    except ImportError:
        return False
    with py7zr.SevenZipFile(archive, "r") as z:
        names = z.getnames()
        target = next((n for n in names if os.path.basename(n) == "Posts.xml"), None)
        if not target:
            raise RuntimeError(f"Posts.xml not found inside {archive}")
        z.extract(path=out_dir, targets=[target])
        # py7zr preserves internal paths; flatten to <out_dir>/Posts.xml
        src = os.path.join(out_dir, target)
        dst = os.path.join(out_dir, "Posts.xml")
        if src != dst:
            os.replace(src, dst)
    return True


def _extract_system_7z(archive: str, out_dir: str) -> bool:
    exe = next((b for b in ("7z", "7za", "7zr") if shutil.which(b)), None)
    if not exe:
        return False
    # `e` extracts the named file flat into -o<dir>
    subprocess.run([exe, "e", archive, "Posts.xml", f"-o{out_dir}", "-y"],
                   check=True, stdout=subprocess.DEVNULL)
    return True


def extract_posts(archive: str, out_dir: str) -> str:
    """Extract Posts.xml from the archive into out_dir; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "Posts.xml")
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  Posts.xml already extracted: {dst}")
        return dst
    print(f"  extracting Posts.xml -> {out_dir}")
    if _extract_py7zr(archive, out_dir) or _extract_system_7z(archive, out_dir):
        if not os.path.isfile(dst):
            raise RuntimeError(f"extraction did not yield {dst}")
        return dst
    raise SystemExit(
        "No 7z extractor available. Install `pip install py7zr` or a system "
        "7-Zip binary (7z/7za/7zr), then re-run."
    )


def run(dumps_dir="data/stackexchange", communities=None,
        archive_base=ARCHIVE_BASE, keep_archive=False):
    """Download + extract Posts.xml for each community into dumps_dir/<host>/."""
    communities = communities or DEFAULT_COMMUNITIES
    for host in communities:
        print(host)
        try:
            archive = download(host, dumps_dir, archive_base)
            extract_posts(archive, os.path.join(dumps_dir, host))
            if not keep_archive and os.path.isfile(archive):
                os.remove(archive)
        except Exception as e:  # keep going on a single-community failure
            print(f"  FAILED ({e}); skipping.")

    print("\nDone. Build the QA set with:")
    print(f"  python -m question_answer.build_stackexchange_qa "
          f"--dumps_dir {dumps_dir} --out question_answer/qa.json --cap 5000")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps_dir", default="data/stackexchange")
    ap.add_argument("--communities", nargs="*", default=DEFAULT_COMMUNITIES,
                    help="community hosts to fetch (default: engineering set)")
    ap.add_argument("--archive_base", default=ARCHIVE_BASE)
    ap.add_argument("--keep_archive", action="store_true",
                    help="keep the .7z after extraction (default: delete to save space)")
    args = ap.parse_args()
    run(dumps_dir=args.dumps_dir, communities=args.communities,
        archive_base=args.archive_base, keep_archive=args.keep_archive)


if __name__ == "__main__":
    main()
