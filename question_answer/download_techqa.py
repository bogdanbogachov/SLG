"""Download the IBM **TechQA** dataset (for the retrieval-centric trustworthy-QA
direction — see CLAUDE.md "Candidate direction: retrieval-centric trustworthy QA").

TechQA (Castelli et al., ACL 2020) is real IBM technical-support QA grounded in
IBM Technotes. Its decisive property for this project: every question is flagged
**answerable / unanswerable**, so the *out-of-scope* evaluation set required to
demonstrate calibrated abstention comes ready-made — no synthesis, no partitioning.

This module mirrors ``question_answer/download_stackexchange.py``: it downloads a
single archive from a configurable base URL, extracts the QA JSON + the technote
corpus, and builds ``question_answer/qa_techqa.json`` in the **same record schema**
as ``build_stackexchange_qa.py`` (``chapter/title/question/answer/source_url/
license`` ...), plus two extra fields the abstention story needs:

    "answerable": bool      # False  -> gold out-of-scope question (system should abstain)
    "document":   str|None  # gold technote id that contains the answer (retrieval target)

Only the standard library is required (download via ``urllib``; extraction via
``tarfile``/``zipfile`` — no 7z needed, unlike the Stack Exchange dumps).

Usage:
    python -m question_answer.download_techqa --dumps_dir data/techqa
    python -m question_answer.download_techqa --dumps_dir data/techqa \
        --url https://<host>/<techqa-archive>.tar.gz --out question_answer/qa_techqa.json

IMPORTANT — verify before use:
    TechQA is distributed by IBM and historically required accepting IBM's terms
    (registration / a research-data agreement); the public mirror URL has moved
    over time. ``TECHQA_URL`` below is a *placeholder default* — confirm the
    current download location and licence (commonly cited as CDLA-Sharing-1.0)
    and pass the real archive via ``--url``. The parser tolerates the documented
    TechQA JSON field names but will ``--skip`` records it cannot read.
"""

import argparse
import json
import os
import sys
import tarfile
import urllib.request
import zipfile

# Placeholder — TechQA's canonical host has changed over time. Override with --url.
TECHQA_URL = "https://ibm.ent.box.com/v/techqa-data"  # VERIFY before use
LICENSE = "CDLA-Sharing-1.0 (verify)"  # IBM TechQA — confirm current terms

# The QA JSON files inside the archive (train + dev carry the answerable flag).
_QA_MEMBER_HINTS = ("training_Q_A.json", "dev_Q_A.json", "Q_A.json")
# The technote corpus (doc_id -> {title, text}); used to attach the gold answer text.
_CORPUS_MEMBER_HINTS = ("technotes.json", "training_dev_technotes.json", "corpus.json")


# --------------------------------------------------------------------------- #
# download + extract
# --------------------------------------------------------------------------- #
def _progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def download(url: str, dumps_dir: str) -> str:
    """Download the TechQA archive into dumps_dir; return its path (skip if present)."""
    os.makedirs(dumps_dir, exist_ok=True)
    name = os.path.basename(url.split("?", 1)[0]) or "techqa_archive"
    archive = os.path.join(dumps_dir, name)
    if os.path.isfile(archive) and os.path.getsize(archive) > 0:
        print(f"  archive already present: {archive}")
        return archive
    print(f"  downloading {url}")
    tmp = archive + ".part"
    urllib.request.urlretrieve(url, tmp, _progress)
    sys.stdout.write("\n")
    os.replace(tmp, archive)
    return archive


def _extract_all(archive: str, out_dir: str) -> None:
    """Extract the archive (tar.* or zip) into out_dir; skip if already extracted."""
    marker = os.path.join(out_dir, ".extracted")
    if os.path.isfile(marker):
        print(f"  already extracted: {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"  extracting -> {out_dir}")
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out_dir)  # noqa: S202 — trusted research archive
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:
        raise RuntimeError(
            f"{archive} is neither a tar nor a zip archive; if TechQA now ships in "
            "another container, extract it manually into this directory."
        )
    open(marker, "w").close()


def _find_member(root: str, hints) -> str | None:
    """Return the first file under root whose basename matches one of hints."""
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f in hints:
                return os.path.join(dirpath, f)
    # looser fallback: any file that *ends with* a hint
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if any(f.endswith(h) for h in hints):
                return os.path.join(dirpath, f)
    return None


# --------------------------------------------------------------------------- #
# parse TechQA JSON -> unified QA records
# --------------------------------------------------------------------------- #
def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _corpus_index(corpus_path: str | None) -> dict:
    """doc_id -> {title, text}. TechQA field names vary; tolerate the common ones."""
    if not corpus_path or not os.path.isfile(corpus_path):
        return {}
    raw = _load_json(corpus_path)
    # TechQA corpora appear both as a dict keyed by doc-id and as a list of docs.
    items = raw.items() if isinstance(raw, dict) else ((None, d) for d in raw)
    index = {}
    for key, d in items:
        if not isinstance(d, dict):
            continue
        did = key or d.get("id") or d.get("DOCUMENT_ID") or d.get("doc_id")
        if did is None:
            continue
        index[str(did)] = {
            "title": d.get("title") or d.get("TITLE") or "",
            "text": d.get("text") or d.get("TEXT") or d.get("body") or "",
        }
    return index


def _get(q: dict, *keys, default=None):
    for k in keys:
        if k in q and q[k] not in (None, ""):
            return q[k]
    return default


def build_records(qa_path: str, corpus: dict, split_label: str) -> list:
    """Convert one TechQA Q_A.json file into unified QA records."""
    raw = _load_json(qa_path)
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("questions", []))
    records = []
    for q in items:
        if not isinstance(q, dict):
            continue
        title = _get(q, "QUESTION_TITLE", "question_title", "title", default="")
        body = _get(q, "QUESTION_TEXT", "question_text", "body", default="")
        q_text = (str(title).strip() + "\n\n" + str(body).strip()).strip()
        if not q_text:
            continue
        ans_flag = _get(q, "ANSWERABLE", "answerable", default="Y")
        answerable = str(ans_flag).strip().upper() in ("Y", "YES", "TRUE", "1")
        answer = _get(q, "ANSWER", "answer", "ANSWER_TEXT", default="") or ""
        gold_doc = _get(q, "DOCUMENT", "document", "doc_id", default=None)
        # If the answer text is absent but a gold doc is known, fall back to its body.
        if answerable and not answer and gold_doc and str(gold_doc) in corpus:
            answer = corpus[str(gold_doc)]["text"]
        qid = _get(q, "QUESTION_ID", "question_id", "id", default="")
        records.append({
            "chapter": "TechQA",
            "title": split_label,        # expert label -> slug(title) = expert id
            "question": q_text,
            "answer": (answer or "").strip(),
            "answerable": answerable,     # False -> gold out-of-scope (abstain)
            "document": str(gold_doc) if gold_doc is not None else None,
            "source_url": f"techqa://{qid}",
            "license": LICENSE,
        })
    return records


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(dumps_dir="data/techqa", url=TECHQA_URL,
        out="question_answer/qa_techqa.json", split_label="techqa",
        keep_archive=False):
    """Download + extract + build the unified TechQA QA set at ``out``."""
    print("TechQA")
    archive = download(url, dumps_dir)
    extract_dir = os.path.join(dumps_dir, "extracted")
    _extract_all(archive, extract_dir)
    if not keep_archive and os.path.isfile(archive):
        os.remove(archive)

    corpus = _corpus_index(_find_member(extract_dir, _CORPUS_MEMBER_HINTS))
    all_records = []
    for member in sorted(set(_iter_qa_members(extract_dir))):
        label = _split_label_for(member, split_label)
        print(f"  building from {os.path.basename(member)} (title={label})")
        try:
            all_records.extend(build_records(member, corpus, label))
        except Exception as e:  # keep going on a single-file failure
            print(f"  FAILED ({e}); skipping {member}.")

    if not all_records:
        raise SystemExit(
            "No TechQA records built. Confirm the archive URL/format and that a "
            "*_Q_A.json file was extracted (see _QA_MEMBER_HINTS)."
        )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    n_ans = sum(1 for r in all_records if r["answerable"])
    n_unans = len(all_records) - n_ans
    print(f"\nWrote {len(all_records)} records -> {out}")
    print(f"  answerable: {n_ans}   unanswerable (gold out-of-scope): {n_unans}")
    print(f"License: {LICENSE} — verify IBM's current terms before redistributing.")
    print("The unanswerable questions are the ready-made abstention (out-of-scope) set.")


def _iter_qa_members(extract_dir: str):
    """Yield every *_Q_A.json under extract_dir (train + dev)."""
    for dirpath, _dirs, files in os.walk(extract_dir):
        for f in files:
            if f in _QA_MEMBER_HINTS or f.endswith("_Q_A.json"):
                yield os.path.join(dirpath, f)


def _split_label_for(member_path: str, default: str) -> str:
    """Derive a per-file expert label so train/dev stay distinguishable if wanted."""
    base = os.path.basename(member_path).lower()
    if base.startswith("training"):
        return f"{default}_train"
    if base.startswith("dev"):
        return f"{default}_dev"
    return default


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps_dir", default="data/techqa")
    ap.add_argument("--url", default=TECHQA_URL,
                    help="TechQA archive URL (VERIFY — the placeholder default will not work)")
    ap.add_argument("--out", default="question_answer/qa_techqa.json")
    ap.add_argument("--split_label", default="techqa",
                    help="base expert label (title) for the built records")
    ap.add_argument("--keep_archive", action="store_true",
                    help="keep the downloaded archive after extraction")
    args = ap.parse_args()
    run(dumps_dir=args.dumps_dir, url=args.url, out=args.out,
        split_label=args.split_label, keep_archive=args.keep_archive)


if __name__ == "__main__":
    main()
