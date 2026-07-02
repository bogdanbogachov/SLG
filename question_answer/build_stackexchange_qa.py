"""Build a real-world engineering QA set from Stack Exchange data dumps.

Turns per-community Stack Exchange ``Posts.xml`` files into ``question_answer/qa.json``
in this project's schema, so the existing ``--split_qa`` step produces the full
``qa_train.json`` / ``qa_test.json`` and then the per-expert ``split_by_title/``
files with no further changes.

Each community becomes one **expert** (``title`` = community label, so
``slug(title)`` is the expert id). Per question we keep the *accepted* answer,
or — if none — the highest-scored positive answer. Bodies are HTML-stripped.
Records are deduplicated (normalised question text) to keep the later 80/20 split
leak-free, then capped at ``--cap`` per expert (default 5000), taking the
highest-scored answers first.

Provenance for CC BY-SA attribution is preserved on every record: ``source_url``,
``author`` (answer), ``question_author``, ``tags``, ``score``, ``license``.

Layout expected (one folder per community, each holding its ``Posts.xml``):

    <dumps_dir>/
        aviation.stackexchange.com/Posts.xml
        engineering.stackexchange.com/Posts.xml
        electronics.stackexchange.com/Posts.xml
        ...

Usage:
    python -m question_answer.build_stackexchange_qa \
        --dumps_dir data/stackexchange --out question_answer/qa.json --cap 5000

Then:  python main.py --split_qa

Only the Python standard library is used.
"""

import argparse
import html
import json
import os
import random
import re
from collections import defaultdict
from html.parser import HTMLParser
from xml.etree.ElementTree import iterparse

# Nice, slug-friendly expert labels for known communities (folder host -> label).
# Anything not listed falls back to the host's first dotted segment, title-cased.
COMMUNITY_LABELS = {
    "aviation.stackexchange.com": "Aviation",
    "engineering.stackexchange.com": "Engineering",
    "electronics.stackexchange.com": "Electrical Engineering",
    "physics.stackexchange.com": "Physics",
    "chemistry.stackexchange.com": "Chemistry",
    "space.stackexchange.com": "Space Exploration",
    "robotics.stackexchange.com": "Robotics",
    "3dprinting.stackexchange.com": "3D Printing",
    "dsp.stackexchange.com": "Signal Processing",
    "mechanics.stackexchange.com": "Motor Vehicle Maintenance",
    "networkengineering.stackexchange.com": "Network Engineering",
    "scicomp.stackexchange.com": "Computational Science",
    "ham.stackexchange.com": "Amateur Radio",
    "drones.stackexchange.com": "Drones",
    "diy.stackexchange.com": "Home Improvement",
}

LICENSE = "CC BY-SA 4.0"


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #
class _Textifier(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self):
        return "".join(self._parts)


def strip_html(raw: str) -> str:
    if not raw:
        return ""
    p = _Textifier()
    try:
        p.feed(raw)
        txt = p.text()
    except Exception:
        txt = re.sub(r"<[^>]+>", " ", raw)  # fallback: crude tag removal
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def parse_tags(raw: str):
    if not raw:
        return []
    return re.findall(r"<([^>]+)>", html.unescape(raw))


def _norm(text: str) -> str:
    """Normalised key for dedup: lowercase, alphanumerics + single spaces."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


# --------------------------------------------------------------------------- #
# Posts.xml parsing (two memory-lean iterparse passes)
# --------------------------------------------------------------------------- #
def _iter_rows(posts_path: str):
    for _, elem in iterparse(posts_path, events=("end",)):
        if elem.tag == "row":
            yield elem.attrib
            elem.clear()


def extract_community(posts_path: str, label: str, host: str, min_score: int):
    """Return a list of QA records for one community's Posts.xml."""
    # Pass A: question metadata (no body) + best answer id/score per question.
    questions = {}                      # qid -> meta
    best_ans = {}                       # qid -> (score, answer_id)
    for a in _iter_rows(posts_path):
        pt = a.get("PostTypeId")
        if pt == "1":
            qid = a.get("Id")
            questions[qid] = {
                "title": a.get("Title", ""),
                "tags": parse_tags(a.get("Tags", "")),
                "accepted": a.get("AcceptedAnswerId"),
                "owner": a.get("OwnerDisplayName") or a.get("OwnerUserId"),
            }
        elif pt == "2":
            parent = a.get("ParentId")
            if parent is None:
                continue
            try:
                score = int(a.get("Score", "0"))
            except ValueError:
                score = 0
            prev = best_ans.get(parent)
            if prev is None or score > prev[0]:
                best_ans[parent] = (score, a.get("Id"))

    # Decide the selected answer per question: accepted if present, else the
    # highest-scored positive answer.
    selected = {}                       # qid -> answer_id
    ans_score = {}                      # qid -> score (for ranking/output)
    for qid, meta in questions.items():
        acc = meta["accepted"]
        if acc:
            selected[qid] = acc
            ans_score[qid] = best_ans.get(qid, (0, acc))[0]
        else:
            b = best_ans.get(qid)
            if b and b[0] >= min_score:
                selected[qid] = b[1]
                ans_score[qid] = b[0]
    want_answer_ids = set(selected.values())
    want_question_ids = set(selected.keys())

    # Pass B: pull the bodies (and answer authors) only for selected posts.
    q_body = {}
    a_body = {}
    a_owner = {}
    for a in _iter_rows(posts_path):
        pid = a.get("Id")
        pt = a.get("PostTypeId")
        if pt == "1" and pid in want_question_ids:
            q_body[pid] = a.get("Body", "")
        elif pt == "2" and pid in want_answer_ids:
            a_body[pid] = a.get("Body", "")
            a_owner[pid] = a.get("OwnerDisplayName") or a.get("OwnerUserId")

    records = []
    for qid, aid in selected.items():
        meta = questions[qid]
        q_text = strip_html(meta["title"]) + "\n\n" + strip_html(q_body.get(qid, ""))
        answer = strip_html(a_body.get(aid, ""))
        if not answer or not q_text.strip():
            continue
        records.append({
            "chapter": f"Stack Exchange: {label}",
            "title": label,                       # expert label -> slug(title) = expert id
            "question": q_text.strip(),
            "answer": answer,
            "source_url": f"https://{host}/q/{qid}",
            "tags": meta["tags"],
            "score": ans_score.get(qid, 0),
            "author": a_owner.get(aid),
            "question_author": meta["owner"],
            "license": LICENSE,
        })
    return records


# --------------------------------------------------------------------------- #
# dedup + cap + assemble
# --------------------------------------------------------------------------- #
def dedup(records):
    """Drop near-duplicate questions, keeping the highest-scored answer."""
    best = {}
    for r in records:
        k = _norm(r["question"])
        if len(k) < 12:
            continue  # too short to be a real question
        cur = best.get(k)
        if cur is None or r["score"] > cur["score"]:
            best[k] = r
    return list(best.values())


def cap_per_expert(records, cap, rng):
    """Keep the top-`cap` records per expert by answer score (ties shuffled)."""
    by_title = defaultdict(list)
    for r in records:
        by_title[r["title"]].append(r)
    out = []
    for title, rs in by_title.items():
        rng.shuffle(rs)  # break score ties without positional bias
        rs.sort(key=lambda r: r["score"], reverse=True)
        out.extend(rs[:cap])
    return out


def truncate(records, max_q, max_a):
    for r in records:
        if max_q and len(r["question"]) > max_q:
            r["question"] = r["question"][:max_q].rstrip() + " ..."
        if max_a and len(r["answer"]) > max_a:
            r["answer"] = r["answer"][:max_a].rstrip() + " ..."
    return records


def build(dumps_dir, out="question_answer/qa.json", communities=None, cap=5000,
          min_score=1, max_q_chars=2000, max_a_chars=3000, seed=42):
    """Build qa.json from the extracted Stack Exchange dumps under ``dumps_dir``.

    ``communities`` optionally restricts to a subset of community folder names.
    Returns the number of QA pairs written.
    """
    rng = random.Random(seed)
    hosts = sorted(
        d for d in os.listdir(dumps_dir)
        if os.path.isfile(os.path.join(dumps_dir, d, "Posts.xml"))
    )
    if communities:
        hosts = [h for h in hosts if h in set(communities)]
    if not hosts:
        raise SystemExit(f"No <community>/Posts.xml found under {dumps_dir}")

    all_records = []
    print(f"{'community':<34} {'raw':>8} {'dedup':>8} {'capped':>8}")
    print("-" * 62)
    for host in hosts:
        label = COMMUNITY_LABELS.get(host, host.split(".")[0].replace("-", " ").title())
        posts = os.path.join(dumps_dir, host, "Posts.xml")
        recs = extract_community(posts, label, host, min_score)
        raw = len(recs)
        recs = dedup(recs)
        deduped = len(recs)
        recs = cap_per_expert(recs, cap, rng)
        capped = len(recs)
        recs = truncate(recs, max_q_chars, max_a_chars)
        all_records.extend(recs)
        print(f"{label:<34} {raw:>8} {deduped:>8} {capped:>8}")

    # Final cross-community dedup (identical questions across sites are rare but
    # would leak across the train/test split), then a global shuffle.
    all_records = dedup(all_records)
    rng.shuffle(all_records)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print("-" * 62)
    n_experts = len({r["title"] for r in all_records})
    print(f"TOTAL: {len(all_records)} QA pairs across {n_experts} experts -> {out}")
    print("Next: python main.py --split_qa   (writes qa_train.json, qa_test.json, split_by_title/)")
    print(f"License: {LICENSE} — keep source_url/author for attribution; "
          "release any published derivative under CC BY-SA (ShareAlike).")
    return len(all_records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps_dir", required=True,
                    help="dir with one <community>/Posts.xml per community")
    ap.add_argument("--out", default="question_answer/qa.json")
    ap.add_argument("--communities", nargs="*", default=None,
                    help="subset of community folder names to include (default: all found)")
    ap.add_argument("--cap", type=int, default=5000, help="max QA pairs per expert")
    ap.add_argument("--min_score", type=int, default=1,
                    help="min score for a non-accepted answer to qualify")
    ap.add_argument("--max_q_chars", type=int, default=2000)
    ap.add_argument("--max_a_chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(dumps_dir=args.dumps_dir, out=args.out, communities=args.communities,
          cap=args.cap, min_score=args.min_score, max_q_chars=args.max_q_chars,
          max_a_chars=args.max_a_chars, seed=args.seed)


if __name__ == "__main__":
    main()
