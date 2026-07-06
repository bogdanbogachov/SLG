"""Build a small, reproducible subset of a QA file for quick-check runs.

Used by ``--limit N``: draws N questions from the test set, stratified across
experts (``title``) so every expert is represented, seeded for reproducibility.
Records are returned in their original order so evaluation alignment is
preserved. The subset is written next to the source as ``<name>_limit<N>.json``.
"""
import json
import os
import random
from collections import defaultdict


def build_test_subset(src_path: str, n: int, seed: int, out_path: str = None) -> str:
    """Write and return the path to an N-question stratified subset of ``src_path``.

    Returns ``src_path`` unchanged when ``n`` is non-positive or already covers
    the whole file (nothing to subset).
    """
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if n <= 0 or n >= len(data):
        return src_path

    # Group by expert (title) and round-robin across experts for even coverage.
    groups = defaultdict(list)
    for idx, rec in enumerate(data):
        groups[rec.get("title", "_")].append(idx)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)

    titles = sorted(groups)
    chosen = set()
    i = 0
    while len(chosen) < n and any(groups[t] for t in titles):
        t = titles[i % len(titles)]
        if groups[t]:
            chosen.add(groups[t].pop())
        i += 1

    # Preserve original order for eval alignment.
    subset = [data[idx] for idx in sorted(chosen)]

    if out_path is None:
        base, ext = os.path.splitext(src_path)
        out_path = f"{base}_limit{n}{ext}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    return out_path
