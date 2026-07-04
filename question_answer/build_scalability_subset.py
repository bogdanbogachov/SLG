"""Build the real-data scalability subset for the scalability sweep (#5).

The scalability experiment holds a **fixed question set** constant while growing
the expert pool, so latency / routing-accuracy changes are attributable to pool
size alone (see `--slg_scalability`). Historically this used synthetic distractor
experts; here we do it **entirely on real data** instead:

* pick a small, fixed CORE of a few genuinely distinct expert domains,
* sample a capped, seeded number of their real `qa_test` questions,
* write them to `question_answer/qa_scalability.json`.

At sweep time the harness computes `core = {domains present in this file}` and
uses the **remaining real experts as distractors** (experts these questions never
need), padding the pool up to each size in `routing.scalability_sizes`. With 12
real experts and 4 core domains there are 8 distractors, so pool sizes 4→12 are
all reachable with no synthetic data.

Reproducible: seeded by `CONFIG['seed']`. Rerun to rebuild. Writes to a fixed
path (never `qa_test.json`) so it cannot clobber the test set.
"""

import json
import os
import random

from config import CONFIG

# Four genuinely distinct domains -> routing is well-posed (aerospace, chemistry,
# robotics, electrical). Change here to reshape the core; keep them present in
# qa_test.json. The other 8 real experts become distractors automatically.
CORE_TITLES = ["aviation", "chemistry", "robotics", "electrical_engineering"]
PER_TITLE = 25  # questions sampled per core domain (100 total) — enough for a
                # stable latency / routing-accuracy curve, small enough to be fast.

OUT_PATH = os.path.join("question_answer", "qa_scalability.json")


def _slug(title: str) -> str:
    """Mirror evaluate.slg_metrics.slug_title / the split-by-title id."""
    return (title or "").replace(" ", "_").replace("/", "_").replace("\n", "_").lower()


def build() -> None:
    src = CONFIG["files"]["qa_test"]
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_title = {}
    for item in data:
        by_title.setdefault(_slug(item.get("title")), []).append(item)

    missing = [t for t in CORE_TITLES if t not in by_title]
    if missing:
        raise ValueError(f"Core titles absent from {src}: {missing}. Available: {sorted(by_title)}")

    rng = random.Random(int(CONFIG["seed"]))
    subset = []
    for title in CORE_TITLES:
        items = list(by_title[title])
        rng.shuffle(items)
        subset.extend(items[:PER_TITLE])
    rng.shuffle(subset)  # interleave domains so no run answers one domain first

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4)

    print(
        f"Wrote {len(subset)} questions from {len(CORE_TITLES)} core experts "
        f"({', '.join(CORE_TITLES)}) -> {OUT_PATH}"
    )


if __name__ == "__main__":
    build()
