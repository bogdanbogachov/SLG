import argparse
from concurrent.futures import ProcessPoolExecutor

from answer_retrieval.retrieval import DATASETS, SEEDS, evaluate_cell, summarize


def main():
    p = argparse.ArgumentParser(description="Stored-answer retrieval baseline vs SLG (strict procedure accuracy).")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [(d, int(s)) for d in a.datasets.split(",") for s in a.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        cells = list(ex.map(evaluate_cell, *zip(*jobs)))
    for c in cells:
        print(f"{c['dataset']} seed {c['seed']}: retrieval {c['retrieval_acc']:.3f}, SLG {c['slg_acc']:.3f}")
    print(summarize(cells))


if __name__ == "__main__":
    main()
