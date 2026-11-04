"""python -m hidden_router {probe,assemble} [--datasets a,b] [--seeds 42,53,64]"""
import argparse

from hidden_router.cells import DATASETS, SEEDS


def main():
    p = argparse.ArgumentParser(prog="python -m hidden_router")
    p.add_argument("step", choices=["probe", "assemble"],
                   help="probe = plan steps 1-4 (extract, fit/test, compare, estimate); assemble = steps 5-6")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--refit", action="store_true", help="probe: refit routers even if saved")
    p.add_argument("--reuse_threshold", type=float, default=0.95)
    p.add_argument("--allow_reuse_mismatch", action="store_true")
    args = p.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.step == "probe":
        from hidden_router.probe import run
        print(run(datasets, seeds, refit=args.refit))
    else:
        from hidden_router.assemble import run
        run(datasets, seeds, reuse_threshold=args.reuse_threshold, allow_reuse_mismatch=args.allow_reuse_mismatch)


if __name__ == "__main__":
    main()
