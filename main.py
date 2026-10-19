from cli.parser import build_parser
from config import CONFIG
import os
import yaml
from pathlib import Path


def configure_bundle(dataset, seed):
    root = Path(__file__).resolve().parent
    cfg_path = root / "configs" / "datasets" / f"{dataset}.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"Unknown dataset {dataset!r}: missing {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    CONFIG.update(overlay)
    CONFIG["seed"] = int(seed)
    CONFIG["ai_expert_prompt"] = CONFIG.get("dataset", {}).get("ai_expert_prompt", CONFIG["ai_expert_prompt"])
    CONFIG["data"]["inflation_percentage"] = CONFIG.get("dataset", {}).get("inflation_percentage", 0)
    cell = root / "results" / dataset / f"seed_{seed}"
    split = root / "data" / dataset / "splits" / f"seed_{seed}"
    if not (split / "train.json").is_file() or not (split / "test.json").is_file() or not (split / "by_title").is_dir():
        raise SystemExit(f"Missing prepared split for {dataset} seed {seed}: {split}. Run python main.py --build_datasets on the login node.")
    if not (root / "data" / dataset / "source.json").is_file():
        raise SystemExit(f"Missing dataset source: {root / 'data' / dataset / 'source.json'}")
    CONFIG["paths"].update({
        "experiments": str(cell), "answers": str(cell / "answers"),
        "checkpoints": str(cell / "checkpoints"),
        "split_by_title": str(split / "by_title"),
        "slg_index": "slg_index", "downloaded_models": str(root / "downloaded_models"),
    })
    CONFIG["logging"]["log_dir"] = str(cell / "logs")
    CONFIG["files"].update({
        "qa": str(root / "data" / dataset / "source.json"),
        "qa_train": str(split / "train.json"), "qa_test": str(split / "test.json"),
    })
    CONFIG["experiment"] = ""


def run_bundle(args):
    from commands.train import run_training
    from commands.inference import run_rag, run_finetuned, run_slg
    configure_bundle(args.dataset, args.seed)
    if args.router_override:
        run_slg("", router_method="finetuned", router_override=args.router_override)
        return
    cell = Path(CONFIG["paths"]["experiments"])
    done = cell / ".done"
    done.mkdir(parents=True, exist_ok=True)
    stages = [
        ("train", lambda: run_training("")),
        ("infer_1b_8b_slg_rag", lambda: (run_finetuned(""), run_slg("", router_method="finetuned", router_override=args.router_override), run_rag(""))),
        ("router_counterfactuals", lambda: run_router_counterfactuals(run_slg)),
    ]
    for name, action in stages:
        marker = done / name
        if marker.exists() and not (name == "evaluate" and args.score_metrics):
            continue
        action()
        marker.write_text("done\n", encoding="utf-8")


def run_router_counterfactuals(run_slg):
    """Run oracle and wrong-expert inference for every bundle cell."""
    for mode in ("oracle", "wrong"):
        run_slg("", router_method="finetuned", router_override=mode)


def run_evaluate_all(args):
    from commands.evaluation import run_evaluation

    datasets = (args.dataset,) if args.dataset else (
        "l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20"
    )
    seeds = (args.seed,) if args.seed is not None else (42, 53, 64)
    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "dummy":
        raise SystemExit("OPENAI_API_KEY must be set to a valid key before login-node evaluation.")

    for dataset in datasets:
        for seed in seeds:
            print(f"Evaluating {dataset}, seed {seed}", flush=True)
            configure_bundle(dataset, seed)
            run_evaluation(
                "", include_training_metrics=True,
                eval_workers=args.eval_workers,
                metrics_to_compute=args.score_metrics,
            )


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.build_datasets:
        from commands.build_datasets import build_datasets
        build_datasets(force=args.force)
        raise SystemExit(0)

    if args.bundle:
        run_bundle(args)
        raise SystemExit(0)

    if args.evaluate_all:
        run_evaluate_all(args)
        raise SystemExit(0)

    if args.activation_patching:
        from evaluate.activation_patching import run
        print(run())
        raise SystemExit(0)

    if args.post_analysis:
        from evaluate.paired_bootstrap import run as run_stats
        from evaluate.paper_assets import build as build_paper_assets
        from evaluate.deployment_table import build as build_deployment_table
        datasets = ("l_avia", "s_avia", "s_avia_25", "s_avia_50", "s_avia_75", "s_avia_100", "cqad20")
        for dataset in datasets:
            print(f"Paired significance analysis: {dataset} -> {run_stats(dataset)}")
        print(f"Paper quality table -> {build_paper_assets()}")
        print(f"Deployment table -> {build_deployment_table()}")
        from evaluate.activation_patching import run as run_patching
        print(f"Activation patching -> {run_patching()}")
        raise SystemExit(0)

    if args.stats:
        from evaluate.paired_bootstrap import run
        if not args.dataset:
            raise SystemExit("--stats requires --dataset DATASET")
        print(run(args.dataset))
        raise SystemExit(0)

    if args.deployment_table:
        from evaluate.deployment_table import build
        print(build([args.dataset] if args.dataset else None) if args.dataset else build())
        raise SystemExit(0)

    if args.paper_assets:
        from evaluate.paper_assets import build
        print(build([args.dataset] if args.dataset else None))
        raise SystemExit(0)

    if args.slg_all or args.scalability:
        requested = ", ".join(name for name, enabled in (("--slg_all", args.slg_all), ("--scalability", args.scalability)) if enabled)
        raise SystemExit(
            f"{requested} was requested, but the active full-rework checkout does not include "
            "the matching SLG ablation/scalability runner. No experiment was started."
        )

    if args.dataset:
        if args.seed is None:
            raise SystemExit("--dataset requires --seed for a bundle-scoped operation")
        configure_bundle(args.dataset, args.seed)

    # Experiments (config > fallback)
    experiment = CONFIG['experiment']

    # Download models
    from commands.models import download_models
    download_models() if args.download_models else None

    # SLG embeddings + index only (no finetuning)
    from commands.slg_embeddings import run_slg_embeddings
    run_slg_embeddings(experiment) if args.slg_embeddings else None

    # Training
    from commands.train import run_training
    run_training(experiment) if args.finetune else None

    # Inference
    from commands.inference import run_rag, run_finetuned, run_slg, run_interactive_slg
    run_rag(experiment) if args.infer_rag else None
    run_finetuned(experiment) if args.infer_finetuned else None
    run_slg(experiment, router_method=args.router, router_override=args.router_override) if args.infer_slg else None
    run_interactive_slg(experiment, show_routing=args.show_routing, router_method=args.router) if args.interactive_slg else None

    # Evaluation
    from commands.evaluation import run_evaluation
    run_evaluation(
        experiment,
        include_training_metrics=args.training_metrics,
        eval_workers=args.eval_workers,
        metrics_to_compute=args.score_metrics,
    ) if args.evaluate else None
    if args.plot_metrics:
        from commands.plot_metrics import plot_experiments_metrics
        plot_experiments_metrics()
