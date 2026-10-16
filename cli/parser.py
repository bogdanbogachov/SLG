import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_models", action="store_true")
    parser.add_argument("--build_datasets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--router_override", choices=["oracle", "wrong"], default=None)
    parser.add_argument("--slg_all", action="store_true")
    parser.add_argument("--scalability", action="store_true")
    parser.add_argument("--score_metrics", nargs="+", default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--deployment_table", action="store_true")
    parser.add_argument("--paper_assets", action="store_true")
    parser.add_argument("--activation_patching", action="store_true")
    parser.add_argument("--post_analysis", action="store_true")

    parser.add_argument("--slg_embeddings", type=bool, default=False)

    parser.add_argument("--finetune", type=bool, default=False)

    parser.add_argument("--infer_finetuned", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--interactive_slg", type=bool, default=False)
    parser.add_argument("--show_routing", type=bool, default=False)
    parser.add_argument("--router", choices=["cosine", "finetuned"], default=None)

    parser.add_argument("--infer_rag", type=bool, default=False)

    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--eval_workers", type=int, default=None)

    parser.add_argument("--training_metrics", type=bool, default=False)
    parser.add_argument("--plot_metrics", type=bool, default=False)

    return parser
