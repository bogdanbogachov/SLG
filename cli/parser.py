import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_models", type=bool, default=False)

    # Real-world QA (Stack Exchange dumps)
    parser.add_argument("--download_qa", type=bool, default=False)
    parser.add_argument("--build_qa", type=bool, default=False)
    parser.add_argument("--dumps_dir", type=str, default="data/stackexchange")
    parser.add_argument("--se_communities", nargs="*", default=None)
    parser.add_argument("--qa_cap", type=int, default=5000)

    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--combine_all_qa", type=bool, default=False)
    parser.add_argument("--split_qa", type=bool, default=False)
    parser.add_argument("--qa_subset", type=int, default=0,
                        help="if >0, split only a stratified subset of N QA pairs (pipeline smoke test)")

    parser.add_argument("--data_overlap_check", type=bool, default=False)

    parser.add_argument("--inflate_overshadowing", type=bool, default=False)

    parser.add_argument("--slg_descriptions", type=bool, default=False)

    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--finetune_router", type=bool, default=False)

    parser.add_argument("--infer_finetuned", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--chat_slg", type=bool, default=False)

    # Quick-check subset: run inference on N seeded, expert-stratified test
    # questions (0 = full set). Outputs go to a sibling answers/<exp>/<exp>__limitN/
    # folder so a quick run never clobbers the full run's results.
    parser.add_argument("--limit", type=int, default=0,
                        help="Run inference on only N test questions (seeded, stratified by expert).")
    # Quick-check training subset: fine-tune each adapter on only N examples
    # (seeded) instead of the full training set. 0 = full set.
    parser.add_argument("--train_limit", type=int, default=0,
                        help="Fine-tune each adapter on only N training examples (seeded).")
    parser.add_argument("--train_expert", type=str, default="",
                        help="Fine-tune only this SLG expert id/file stem, e.g. aviation.")

    # SLG ablation / experiment tooling
    parser.add_argument("--slg_ablation", type=str, default="")
    parser.add_argument("--slg_ablations", type=bool, default=False)
    parser.add_argument("--slg_scalability", type=bool, default=False)
    parser.add_argument("--slg_metrics", type=bool, default=False)
    parser.add_argument("--slg_all", type=bool, default=False)
    parser.add_argument("--paper_assets", type=bool, default=False)

    parser.add_argument("--infer_baseline", type=bool, default=False)
    parser.add_argument("--infer_rag", type=bool, default=False)

    parser.add_argument("--evaluate", type=bool, default=False)

    parser.add_argument("--training_metrics", type=bool, default=False)
    parser.add_argument("--plot_metrics", type=bool, default=False)

    return parser
