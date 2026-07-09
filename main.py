from cli.parser import build_parser
from commands.data_processing import (
    create_qa, combine_all_qa, inflate_overshadowing, split_qa, data_overlap_check,
    download_qa, build_qa,
)
from commands.train import run_training, run_finetune_router
from commands.slg_descriptions import run_slg_descriptions
from commands.inference import (
    run_baseline, run_rag, run_finetuned, run_slg, run_slg_chat,
    run_slg_ablations, run_slg_scalability, run_slg_metrics, run_slg_all,
    run_paper_assets,
)
from commands.evaluation import run_evaluation
from commands.plot_metrics import plot_experiments_metrics
from commands.models import download_models
from config import CONFIG


if __name__ == '__main__':
    import os
    parser = build_parser()
    args = parser.parse_args()

    # Experiments (config > fallback)
    experiment = CONFIG['experiment']

    # Quick-check subsets: repoint qa_test for this process and spawned GPU
    # workers, then isolate outputs under a matching sibling folder. If
    # --train_expert is used with inference, test only that expert; --limit then
    # samples inside the expert-only test file.
    output_suffix = ""
    inference_requested = any([
        args.infer_baseline, args.infer_rag, args.infer_finetuned, args.infer_slg,
        args.slg_ablations, args.slg_metrics, args.slg_all, args.evaluate,
    ])
    test_expert = args.train_expert.strip()
    if inference_requested and test_expert:
        from utils.subset import build_expert_test_subset, slug_title
        test_expert = slug_title(test_expert)
        if test_expert.endswith(".json"):
            test_expert = os.path.splitext(test_expert)[0]
        subset_path = build_expert_test_subset(CONFIG['files']['qa_test'], test_expert)
        if subset_path != CONFIG['files']['qa_test']:
            CONFIG['files']['qa_test'] = subset_path
            os.environ['SLG_QA_TEST_OVERRIDE'] = subset_path
        output_suffix += f"__{test_expert}"
        print(f"[--train_expert] test subset for '{test_expert}' -> {subset_path}; "
              f"outputs -> answers/{experiment}/{experiment}{output_suffix}/")

    if args.limit and args.limit > 0:
        from utils.subset import build_test_subset
        subset_path = build_test_subset(
            CONFIG['files']['qa_test'], args.limit, int(CONFIG['seed']))
        if subset_path != CONFIG['files']['qa_test']:
            CONFIG['files']['qa_test'] = subset_path          # this process
            os.environ['SLG_QA_TEST_OVERRIDE'] = subset_path  # spawned GPU workers
            output_suffix += f"__limit{args.limit}"
            print(f"[--limit] {args.limit}-question subset -> {subset_path}; "
                  f"outputs -> answers/{experiment}/{experiment}{output_suffix}/")

    # Download models
    download_models() if args.download_models else None

    # Real-world QA acquisition (Stack Exchange dumps -> qa.json)
    download_qa(args.dumps_dir, args.se_communities) if args.download_qa else None
    build_qa(args.dumps_dir, args.se_communities, args.qa_cap) if args.build_qa else None

    # Data processing
    create_qa() if args.create_qa else None
    combine_all_qa() if args.combine_all_qa else None
    inflate_overshadowing() if args.inflate_overshadowing else None
    split_qa(subset=args.qa_subset) if args.split_qa else None

    # Analysis
    data_overlap_check() if args.data_overlap_check else None

    # Generate SLG expert descriptions (run after split_qa, before finetune or infer_slg)
    run_slg_descriptions(experiment) if args.slg_descriptions else None

    # Training
    run_training(experiment, train_limit=args.train_limit, train_expert=args.train_expert) if args.finetune else None
    run_finetune_router(experiment) if args.finetune_router else None

    # Inference
    run_baseline(experiment) if args.infer_baseline else None
    run_rag(experiment) if args.infer_rag else None
    run_finetuned(experiment, output_suffix=output_suffix) if args.infer_finetuned else None
    run_slg(experiment, ablation=args.slg_ablation or "full",
            output_suffix=output_suffix) if args.infer_slg else None
    run_slg_chat(experiment) if args.chat_slg else None

    # SLG ablation experiments / metrics tooling
    run_slg_ablations(experiment) if args.slg_ablations else None
    run_slg_scalability(experiment) if args.slg_scalability else None
    run_slg_metrics(experiment) if args.slg_metrics else None
    run_slg_all(experiment) if args.slg_all else None

    # Evaluation
    run_evaluation(experiment, include_training_metrics=args.training_metrics) if args.evaluate else None

    # Aggregate every result into paper-ready LaTeX tables + figures
    run_paper_assets(experiment) if args.paper_assets else None
    plot_experiments_metrics() if args.plot_metrics else None
