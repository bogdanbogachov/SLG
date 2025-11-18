from cli.parser import build_parser
from commands.data_processing import create_qa, combine_all_qa, inflate_overshadowing, split_qa, data_overlap_check
from commands.train import run_training
from commands.inference import run_baseline, run_rag, run_finetuned, run_slg
from commands.evaluation import run_evaluation
from commands.models import download_models
from config import CONFIG


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # Experiments (config > fallback)
    experiment = CONFIG['experiment']

    # Download models
    download_models() if args.download_models else None

    # Data processing
    create_qa() if args.create_qa else None
    combine_all_qa() if args.combine_all_qa else None
    inflate_overshadowing() if args.inflate_overshadowing else None
    split_qa() if args.split_qa else None

    # Analysis
    data_overlap_check() if args.data_overlap_check else None

    # Training
    run_training(experiment) if args.finetune else None

    # Inference
    run_baseline(experiment) if args.infer_baseline else None
    run_rag(experiment) if args.infer_rag else None
    run_finetuned(experiment) if args.infer_finetuned else None
    run_slg(experiment) if args.infer_slg else None

    # Evaluation
    run_evaluation(experiment, include_training_metrics=args.training_metrics) if args.evaluate else None
