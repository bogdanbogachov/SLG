from cli.parser import build_parser
from commands.data_processing import create_qa, combine_all, inflate_overshadowing, split_qa, data_overlap_check
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
    if args.download_models:
        download_models()

    # Data processing
    if args.create_qa:
        create_qa()
    if args.combine_all_qa:
        combine_all()
    if args.inflate_overshadowing:
        inflate_overshadowing()
    if args.split_qa:
        split_qa()

    # Analysis
    if args.data_overlap_check:
        data_overlap_check()

    # Training
    if args.finetune:
        run_training(experiment)

    # Inference
    if args.infer_baseline:
        run_baseline(experiment)
    if args.infer_rag:
        run_rag(experiment)
    if args.infer_finetuned:
        run_finetuned(experiment)
    if args.infer_slg:
        run_slg(experiment)

    # Evaluation
    if args.evaluate:
        run_evaluation(experiment, include_training_metrics=args.training_metrics)
