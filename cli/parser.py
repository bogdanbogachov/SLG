import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--split_qa", type=bool, default=False)
    parser.add_argument("--inflate_overshadowing", type=bool, default=False)
    parser.add_argument("--combine_all_qa", type=bool, default=False)
    parser.add_argument("--data_overlap_check", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--training_metrics", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--infer_baseline", type=bool, default=False)
    parser.add_argument("--infer_rag", type=bool, default=False)
    parser.add_argument("--infer_finetuned", type=bool, default=False)
    parser.add_argument("--download_models", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    return parser
