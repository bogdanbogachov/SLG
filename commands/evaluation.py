import os
import json
from typing import Optional
from utils.path_utils import ensure_dir
from config import CONFIG


def run_evaluation(
    experiment: str,
    include_training_metrics: bool = False,
    eval_workers: Optional[int] = None,
    metrics_to_compute: Optional[list[str]] = None,
):
    from evaluate.evaluate import load_data, evaluate, pull_training_metrics
    from logging_config import logger

    files_config = CONFIG['files']
    paths_config = CONFIG['paths']
    ground_truth_file = files_config['qa_test']
    experiments_dir = paths_config['experiments']
    answers_dir = os.path.join(paths_config['answers'], experiment)

    experiment_dir = os.path.join(experiments_dir, experiment)
    ensure_dir(experiment_dir)
    eval_ckpt_dir = os.path.join(paths_config['checkpoints'], 'evaluation')
    scores_dir = os.path.join(experiment_dir, 'scores')
    ensure_dir(scores_dir)
    ensure_dir(eval_ckpt_dir)

    metrics_path = os.path.join(experiment_dir, files_config['metrics'])
    by_stem = {}
    saved_training_metrics = []
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding='utf-8') as f:
            existing = json.load(f)
        for item in existing:
            for key, value in item.items():
                if isinstance(value, dict) and any(metric in value for metric in ('BLEU', 'ROUGE', 'Exact Match')):
                    by_stem[key] = value
                else:
                    saved_training_metrics.append(item)
                    break

    for predictions_file in sorted(os.listdir(answers_dir)):
        if not predictions_file.endswith(('.json', '.jsonl')):
            continue
        stem = os.path.splitext(predictions_file)[0]
        checkpoint_path = os.path.join(eval_ckpt_dir, f'{stem}.json')

        predictions_path = os.path.join(answers_dir, predictions_file)
        predictions, ground_truth = load_data(predictions_path, ground_truth_file)
        _require_complete_qids(predictions, ground_truth, predictions_path)
        complete_checkpoint = False
        if os.path.isfile(checkpoint_path) and not metrics_to_compute:
            with open(checkpoint_path, encoding='utf-8') as f:
                state = json.load(f)
            complete_checkpoint = (
                len(state.get('completed_scores', {})) + len(state.get('skipped_indices', []))
                == len(ground_truth)
            )
            if complete_checkpoint:
                by_stem.setdefault(stem, state.get('partial_metrics', {}))
        if not complete_checkpoint:
            results = evaluate(
                predictions,
                ground_truth,
                checkpoint_path=checkpoint_path,
                eval_workers=eval_workers,
                metrics_to_compute=metrics_to_compute,
            )
            by_stem[stem] = results
        score_path = os.path.join(scores_dir, f'{stem}.jsonl')
        if os.path.isfile(checkpoint_path):
            _write_score_jsonl(checkpoint_path, predictions_path, ground_truth_file, score_path)

    metrics_list = [{k: by_stem[k]} for k in sorted(by_stem.keys())]
    if include_training_metrics:
        saved_training_metrics = pull_training_metrics(experiment_dir)
    metrics_list.extend(saved_training_metrics)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=4)


def _write_score_jsonl(checkpoint_path, predictions_path, ground_truth_path, output_path):
    import json
    with open(checkpoint_path, encoding="utf-8") as f:
        state = json.load(f)
    from evaluate.evaluate import load_data
    predictions, truths = load_data(predictions_path, ground_truth_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, raw_scores in sorted(((int(i), v) for i, v in state.get("completed_scores", {}).items())):
            if idx >= len(truths) or idx >= len(predictions):
                continue
            truth, pred = truths[idx], predictions[idx]
            score = {
                "qid": truth.get("qid"), "question": truth.get("question"),
                "gold_answer": truth.get("answer"), "prediction": pred.get("answer"),
                "BLEU": raw_scores.get("BLEU"),
                "ROUGE-1": raw_scores.get("ROUGE", {}).get("rouge1"),
                "ROUGE-2": raw_scores.get("ROUGE", {}).get("rouge2"),
                "ROUGE-L": raw_scores.get("ROUGE", {}).get("rougeL"),
                "EM": raw_scores.get("Exact Match"), "METEOR": raw_scores.get("METEOR"),
                "Entailment": raw_scores.get("Entailment"), "AI_Expert": raw_scores.get("AI Expert"),
                "latency_s": pred.get("latency_s"),
                "peak_gpu_memory_mb": pred.get("peak_gpu_memory_mb"),
            }
            f.write(json.dumps(score, ensure_ascii=False) + "\n")


def _require_complete_qids(predictions, ground_truth, predictions_path):
    expected = {row.get("qid") for row in ground_truth}
    actual = {row.get("qid") for row in predictions}
    if None not in expected and actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(f"Incomplete qid coverage in {predictions_path}: missing={len(missing)}, extra={len(extra)}")
