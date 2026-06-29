import os
import json
from utils.path_utils import ensure_dir
from config import CONFIG


def _is_predictions(data) -> bool:
    """A predictions file is a non-empty list of dicts each carrying a 'question'."""
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(isinstance(item, dict) and 'question' in item for item in data)
    )


def run_evaluation(experiment: str, include_training_metrics: bool = False):
    from evaluate.evaluate import load_data, evaluate, pull_training_metrics
    from logging_config import logger

    files_config = CONFIG['files']
    paths_config = CONFIG['paths']
    ground_truth_file = files_config['qa_test']
    experiments_dir = paths_config['experiments']
    answers_dir = os.path.join(paths_config['answers'], experiment)

    experiment_dir = os.path.join(experiments_dir, experiment)
    ensure_dir(experiment_dir)
    eval_ckpt_dir = os.path.join(experiment_dir, 'evaluation_checkpoints')
    ensure_dir(eval_ckpt_dir)

    metrics_path = os.path.join(experiment_dir, files_config['metrics'])
    by_stem = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding='utf-8') as f:
            existing = json.load(f)
        for item in existing:
            by_stem.update(item)

    for predictions_file in sorted(os.listdir(answers_dir)):
        if not predictions_file.endswith('.json'):
            continue
        stem = os.path.splitext(predictions_file)[0]
        checkpoint_path = os.path.join(eval_ckpt_dir, f'{stem}.json')
        predictions_path = os.path.join(answers_dir, predictions_file)

        # Only score prediction files: a list of dicts carrying a 'question'. Auxiliary
        # artifacts (e.g. routing logs) are not predictions and must be skipped, otherwise
        # evaluate() crashes trying to read them like predictions.
        predictions, ground_truth = load_data(predictions_path, ground_truth_file)
        if not _is_predictions(predictions):
            logger.info('Skipping %s (not a predictions file).', predictions_file)
            continue

        if os.path.isfile(checkpoint_path):
            results = evaluate(predictions, ground_truth, checkpoint_path=checkpoint_path)
            by_stem[stem] = results
        elif stem in by_stem:
            logger.info(
                'Skipping evaluation for %s (already in %s; remove that entry to force re-evaluation).',
                stem,
                metrics_path,
            )
            continue
        else:
            results = evaluate(predictions, ground_truth, checkpoint_path=checkpoint_path)
            by_stem[stem] = results

        metrics_list = [{k: by_stem[k]} for k in sorted(by_stem.keys())]
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_list, f, indent=4)

    if include_training_metrics:
        experiment_dir = os.path.join(experiments_dir, experiment)
        metrics_path = os.path.join(experiment_dir, files_config['metrics'])
        training_metrics = pull_training_metrics(experiment_dir)
        with open(metrics_path, "r") as f:
            data = json.load(f)
        data.extend(training_metrics)
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)
