import os
import json
from utils.path_utils import ensure_dir, get_answers_root, get_experiments_root, get_experiment_path
from config import CONFIG


def _is_predictions(data) -> bool:
    """A predictions file is a non-empty list of dicts each carrying a 'question'."""
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(isinstance(item, dict) and 'question' in item for item in data)
    )


def _score_run_dir(answers_dir: str, experiment_dir: str, ground_truth_file: str, logger) -> str:
    """Score every predictions file in one run folder into experiment_dir/metrics.json.

    Resumable per file (per-question checkpoints) and idempotent (a stem already
    present in metrics.json is skipped unless a checkpoint forces a re-score).
    Returns the metrics.json path.
    """
    from evaluate.evaluate import load_data, evaluate

    ensure_dir(experiment_dir)
    eval_ckpt_dir = os.path.join(experiment_dir, 'evaluation_checkpoints')
    ensure_dir(eval_ckpt_dir)

    metrics_path = os.path.join(experiment_dir, CONFIG['files']['metrics'])
    by_stem = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding='utf-8') as f:
            for item in json.load(f):
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
            by_stem[stem] = evaluate(predictions, ground_truth, checkpoint_path=checkpoint_path)
        elif stem in by_stem:
            logger.info(
                'Skipping %s (already in %s; remove that entry to force re-evaluation).',
                stem, metrics_path,
            )
            continue
        else:
            by_stem[stem] = evaluate(predictions, ground_truth, checkpoint_path=checkpoint_path)

        metrics_list = [{k: by_stem[k]} for k in sorted(by_stem.keys())]
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_list, f, indent=4)

    return metrics_path


def run_evaluation(experiment: str, include_training_metrics: bool = False,
                   include_scalability: bool = False):
    """Score answer quality for every run under the experiment's umbrella.

    Sweeps ``answers/<exp>/`` — the full run + baselines (``<exp>/``) and each
    leave-one-out ablation (``<exp>__no_competence/`` ...) — writing one
    ``experiments/<label>/metrics.json`` per run so ``--paper_assets`` can pull
    quality for all of them (main table + per-ablation columns). Scalability runs
    (``<exp>__scale*``) are skipped by default: they repeat the same task and no
    table consumes their quality; pass ``include_scalability=True`` to force them.
    """
    from evaluate.evaluate import pull_training_metrics
    from logging_config import logger

    files_config = CONFIG['files']
    ground_truth_file = files_config['qa_test']
    answers_umbrella = get_answers_root(experiment)
    experiments_umbrella = get_experiments_root(experiment)

    if not os.path.isdir(answers_umbrella):
        logger.warning('No answers found at %s; nothing to evaluate.', answers_umbrella)
        return

    labels = sorted(
        d for d in os.listdir(answers_umbrella)
        if os.path.isdir(os.path.join(answers_umbrella, d))
    )
    for label in labels:
        if not include_scalability and '__scale' in label:
            logger.info('Skipping scalability run %s (quality eval not needed; '
                        'pass include_scalability=True to force).', label)
            continue
        answers_dir = os.path.join(answers_umbrella, label)
        if not any(f.endswith('.json') for f in os.listdir(answers_dir)):
            continue  # no predictions in this folder
        logger.info('=== Evaluating run: %s ===', label)
        # Eval outputs mirror the answers umbrella: experiments/<exp>/<label>/.
        _score_run_dir(answers_dir, os.path.join(experiments_umbrella, label),
                       ground_truth_file, logger)

    if include_training_metrics:
        # Training logs live in the experiment's training home; append them to the
        # full run's metrics.json (experiments/<exp>/<exp>/metrics.json).
        training_home = get_experiment_path(experiment, CONFIG['paths']['experiments'])
        metrics_path = os.path.join(experiments_umbrella, experiment, files_config['metrics'])
        if os.path.isfile(metrics_path):
            training_metrics = pull_training_metrics(training_home)
            with open(metrics_path, "r") as f:
                data = json.load(f)
            data.extend(training_metrics)
            with open(metrics_path, "w") as f:
                json.dump(data, f, indent=4)
