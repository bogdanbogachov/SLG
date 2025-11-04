import os
import json
from utils.path_utils import ensure_dir
from config import CONFIG


def run_evaluation(experiment: str, include_training_metrics: bool = False):
    from evaluate.evaluate import load_data, evaluate, pull_training_metrics

    metrics_list = []
    files_config = CONFIG['files']
    paths_config = CONFIG['paths']
    ground_truth_file = files_config['qa_test']
    experiments_dir = paths_config['experiments']
    answers_dir = os.path.join(paths_config['answers'], experiment)

    for predictions_file in os.listdir(answers_dir):
        predictions_path = os.path.join(answers_dir, predictions_file)
        predictions, ground_truth = load_data(predictions_path, ground_truth_file)
        results = evaluate(predictions, ground_truth)
        new_dict = {os.path.splitext(predictions_file)[0]: results}
        metrics_list.append(new_dict)

        experiment_dir = os.path.join(experiments_dir, experiment)
        ensure_dir(experiment_dir)
        metrics_path = os.path.join(experiment_dir, files_config['metrics'])
        with open(metrics_path, 'w') as f:
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
