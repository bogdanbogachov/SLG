import os
from finetune.finetune import finetune
from config import CONFIG


def run_training(experiment: str):
    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    models_paths = paths_config['models']
    
    experiments_dir = paths_config['experiments']
    split_by_title_dir = paths_config['split_by_title']
    downloaded_models_dir = paths_config['downloaded_models']
    
    os.makedirs(experiments_dir, exist_ok=True)

    # Finetune SLG experts per title
    for file in os.listdir(split_by_title_dir):
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            adapter_name=os.path.splitext(file)[0],
            data=os.path.join(split_by_title_dir, file),
            experiment_number=experiment,
            slg=True
        )

    # Orchestrator
    finetune(
        model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
        adapter_name='orchestrator_3_2_1b',
        data=files_config['qa_train'],
        experiment_number=experiment,
        orchestrator=True
    )

    # Baselines
    finetune(
        model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
        adapter_name='3_2_1b',
        data=files_config['qa_train'],
        experiment_number=experiment
    )

    finetune(
        model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_1_8b']),
        adapter_name='3_1_8b',
        data=files_config['qa_train'],
        experiment_number=experiment
    )
