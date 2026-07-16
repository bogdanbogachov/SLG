import os
from typing import List

from finetune.finetune import finetune
from config import CONFIG
from logging_config import logger
from commands.slg_embeddings import run_slg_embeddings
from utils.path_utils import slg_expert_id_from_filename


def run_training(experiment: str):
    """
    Run training for specified model components.

    Args:
        experiment: Experiment identifier
    """
    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']

    # Get training configuration from config
    training_config = CONFIG.get('training_components', {})

    train_slg_system = training_config.get('train_slg_system', False)
    train_3_2_1b = training_config.get('train_3_2_1b', False)
    train_3_1_8b = training_config.get('train_3_1_8b', False)

    experiments_dir = paths_config['experiments']
    split_by_title_dir = paths_config['split_by_title']
    downloaded_models_dir = paths_config['downloaded_models']

    os.makedirs(experiments_dir, exist_ok=True)

    # Finetune SLG experts per title (routing uses Jina embeddings + index.json, not a finetuned orchestrator)
    if train_slg_system:
        logger.info("Training SLG experts...")
        split_by_title_files = sorted(
            [f for f in os.listdir(split_by_title_dir) if f.endswith(".json")]
        )

        for file in split_by_title_files:
            logger.info(f"Training SLG expert for: {file}")
            adapter_name = slg_expert_id_from_filename(file)
            data_path = os.path.join(split_by_title_dir, file)

            finetune(
                model_to_tune=os.path.join(
                    downloaded_models_dir, models_paths["3_2_1b"]
                ),
                adapter_name=adapter_name,
                data=data_path,
                experiment_number=experiment,
                slg=True,
            )

        if split_by_title_files:
            logger.info("Building SLG similarity index (chunk embeddings + index.json)...")
            run_slg_embeddings(experiment)
        else:
            logger.warning(
                "No JSON files in split_by_title; skipping SLG embedding index build."
            )

    else:
        logger.info("Skipping SLG system training")

    # Baseline 3_2_1b
    if train_3_2_1b:
        logger.info("Training baseline model: 3_2_1b")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            adapter_name=adapters_config['finetuned_3_2_1b'],
            data=files_config['qa_train'],
            experiment_number=experiment
        )
    else:
        logger.info("Skipping baseline 3_2_1b training")

    # Baseline 3_1_8b
    if train_3_1_8b:
        logger.info("Training baseline model: 3_1_8b")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_1_8b']),
            adapter_name=adapters_config['finetuned_3_1_8b'],
            data=files_config['qa_train'],
            experiment_number=experiment
        )
    else:
        logger.info("Skipping baseline 3_1_8b training")
