import os
from finetune.finetune import finetune
from config import CONFIG
from logging_config import logger


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

    # Finetune SLG experts per title and orchestrator
    if train_slg_system:
        logger.info("Training SLG system (experts + orchestrator)...")
        
        # Train SLG experts per title
        for file in os.listdir(split_by_title_dir):
            if file.endswith('.json'):
                logger.info(f"Training SLG expert for: {file}")
                finetune(
                    model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
                    adapter_name=os.path.splitext(file)[0],
                    data=os.path.join(split_by_title_dir, file),
                    experiment_number=experiment,
                    slg=True
                )
        
        # Train orchestrator
        logger.info("Training orchestrator...")
        finetune(
            model_to_tune=os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            adapter_name=adapters_config['orchestrator_3_2_1b'],
            data=files_config['qa_train'],
            experiment_number=experiment,
            orchestrator=True
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
