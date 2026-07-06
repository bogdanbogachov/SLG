import os

from config import CONFIG
from logging_config import logger
from utils.parallel import run_parallel


def _finetune_worker(task: dict) -> None:
    """Run one LoRA fine-tune. Top-level (picklable) so it can be dispatched to
    the multi-GPU pool; imports torch lazily after the worker has pinned its
    device."""
    from finetune.finetune import finetune

    finetune(**task)


def run_training(experiment: str, train_limit: int = 0) -> None:
    """Fine-tune every requested model.

    Each expert adapter and each baseline is an independent job, so the whole
    set is dispatched across all visible GPUs (one job per GPU). With a single
    GPU this runs sequentially, exactly as before — the per-adapter result is
    identical either way since an adapter depends only on its data and the seed,
    not on execution order.

    Args:
        experiment: Experiment identifier
    """
    paths_config = CONFIG['paths']
    files_config = CONFIG['files']
    models_paths = paths_config['models']
    adapters_config = CONFIG['adapters']

    training_config = CONFIG.get('training_components', {})
    train_slg_system = training_config.get('train_slg_system', False)
    train_3_2_1b = training_config.get('train_3_2_1b', False)
    train_3_1_8b = training_config.get('train_3_1_8b', False)

    experiments_dir = paths_config['experiments']
    split_by_title_dir = paths_config['split_by_title']
    downloaded_models_dir = paths_config['downloaded_models']

    os.makedirs(experiments_dir, exist_ok=True)

    tasks = []

    # One LoRA expert per title split (routing uses the prompt-based router +
    # descriptions.json).
    if train_slg_system:
        split_files = sorted(
            f for f in os.listdir(split_by_title_dir) if f.endswith(".json")
        )
        if not split_files:
            logger.warning("No JSON files in split_by_title; no SLG experts were trained.")
        for file in split_files:
            tasks.append({
                "model_to_tune": os.path.join(downloaded_models_dir, models_paths["3_2_1b"]),
                "adapter_name": os.path.splitext(file)[0],
                "data": os.path.join(split_by_title_dir, file),
                "experiment_number": experiment,
                "slg": True,
                "train_limit": train_limit,
            })
    else:
        logger.info("Skipping SLG system training")

    # Baseline 3_2_1b
    if train_3_2_1b:
        tasks.append({
            "model_to_tune": os.path.join(downloaded_models_dir, models_paths['3_2_1b']),
            "adapter_name": adapters_config['finetuned_3_2_1b'],
            "data": files_config['qa_train'],
            "experiment_number": experiment,
            "slg": False,
            "train_limit": train_limit,
        })
    else:
        logger.info("Skipping baseline 3_2_1b training")

    # Baseline 3_1_8b
    if train_3_1_8b:
        tasks.append({
            "model_to_tune": os.path.join(downloaded_models_dir, models_paths['3_1_8b']),
            "adapter_name": adapters_config['finetuned_3_1_8b'],
            "data": files_config['qa_train'],
            "experiment_number": experiment,
            "slg": False,
            "train_limit": train_limit,
        })
    else:
        logger.info("Skipping baseline 3_1_8b training")

    if not tasks:
        logger.info("No training tasks selected; nothing to do.")
        return

    logger.info("Fine-tuning %d model(s)...", len(tasks))
    run_parallel(("commands.train", "_finetune_worker"), tasks, label="finetune")
