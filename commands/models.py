from download_llama.download import (
    download_llama_3_2_1b, download_llama_3_1_8b, download_hf_causal_lm,
)
from config import CONFIG
import os


def download_models() -> None:
    """Download required models to local directories using config values."""
    models_config = CONFIG['models']
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']

    downloaded_models_dir = paths_config['downloaded_models']
    model_dir_3_2_1b = os.path.join(downloaded_models_dir, models_paths['3_2_1b'])
    model_dir_3_1_8b = os.path.join(downloaded_models_dir, models_paths['3_1_8b'])
    model_dir_qwen_3b = os.path.join(downloaded_models_dir, models_paths['qwen_3b'])

    download_llama_3_2_1b(
        model_name=models_config['3_2_1b'],
        save_directory=model_dir_3_2_1b
    )
    download_llama_3_1_8b(
        model_name=models_config['3_1_8b'],
        save_directory=model_dir_3_1_8b
    )
    # Qwen-3B: experts + reasoner (router tiebreak / aggregate / compress).
    download_hf_causal_lm(
        model_name=models_config['qwen_3b'],
        save_directory=model_dir_qwen_3b,
    )
