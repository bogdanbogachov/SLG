from download_llama.download import download_llama_3_2_1b, download_llama_3_1_8b
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
    
    download_llama_3_2_1b(
        model_name=models_config['3_2_1b'],
        save_directory=model_dir_3_2_1b
    )
    download_llama_3_1_8b(
        model_name=models_config['3_1_8b'],
        save_directory=model_dir_3_1_8b
    )
