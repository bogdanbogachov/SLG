from config import CONFIG
import os
from huggingface_hub import snapshot_download


def download_models() -> None:
    """Download required models to local directories using config values."""
    models_config = CONFIG['models']
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']
    
    downloaded_models_dir = paths_config['downloaded_models']
    model_dir_3_2_1b = os.path.join(downloaded_models_dir, models_paths['3_2_1b'])
    model_dir_3_1_8b = os.path.join(downloaded_models_dir, models_paths['3_1_8b'])
    
    for repo_id, destination in (
        (models_config['3_2_1b'], model_dir_3_2_1b),
        (models_config['3_1_8b'], model_dir_3_1_8b),
        (models_config['jina_embeddings'], os.path.join(downloaded_models_dir, models_paths['jina_embeddings'])),
    ):
        os.makedirs(destination, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=destination)
