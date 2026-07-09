from download_llama.download import (
    download_llama_3_2_1b, download_llama_3_1_8b, download_hf_causal_lm,
    download_hf_snapshot,
)
from config import CONFIG
from logging_config import logger
import os


def _present(path: str) -> bool:
    """True if a model directory already exists and is non-empty (skip re-download)."""
    return os.path.isdir(path) and any(os.scandir(path))


def download_models() -> None:
    """Download every model the pipeline needs, skipping any already present.

    Every role must be available downstream: the Llama-1B router base
    (--finetune_router), the Llama-8B critic + description generator, and the Qwen
    experts/reasoner. Both Qwen sizes are fetched so ``slg.expert_model`` /
    ``slg.reasoner_model`` can be switched between them without a re-download.
    Each download is skipped if its directory already holds files, so re-running
    is cheap and a fresh setup gets everything.
    """
    models_config = CONFIG['models']
    paths_config = CONFIG['paths']
    models_paths = paths_config['models']
    downloaded_models_dir = paths_config['downloaded_models']

    def _dir(key):
        return os.path.join(downloaded_models_dir, models_paths[key])

    # (config key, target dir, downloader) — the 8B is fetched shard-by-shard; the
    # 14B by snapshot, since loading it just to re-save it would cost ~56GB of RAM.
    jobs = [
        ('3_2_1b', _dir('3_2_1b'), download_llama_3_2_1b),    # router classifier base
        ('3_1_8b', _dir('3_1_8b'), download_llama_3_1_8b),    # critic + descriptions
        ('qwen_3b', _dir('qwen_3b'), download_hf_causal_lm),  # experts + reasoner (small)
        ('qwen_14b', _dir('qwen_14b'), download_hf_snapshot), # experts + reasoner (large)
    ]
    for key, target, downloader in jobs:
        if _present(target):
            logger.info("Model '%s' already present at %s; skipping download.", key, target)
            continue
        downloader(model_name=models_config[key], save_directory=target)
