from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, snapshot_download
from logging_config import logger
import os


def download_llama_3_2_1b(model_name, save_directory):
    logger.info(f"Downloading {model_name}...")


    save_directory = save_directory

    os.makedirs(save_directory, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=False,
        trust_remote_code=True
    )

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    logger.info(f"{model_name} downloaded.")

    return None


def download_hf_causal_lm(model_name, save_directory):
    """Generic HF causal-LM downloader (weights + tokenizer) to a local dir.

    Used for models pulled as a whole (e.g. Qwen-3B) rather than by explicit
    shard filenames. Same approach as :func:`download_llama_3_2_1b`."""
    logger.info(f"Downloading {model_name}...")
    os.makedirs(save_directory, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=False, trust_remote_code=True
    )
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    logger.info(f"{model_name} downloaded.")
    return None


def download_hf_snapshot(model_name, save_directory):
    """Copy a model repo's files straight to disk, without instantiating it.

    Preferred over :func:`download_hf_causal_lm` for anything large. That path
    calls ``AutoModelForCausalLM.from_pretrained``, which materialises the weights
    in RAM at the default dtype (fp32) and then re-serialises them, so a 14B model
    costs ~56GB of host memory and lands on disk at twice its published size. A
    snapshot copies the repo's native bf16 safetensors as-is: no GPU, no model
    load, no dtype conversion.

    Only the files the pipeline loads are fetched — the ``.bin`` mirrors, GGUF
    quantisations, and any ``original/`` directory are skipped.
    """
    logger.info(f"Downloading {model_name} (snapshot)...")
    os.makedirs(save_directory, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=save_directory,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        ignore_patterns=["original/*", "consolidated*", "*.gguf"],
    )
    logger.info(f"{model_name} downloaded.")
    return None


def download_llama_3_1_8b(model_name, save_directory):
    model_files = [
        "config.json",
        "generation_config.json",
        'model-00001-of-00004.safetensors',
        'model-00002-of-00004.safetensors',
        'model-00003-of-00004.safetensors',
        'model-00004-of-00004.safetensors',
        'model.safetensors.index.json',
        'special_tokens_map.json',
        'tokenizer.json',
        "tokenizer_config.json"
    ]

    for file in model_files:
        hf_hub_download(repo_id=model_name, filename=file, local_dir=save_directory)

    return None
