from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
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
