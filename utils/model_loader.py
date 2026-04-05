"""Utility functions for loading and managing models."""
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from exceptions import ModelNotFoundError
from logging_config import logger
from config import CONFIG


def _require_cuda_device() -> torch.device:
    """Single CUDA device for all model weights; no CPU placement."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but no GPU was found. "
            "This project expects models to run on GPU only."
        )
    return torch.device("cuda")


def load_base_model_and_tokenizer(
    model_path: str,
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model and tokenizer from a local path.

    The full model is loaded without ``device_map`` fragmentation, then moved
    entirely to ``cuda`` so training/eval do not mix CPU and GPU tensors.

    Args:
        model_path: Path to the model directory
        torch_dtype: Torch data type for the model
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelNotFoundError: If model path does not exist
        RuntimeError: If CUDA is not available
    """
    import os

    if not os.path.isdir(model_path):
        raise ModelNotFoundError(f"Model directory not found: {model_path}")

    device = _require_cuda_device()
    logger.info(f"Loading base model from: {model_path} (target device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)

    logger.info(f"Model loaded on device: {next(model.parameters()).device}")
    logger.debug(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    logger.debug(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    return model, tokenizer


def load_model_with_adapter(
    base_model_path: str,
    adapter_path: str,
    torch_dtype: torch.dtype = torch.float16,
    resize_token_embeddings: bool = False,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model with a LoRA adapter applied.

    The full model is placed entirely on ``cuda`` (no ``device_map='auto'``).

    Args:
        base_model_path: Path to the base model directory
        adapter_path: Path to the adapter directory
        torch_dtype: Torch data type for the model
        resize_token_embeddings: Whether to resize token embeddings
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model_with_adapter, tokenizer)

    Raises:
        ModelNotFoundError: If model or adapter path does not exist
        RuntimeError: If CUDA is not available
    """
    import os

    if not os.path.isdir(base_model_path):
        raise ModelNotFoundError(f"Base model directory not found: {base_model_path}")
    if not os.path.isdir(adapter_path):
        raise ModelNotFoundError(f"Adapter directory not found: {adapter_path}")

    device = _require_cuda_device()
    logger.info(f"Loading model with adapter from: {adapter_path} (device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)

    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    finetuned_model = PeftModel.from_pretrained(model, adapter_path)
    finetuned_model = finetuned_model.to(device)

    logger.info(
        f"Model with adapter loaded on device: {next(finetuned_model.parameters()).device}"
    )

    return finetuned_model, tokenizer


def cleanup_model_memory(model, tokenizer=None):
    """
    Clean up GPU memory after model usage.

    Args:
        model: Model to delete
        tokenizer: Tokenizer to delete (optional)
    """
    del model
    if tokenizer is not None:
        del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Helps defragment GPU memory
        logger.debug("GPU memory cleaned up")
