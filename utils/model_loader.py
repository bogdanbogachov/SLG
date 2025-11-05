"""Utility functions for loading and managing models."""
import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from exceptions import ModelNotFoundError
from logging_config import logger
from config import CONFIG


def load_base_model_and_tokenizer(
    model_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model and tokenizer from a local path.
    
    Args:
        model_path: Path to the model directory
        torch_dtype: Torch data type for the model
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ModelNotFoundError: If model path doesn't exist
    """
    import os
    if not os.path.isdir(model_path):
        raise ModelNotFoundError(f"Model directory not found: {model_path}")
    
    logger.info(f"Loading base model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    
    # Ensure model is on GPU if available (for consistency with load_model_with_adapter)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    logger.info(f"Model loaded on device: {model.device}")
    logger.debug(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    logger.debug(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    
    return model, tokenizer


def load_model_with_adapter(
    base_model_path: str,
    adapter_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    resize_token_embeddings: bool = False,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model with a LoRA adapter applied.
    
    Args:
        base_model_path: Path to the base model directory
        adapter_path: Path to the adapter directory
        torch_dtype: Torch data type for the model
        device_map: Device mapping strategy
        resize_token_embeddings: Whether to resize token embeddings
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model_with_adapter, tokenizer)
        
    Raises:
        ModelNotFoundError: If model or adapter path doesn't exist
    """
    import os
    if not os.path.isdir(base_model_path):
        raise ModelNotFoundError(f"Base model directory not found: {base_model_path}")
    if not os.path.isdir(adapter_path):
        raise ModelNotFoundError(f"Adapter directory not found: {adapter_path}")
    
    logger.info(f"Loading model with adapter from: {adapter_path}")
    
    # Load tokenizer from adapter (usually has the correct config)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    
    # Resize embeddings if needed
    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    
    # Apply adapter
    finetuned_model = PeftModel.from_pretrained(model, adapter_path)
    
    # Ensure on GPU
    if torch.cuda.is_available():
        finetuned_model.to("cuda")
    
    logger.info(f"Model with adapter loaded on device: {finetuned_model.device}")
    
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

