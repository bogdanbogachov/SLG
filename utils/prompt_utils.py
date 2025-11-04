"""Utility functions for prompt formatting and chat templates."""
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = True,
    tokenize: bool = False
) -> str:
    """
    Apply chat template to messages using tokenizer.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: Tokenizer with chat template support
        add_generation_prompt: Whether to add generation prompt
        tokenize: Whether to tokenize (returns string if False)
        
    Returns:
        Formatted prompt string
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )


def create_user_message(content: str) -> Dict[str, str]:
    """
    Create a user message dictionary.
    
    Args:
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {"role": "user", "content": content}


def create_assistant_message(content: str) -> Dict[str, str]:
    """
    Create an assistant message dictionary.
    
    Args:
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {"role": "assistant", "content": content}


def create_system_message(content: str) -> Dict[str, str]:
    """
    Create a system message dictionary.
    
    Args:
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {"role": "system", "content": content}

