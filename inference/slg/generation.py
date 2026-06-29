"""Shared greedy text generation helper for the routing pipeline.

All LLaMA-backed roles (router, critic, aggregator, compressor) and the expert
adapters generate through this single function so decoding behaviour stays
consistent across the pipeline.
"""

from typing import Dict, List

import torch

from config import CONFIG
from utils.prompt_utils import apply_chat_template


def generate(
    messages: List[Dict[str, str]],
    model,
    tokenizer,
    max_new_tokens: int,
) -> str:
    """Render ``messages`` with the chat template and greedily decode a reply."""
    formatted = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=False, truncation=True
    ).to("cuda")
    eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=CONFIG["generation"]["temperature"],
            eos_token_id=eos_id,
            do_sample=False,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in decoded:
        return decoded.split("assistant")[-1].strip()
    return decoded.strip()
