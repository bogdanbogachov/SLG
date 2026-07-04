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


def generate_batch(
    messages_list: List[List[Dict[str, str]]],
    model,
    tokenizer,
    max_new_tokens: int,
    batch_size: int = 1,
) -> List[str]:
    """Greedily decode a reply for each conversation in ``messages_list``.

    Several prompts are decoded together in one forward pass (left-padded so a
    decoder-only model sees each real prompt flush against the generation
    position), which fills the GPU far better than one call at a time. Results
    are returned in input order.

    Determinism note: batched decoding is greedy (``do_sample=False``) exactly
    like :func:`generate`, so each prompt yields the same *tokens* as a solo
    call in practice; only padding + float reduction order differ, which almost
    never flips the greedy argmax. A chunk of size one is routed through the
    unbatched :func:`generate` so single-question rounds stay bit-identical.
    """
    if not messages_list:
        return []
    if batch_size is None or batch_size <= 0:
        batch_size = len(messages_list)

    results: List[str] = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start:start + batch_size]
        if len(chunk) == 1:
            results.append(generate(chunk[0], model, tokenizer, max_new_tokens))
            continue
        results.extend(_generate_chunk(chunk, model, tokenizer, max_new_tokens))
    return results


def _generate_chunk(
    chunk: List[List[Dict[str, str]]],
    model,
    tokenizer,
    max_new_tokens: int,
) -> List[str]:
    """Decode one padded batch (>=2 prompts) and return the replies in order."""
    formatted = [
        apply_chat_template(m, tokenizer, add_generation_prompt=True) for m in chunk
    ]
    eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # Decoder-only batched generation requires left padding so every prompt ends
    # at the same position; a mask keeps the pad tokens out of attention.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=CONFIG["generation"]["temperature"],
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
        # Left padding makes every prompt the same length, so the freshly
        # generated tokens are exactly the tail past the prompt width.
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = prev_side

    return [d.strip() for d in decoded]
