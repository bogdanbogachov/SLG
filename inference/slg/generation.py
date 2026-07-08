"""Shared greedy text generation helper for the routing pipeline.

All LLaMA-backed roles (router, critic, aggregator, compressor) and the expert
adapters generate through this single function so decoding behaviour stays
consistent across the pipeline.
"""

from typing import Dict, List

import torch

from config import CONFIG
from logging_config import logger
from utils.prompt_utils import apply_chat_template

# Chat-terminator tokens across the model families we load (Llama uses
# <|eot_id|>, Qwen uses <|im_end|>). We stop on any that the tokenizer knows.
_CHAT_STOP_TOKENS = ("<|eot_id|>", "<|im_end|>", "<|end|>")


def _eos_ids(tokenizer):
    """EOS token id(s) valid for *this* tokenizer, so generation stops correctly.

    Hardcoding Llama's ``<|eot_id|>`` made Qwen models run to ``max_new_tokens``
    (Qwen terminates on ``<|im_end|>``). We combine the tokenizer's own
    ``eos_token_id`` with any chat-terminator tokens present in its vocab.
    """
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for tok in _CHAT_STOP_TOKENS:
        tid = vocab.get(tok)
        if tid is not None and tid not in ids:
            ids.append(tid)
    return ids or None


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
    eos_id = _eos_ids(tokenizer)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=CONFIG["generation"]["temperature"],
            eos_token_id=eos_id,
            repetition_penalty=CONFIG["generation"]["repetition_penalty"],
            no_repeat_ngram_size=CONFIG["generation"]["no_repeat_ngram_size"],
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
        results.extend(_generate_chunk_safe(chunk, model, tokenizer, max_new_tokens))
    return results


def _generate_chunk_safe(
    chunk: List[List[Dict[str, str]]],
    model,
    tokenizer,
    max_new_tokens: int,
) -> List[str]:
    """Decode ``chunk`` with an automatic batch-halving fallback on CUDA OOM.

    Fixed batch sizes can occasionally overflow VRAM on an unusually long
    prompt/answer. Rather than crash a multi-day job, we free the cache and retry
    the chunk in two halves (recursively), degrading to smaller batches only for
    the offending span. A single item that still OOMs is genuinely too large and
    is re-raised.
    """
    if len(chunk) == 1:
        return [generate(chunk[0], model, tokenizer, max_new_tokens)]
    try:
        return _generate_chunk(chunk, model, tokenizer, max_new_tokens)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        mid = len(chunk) // 2
        logger.warning(
            "CUDA OOM on a batch of %d; freeing cache and retrying as %d + %d.",
            len(chunk), mid, len(chunk) - mid,
        )
        left = _generate_chunk_safe(chunk[:mid], model, tokenizer, max_new_tokens)
        right = _generate_chunk_safe(chunk[mid:], model, tokenizer, max_new_tokens)
        return left + right


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
    eos_id = _eos_ids(tokenizer)

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
                repetition_penalty=CONFIG["generation"]["repetition_penalty"],
                no_repeat_ngram_size=CONFIG["generation"]["no_repeat_ngram_size"],
                do_sample=False,
            )
        # Left padding makes every prompt the same length, so the freshly
        # generated tokens are exactly the tail past the prompt width.
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = prev_side

    return [d.strip() for d in decoded]
