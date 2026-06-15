"""Generate short, distinct expert descriptions using LLaMA 3.2-1B-Instruct."""

import json
import os
from typing import Dict, List

import torch

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_base_model_and_tokenizer
from utils.path_utils import ensure_dir, get_slg_descriptions_path
from utils.prompt_utils import apply_chat_template, create_system_message, create_user_message


_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a precise technical summariser. "
    "Your output must be a single phrase of at most {max_words} words "
    "that captures the specific engineering topics covered by a set of answers. "
    "Output only the phrase — no punctuation at the end, no extra text."
)

_DESCRIPTION_USER_TEMPLATE = (
    "Answers from the knowledge base:\n{answers}\n\n"
    "{distinctness_block}"
    "Generate a description of at most {max_words} words for these topics. "
    "Be maximally specific."
)

_DISTINCTNESS_BLOCK = (
    "Already generated descriptions — your output MUST differ from all of these:\n"
    "{previous}\n\n"
)


def _load_deduplicated_answers(data_path: str) -> List[str]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen: set = set()
    answers: List[str] = []
    for entry in data:
        answer = str(entry.get("answer", "")).strip()
        if answer and answer not in seen:
            seen.add(answer)
            answers.append(answer)
    return answers


def _build_prompt(
    answers: List[str],
    previous_descriptions: Dict[str, str],
    max_words: int,
) -> List[Dict[str, str]]:
    answers_text = "\n".join(f"- {a}" for a in answers)

    if previous_descriptions:
        prev_block = _DISTINCTNESS_BLOCK.format(
            previous="\n".join(f"- {d}" for d in previous_descriptions.values())
        )
    else:
        prev_block = ""

    system = create_system_message(
        _DESCRIPTION_SYSTEM_PROMPT.format(max_words=max_words)
    )
    user = create_user_message(
        _DESCRIPTION_USER_TEMPLATE.format(
            answers=answers_text,
            distinctness_block=prev_block,
            max_words=max_words,
        )
    )
    return [system, user]


def _generate(model, tokenizer, messages: List[Dict[str, str]]) -> str:
    formatted = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=False, truncation=True
    ).to("cuda")

    eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            temperature=CONFIG["generation"]["temperature"],
            eos_token_id=eos_id,
            do_sample=False,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in decoded:
        return decoded.split("assistant")[-1].strip()
    return decoded.strip()


def run_slg_descriptions() -> None:
    descriptions_path = get_slg_descriptions_path()
    if os.path.isfile(descriptions_path):
        logger.info(
            "Descriptions file already exists at %s; skipping. Delete to rebuild.",
            descriptions_path,
        )
        return

    ensure_dir(os.path.dirname(descriptions_path))

    paths_config = CONFIG["paths"]
    split_by_title_dir = paths_config["split_by_title"]
    max_words = CONFIG["slg"].get("max_description_words", 10)

    files = sorted(f for f in os.listdir(split_by_title_dir) if f.endswith(".json"))
    if not files:
        raise ValueError(f"No split_by_title files found in {split_by_title_dir}.")

    base_model_path = os.path.join(
        paths_config["downloaded_models"], paths_config["models"]["3_2_1b"]
    )
    model, tokenizer = load_base_model_and_tokenizer(base_model_path)

    descriptions: Dict[str, str] = {}
    try:
        for file in files:
            expert_id = os.path.splitext(file)[0]
            data_path = os.path.join(split_by_title_dir, file)
            answers = _load_deduplicated_answers(data_path)

            if not answers:
                logger.warning("No answers found for expert '%s'; skipping.", expert_id)
                continue

            messages = _build_prompt(answers, descriptions, max_words)
            description = _generate(model, tokenizer, messages)
            descriptions[expert_id] = description
            logger.info("Expert '%s' → '%s'", expert_id, description)
    finally:
        cleanup_model_memory(model, tokenizer)

    with open(descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2)
    logger.info("Saved %d expert descriptions to %s", len(descriptions), descriptions_path)
