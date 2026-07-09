"""Build the expert description registry (``slg_descriptions/descriptions.json``).

Descriptions serve three consumers: the routable-expert registry (an adapter with
no description is not routable), the Qwen-3B router tiebreaker prompt, and — until
recently — the critic prompt. They are *not* used by the classifier router, which
is the primary routing path.

Two sources, selected by ``slg.description_source``:

* ``metadata`` (default) — the expert's own community name, taken from the corpus
  (``title``), optionally enriched with the community's published tagline. On the
  Stack Exchange corpus the expert id *is* the ground-truth domain label, so a
  model is not needed to rediscover it. Deterministic, GPU-free, reproducible.
* ``llm`` — summarise a sample of the expert's answers with LLaMA 3.1-8B. Retained
  for corpora whose splits carry no meaningful topic label. Note the failure mode
  this path is prone to: with a very long answer block the 8B latches onto the
  trailing "must differ from these" list and paraphrases the *previous*
  descriptions instead of reading the answers. The prompt below guards against
  that (answers truncated, the distinctness list placed before them, and the
  instruction to summarise the answers placed last).
"""

import json
import os
import random
from collections import Counter
from typing import Dict, List

# Descriptions are summarised from a sample of an expert's answers (an expert can
# have thousands, which would overflow the model context). Keep this small.
_DESCRIPTION_SAMPLE_SIZE = 25
# Per-answer character cap for the LLM path. 25 full Stack Exchange answers run to
# ~40k characters; that buries the instruction and the model stops reading them.
_ANSWER_CHAR_CAP = 300
# Optional per-community taglines, e.g. {"robotics": "for professional robotic
# engineers, hobbyists, researchers and students"}. Real published metadata, not
# model output. Absent file => bare community name.
_TAGLINES_FILE = "question_answer/expert_taglines.json"

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

# The answers come last and the instruction refers back to them, so the model's
# most recent context is the material it must actually summarise. The
# already-used descriptions are stated up front purely as a constraint.
_DESCRIPTION_USER_TEMPLATE = (
    "{distinctness_block}"
    "Answers from the knowledge base:\n{answers}\n\n"
    "Summarise the topics covered by the answers above in at most {max_words} words. "
    "Be maximally specific to those answers.{distinctness_reminder}"
)

_DISTINCTNESS_BLOCK = (
    "These descriptions are already taken. Yours must describe different topics "
    "and must not paraphrase any of them:\n{previous}\n\n"
)

_DISTINCTNESS_REMINDER = " Do not reuse any of the already-taken descriptions."


# ------------------------------------------------------------------- metadata
def _load_taglines() -> Dict[str, str]:
    if not os.path.isfile(_TAGLINES_FILE):
        return {}
    with open(_TAGLINES_FILE, "r", encoding="utf-8") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def _community_name(records: List[dict], expert_id: str) -> str:
    """The expert's own domain label, from the corpus rather than from a model."""
    titles = Counter(
        str(r.get("title", "")).strip() for r in records if str(r.get("title", "")).strip()
    )
    if titles:
        return titles.most_common(1)[0][0]
    # No title in the corpus: prettify the split's filename.
    return expert_id.replace("_", " ").title()


def _metadata_description(records: List[dict], expert_id: str, taglines: Dict[str, str]) -> str:
    name = _community_name(records, expert_id)
    tagline = taglines.get(expert_id, "").strip()
    return f"{name} — {tagline}" if tagline else name


# ------------------------------------------------------------------------ llm
def _load_deduplicated_answers(records: List[dict]) -> List[str]:
    seen: set = set()
    answers: List[str] = []
    for entry in records:
        answer = str(entry.get("answer", "")).strip()
        if answer and answer not in seen:
            seen.add(answer)
            answers.append(answer)
    return answers


def _truncate(answer: str) -> str:
    """Keep the topical head of an answer; the tail rarely adds new topic signal."""
    collapsed = " ".join(answer.split())
    if len(collapsed) <= _ANSWER_CHAR_CAP:
        return collapsed
    return collapsed[:_ANSWER_CHAR_CAP].rsplit(" ", 1)[0] + " ..."


def _build_prompt(
    answers: List[str],
    previous_descriptions: Dict[str, str],
    max_words: int,
) -> List[Dict[str, str]]:
    answers_text = "\n".join(f"- {_truncate(a)}" for a in answers)

    if previous_descriptions:
        prev_block = _DISTINCTNESS_BLOCK.format(
            previous="\n".join(f"- {d}" for d in previous_descriptions.values())
        )
        reminder = _DISTINCTNESS_REMINDER
    else:
        prev_block, reminder = "", ""

    system = create_system_message(
        _DESCRIPTION_SYSTEM_PROMPT.format(max_words=max_words)
    )
    user = create_user_message(
        _DESCRIPTION_USER_TEMPLATE.format(
            answers=answers_text,
            distinctness_block=prev_block,
            distinctness_reminder=reminder,
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


def _llm_descriptions(split_dir: str, files: List[str], max_words: int) -> Dict[str, str]:
    paths_config = CONFIG["paths"]
    base_model_path = os.path.join(
        paths_config["downloaded_models"], paths_config["models"]["3_1_8b"]
    )
    model, tokenizer = load_base_model_and_tokenizer(base_model_path)

    descriptions: Dict[str, str] = {}
    try:
        for file in files:
            expert_id = os.path.splitext(file)[0]
            with open(os.path.join(split_dir, file), "r", encoding="utf-8") as f:
                records = json.load(f)
            answers = _load_deduplicated_answers(records)
            if not answers:
                logger.warning("No answers found for expert '%s'; skipping.", expert_id)
                continue

            # Sample a representative subset so the prompt fits the context window.
            if len(answers) > _DESCRIPTION_SAMPLE_SIZE:
                rng = random.Random(f"{CONFIG['seed']}:{expert_id}")
                answers = rng.sample(answers, _DESCRIPTION_SAMPLE_SIZE)

            messages = _build_prompt(answers, descriptions, max_words)
            descriptions[expert_id] = _generate(model, tokenizer, messages)
            logger.info("Expert '%s' → '%s'", expert_id, descriptions[expert_id])
    finally:
        cleanup_model_memory(model, tokenizer)
    return descriptions


# ----------------------------------------------------------------- entrypoint
def run_slg_descriptions(experiment: str) -> None:
    descriptions_path = get_slg_descriptions_path(experiment)
    if os.path.isfile(descriptions_path):
        logger.info(
            "Descriptions file already exists at %s; skipping. Delete to rebuild.",
            descriptions_path,
        )
        return

    ensure_dir(os.path.dirname(descriptions_path))

    split_dir = CONFIG["paths"]["split_by_title"]
    max_words = CONFIG["slg"].get("max_description_words", 10)
    source = str(CONFIG["slg"].get("description_source", "metadata")).lower()

    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
    if not files:
        raise ValueError(f"No split_by_title files found in {split_dir}.")

    if source == "llm":
        descriptions = _llm_descriptions(split_dir, files, max_words)
    elif source == "metadata":
        taglines = _load_taglines()
        descriptions = {}
        for file in files:
            expert_id = os.path.splitext(file)[0]
            with open(os.path.join(split_dir, file), "r", encoding="utf-8") as f:
                records = json.load(f)
            descriptions[expert_id] = _metadata_description(records, expert_id, taglines)
            logger.info("Expert '%s' → '%s'", expert_id, descriptions[expert_id])
    else:
        raise ValueError(
            f"Unknown slg.description_source '{source}'; expected 'metadata' or 'llm'."
        )

    with open(descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2)
    logger.info("Saved %d expert descriptions to %s", len(descriptions), descriptions_path)
