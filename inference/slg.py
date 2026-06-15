"""Small Language Router (SLR) for multi-expert question answering.

Pipeline:
1) Load expert descriptions from slg_descriptions/descriptions.json.
2) Route each question: the router SLM reasons over all descriptions and outputs
   one or more expert names, or NONE if the question is out of scope.
3) Invoke each chosen expert (fine-tuned LoRA adapter) to generate an answer.
4) If multiple experts were called, answers are concatenated under expert labels.
"""

import json
import os
from typing import Dict, List

import torch

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_base_model_and_tokenizer, load_model_with_adapter
from utils.path_utils import (
    ensure_dir,
    get_slg_descriptions_path,
    get_slg_path,
    validate_dir_exists,
    validate_file_exists,
)
from utils.prompt_utils import apply_chat_template, create_system_message, create_user_message


_ROUTER_SYSTEM_PROMPT = (
    "You are a reasoning-based routing assistant for a question-answering system. "
    "Given a question and a list of specialist experts with short descriptions, follow these steps:\n"
    "1. Briefly reason about which expert(s) are most relevant to the question topic.\n"
    "2. On the very last line, output 'Experts: ' followed by the expert name(s) as a comma-separated list "
    "using the exact names provided.\n"
    "   If the question is outside the scope of all experts, output 'NONE' on the last line.\n\n"
    "Example last line (one expert):   Experts: expert_a\n"
    "Example last line (two experts):  Experts: expert_a, expert_b\n"
    "Example last line (out of scope): NONE"
)

_ROUTER_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Available experts:\n{expert_list}\n\n"
    "Reason about which expert(s) can best answer this question, "
    "then on the final line output the expert name(s) or NONE."
)


class SmallLanguageRouter:
    def __init__(self, experts_location: str, experiment: str):
        self.experts_location = experts_location
        self.experiment = experiment

        paths_config = CONFIG["paths"]
        self.experiments_dir = paths_config["experiments"]
        self.slg_path = get_slg_path(self.experts_location, self.experiments_dir)
        validate_dir_exists(
            self.slg_path,
            error_message=(
                f"SLG expert adapters directory not found: {self.slg_path}. "
                "Train SLG experts before running inference."
            ),
        )

        descriptions_path = get_slg_descriptions_path()
        validate_file_exists(
            descriptions_path,
            error_message=(
                f"Expert descriptions not found: {descriptions_path}. "
                "Run --slg_descriptions before inference."
            ),
        )
        with open(descriptions_path, "r", encoding="utf-8") as f:
            self._descriptions: Dict[str, str] = json.load(f)

        self._expert_nodes: List[str] = self._discover_expert_nodes()
        self._out_of_scope_token: str = CONFIG["slg"].get("out_of_scope_token", "NONE")

        desc_keys = set(self._descriptions.keys())
        node_keys = set(self._expert_nodes)
        if desc_keys - node_keys:
            logger.warning(
                "Descriptions exist for experts with no adapter on disk: %s",
                sorted(desc_keys - node_keys),
            )
        if node_keys - desc_keys:
            logger.warning(
                "Adapters exist on disk with no description: %s — these experts will not be routable.",
                sorted(node_keys - desc_keys),
            )

        paths_cfg = CONFIG["paths"]
        self._base_model_path = os.path.join(
            paths_cfg["downloaded_models"], paths_cfg["models"]["3_2_1b"]
        )

    def _discover_expert_nodes(self) -> List[str]:
        return sorted(
            name
            for name in os.listdir(self.slg_path)
            if os.path.isdir(os.path.join(self.slg_path, name))
        )

    def _build_router_prompt(self, question: str) -> List[Dict[str, str]]:
        valid_experts = {
            eid: desc
            for eid, desc in self._descriptions.items()
            if eid in set(self._expert_nodes)
        }
        expert_list = "\n".join(f"- {eid}: {desc}" for eid, desc in valid_experts.items())
        return [
            create_system_message(_ROUTER_SYSTEM_PROMPT),
            create_user_message(_ROUTER_USER_TEMPLATE.format(
                question=question, expert_list=expert_list
            )),
        ]

    def _parse_router_output(self, raw: str) -> List[str]:
        """Return a list of valid expert IDs from the router's CoT output (empty = out-of-scope)."""
        if "assistant" in raw:
            raw = raw.split("assistant")[-1]

        lines = [ln.strip() for ln in raw.strip().splitlines()]
        last_line = next((ln for ln in reversed(lines) if ln), "")

        if last_line.upper() == self._out_of_scope_token.upper():
            return []

        # Strip optional "Experts:" prefix
        if last_line.lower().startswith("experts:"):
            last_line = last_line[len("experts:"):].strip()

        candidates = [c.strip().rstrip(".") for c in last_line.split(",")]
        matched = [c for c in candidates if c in self._expert_nodes]

        if not matched:
            # Fallback: scan full output for expert name mentions
            matched = list(dict.fromkeys(e for e in self._expert_nodes if e in raw))

        if not matched:
            logger.warning("Router output did not match any expert; treating as out-of-scope.")

        return matched

    def _generate_with_model(
        self, messages: List[Dict[str, str]], model, tokenizer, max_new_tokens: int
    ) -> str:
        formatted = apply_chat_template(messages, tokenizer, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", padding=False, truncation=True).to("cuda")
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

    def _route_all(self, questions: List[str], model, tokenizer) -> List[List[str]]:
        """Route all questions in a single base-model session. Returns a list of expert-id lists."""
        routes: List[List[str]] = []
        for i, question in enumerate(questions):
            messages = self._build_router_prompt(question)
            raw = self._generate_with_model(messages, model, tokenizer, max_new_tokens=200)
            experts = self._parse_router_output(raw)
            routes.append(experts)
            logger.info(
                "Q%d routing reasoning:\n%s\n→ experts: %s",
                i + 1, raw, experts or self._out_of_scope_token,
            )
        return routes

    def _generate_answer(self, question: str, expert_id: str) -> str:
        adapter_path = os.path.join(self.slg_path, expert_id)
        model, tokenizer = load_model_with_adapter(
            base_model_path=self._base_model_path,
            adapter_path=adapter_path,
            resize_token_embeddings=True,
        )
        try:
            return self._generate_with_model(
                [create_user_message(question)],
                model,
                tokenizer,
                max_new_tokens=CONFIG["generation"]["max_new_tokens"],
            )
        finally:
            cleanup_model_memory(model, tokenizer)

    @staticmethod
    def _concatenate_answers(expert_answers: Dict[str, str]) -> str:
        return "\n\n".join(f"[{eid}]\n{ans}" for eid, ans in expert_answers.items())

    def ask(self, file: str) -> None:
        validate_file_exists(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paths_config = CONFIG["paths"]
        output_dir = os.path.join(paths_config["answers"], self.experiment)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "slg.json")

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                answers_list: List[Dict] = json.load(f)
            start_index = len(answers_list)
            logger.info("Resuming SLG inference from index %d/%d.", start_index, len(data))
        else:
            answers_list = []
            start_index = 0
            logger.info("Starting fresh SLG inference run.")

        remaining = data[start_index:]
        if not remaining:
            logger.info("All questions already answered.")
            return

        # Phase 1: route all remaining questions in one base-model session
        logger.info("Routing %d questions...", len(remaining))
        router_model, router_tokenizer = load_base_model_and_tokenizer(self._base_model_path)
        try:
            routes = self._route_all(
                [item["question"] for item in remaining],
                router_model,
                router_tokenizer,
            )
        finally:
            cleanup_model_memory(router_model, router_tokenizer)

        # Phase 2: invoke experts and assemble answers
        for i, (item, expert_ids) in enumerate(zip(remaining, routes), start=start_index):
            logger.info("Answering %d/%d — title: %s", i + 1, len(data), item["title"])

            if not expert_ids:
                answer = "OUT_OF_SCOPE"
                logger.info("Question is out of scope.")
            elif len(expert_ids) == 1:
                logger.info("Invoking expert '%s'.", expert_ids[0])
                answer = self._generate_answer(item["question"], expert_ids[0])
            else:
                logger.info("Invoking %d experts: %s", len(expert_ids), expert_ids)
                expert_answers: Dict[str, str] = {}
                for eid in expert_ids:
                    expert_answers[eid] = self._generate_answer(item["question"], eid)
                    logger.info("Expert '%s' answered.", eid)
                answer = self._concatenate_answers(expert_answers)

            answers_list.append({
                "chapter": item["chapter"],
                "title": item["title"],
                "question": item["question"],
                "experts": expert_ids,
                "answer": answer,
            })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(answers_list, f, indent=4)
            logger.info(40 * "-")
