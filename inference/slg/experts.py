"""Expert answer generation (per-expert LoRA adapter on a shared base model).

The base model is ``slg.expert_model`` (Qwen-3B by default; formerly LLaMA-1B).
One adapter is loaded at a time and can answer several questions before being
released, so the batch pipeline groups a round's questions by expert to keep
adapter load/unload churn to a minimum.
"""

import os
from typing import List

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_model_with_adapter
from utils.prompt_utils import create_system_message, create_user_message

from inference.slg.generation import generate_batch


class ExpertRunner:
    """Loads expert adapters on demand and generates answers."""

    def __init__(self, slg_path: str):
        paths_cfg = CONFIG["paths"]
        expert_key = CONFIG.get("slg", {}).get("expert_model", "3_2_1b")
        self._base_model_path = os.path.join(
            paths_cfg["downloaded_models"], paths_cfg["models"][expert_key]
        )
        self._slg_path = slg_path
        self._max_new_tokens = int(CONFIG["generation"]["max_new_tokens"])
        self._system_prompt = CONFIG.get("inference_prompt", "")

    def adapter_path(self, expert_id: str) -> str:
        return os.path.join(self._slg_path, expert_id)

    def _messages(self, question: str, carried_context: str):
        msgs = []
        if self._system_prompt:
            msgs.append(create_system_message(self._system_prompt))
        if carried_context:
            question = f"Conversation context so far:\n{carried_context}\n\nQuestion: {question}"
        msgs.append(create_user_message(question))
        return msgs

    def answer_batch(
        self, expert_id: str, questions: List[str], carried_context: str = ""
    ) -> List[str]:
        """Load ``expert_id`` once, answer every question, then release it."""
        adapter_path = self.adapter_path(expert_id)
        logger.info("Loading expert '%s' to answer %d question(s).", expert_id, len(questions))
        model, tokenizer = load_model_with_adapter(
            base_model_path=self._base_model_path,
            adapter_path=adapter_path,
            resize_token_embeddings=True,
        )
        try:
            batch_size = int(CONFIG["generation"].get("expert_batch_size", 1))
            messages_list = [self._messages(q, carried_context) for q in questions]
            return generate_batch(
                messages_list, model, tokenizer, self._max_new_tokens, batch_size
            )
        finally:
            cleanup_model_memory(model, tokenizer)

    def answer(self, expert_id: str, question: str, carried_context: str = "") -> str:
        return self.answer_batch(expert_id, [question], carried_context)[0]
