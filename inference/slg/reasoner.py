"""The reasoning LLM (LLaMA 3.1-8B) that plays four roles in the pipeline.

A single 8B model is loaded once and reused, with a different prompt per role:

* **route**      — reason over a cosine shortlist and pick the expert(s) to answer.
* **criticize**  — judge an expert answer against seven quality criteria.
* **aggregate**  — merge several expert answers into one cohesive answer.
* **compress**   — shrink an answer into compact context carried across chat turns.
"""

import re
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_base_model_and_tokenizer
from utils.prompt_utils import create_system_message, create_user_message

from inference.slg.generation import generate


# --------------------------------------------------------------------- prompts
_ROUTER_SYSTEM = (
    "You are a reasoning-based routing assistant for a question-answering system. "
    "You are given a user question and a shortlist of candidate specialist experts "
    "with short descriptions. Follow these steps:\n"
    "1. Briefly reason about which candidate expert(s) are most relevant to the question.\n"
    "2. On the very last line, output 'Expert: ' followed by the chosen expert name(s), "
    "using the exact names provided. Choose at most {max_experts}. If you choose more than "
    "one, separate the names with commas.\n"
    "   If none of the candidates can address the question, output 'NONE' on the last line.\n"
    "Do not choose an expert listed as previously-failed unless no other candidate fits.\n\n"
    "Example last line (in scope):  Expert: expert_a\n"
    "Example last line (out of scope): NONE"
)

_ROUTER_USER = (
    "Question: {question}\n"
    "{context_block}"
    "Candidate experts:\n{expert_list}\n"
    "{penalty_block}"
    "\nReason about which expert(s) (at most {max_experts}) can best answer this "
    "question, then on the final line output the expert name(s) or NONE."
)

_CRITIC_SYSTEM = (
    "You are a strict quality critic for an engineering question-answering system. "
    "You are given a user question, the specialist expert that produced an answer "
    "(with its domain description), and the answer itself. Check whether the answer:\n"
    "1. directly answers the user's question\n"
    "2. uses the selected expert's domain appropriately\n"
    "3. avoids unsupported claims\n"
    "4. mentions uncertainty when needed\n"
    "5. follows the requested format\n"
    "6. is complete but concise\n"
    "7. does not contradict the expert's known limitations\n"
    "Briefly note any problems you find. On the very last line output exactly "
    "'VERDICT: PASS' if the answer is acceptable on all critical points, or "
    "'VERDICT: FAIL' otherwise."
)

_CRITIC_USER = (
    "Question: {question}\n\n"
    "Expert: {expert_id} — {description}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluate the answer against the seven criteria, then output the verdict line."
)

_AGGREGATOR_SYSTEM = (
    "You are an answer aggregator. You are given one or more answers from different "
    "specialist experts responding to the same user question. Merge them into a single "
    "cohesive, non-redundant answer that preserves all correct technical detail. "
    "Do not add information that is not present in the provided answers."
)

_AGGREGATOR_USER = (
    "Question: {question}\n\n"
    "Expert answers:\n{answers}\n\n"
    "Write one cohesive answer."
)

_COMPRESSOR_SYSTEM = (
    "You compress an answer into a compact factual summary that will be reused as "
    "context for a follow-up conversation. Keep all key facts, numbers, and entities. "
    "Be concise (at most {max_words} words). Output only the summary."
)

_COMPRESSOR_USER = "Answer to compress:\n{answer}"


class Reasoner:
    """Holds the resident 8B reasoning model and exposes the four roles."""

    def __init__(self):
        import os

        paths_cfg = CONFIG["paths"]
        self._model_path = os.path.join(
            paths_cfg["downloaded_models"], paths_cfg["models"]["3_1_8b"]
        )
        self._routing = CONFIG["routing"]
        self._out_of_scope = CONFIG["slg"].get("out_of_scope_token", "NONE")
        self.model = None
        self.tokenizer = None

    # ----------------------------------------------------------- lifecycle
    def load(self) -> "Reasoner":
        if self.model is None:
            self.model, self.tokenizer = load_base_model_and_tokenizer(self._model_path)
        return self

    def unload(self) -> None:
        if self.model is not None:
            cleanup_model_memory(self.model, self.tokenizer)
            self.model = None
            self.tokenizer = None

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        if self.model is None:
            self.load()
        messages = [create_system_message(system), create_user_message(user)]
        return generate(messages, self.model, self.tokenizer, max_new_tokens)

    # --------------------------------------------------------------- route
    def route(
        self,
        question: str,
        shortlist: List[str],
        descriptions: Dict[str, str],
        max_experts: int,
        penalized: Optional[Dict[str, float]] = None,
        carried_context: str = "",
    ) -> Tuple[str, List[str]]:
        """Return (reasoning_trace, chosen_expert_ids) restricted to the shortlist."""
        if not shortlist:
            return "", []

        expert_list = "\n".join(
            f"- {eid}: {descriptions.get(eid, '(no description)')}" for eid in shortlist
        )
        context_block = (
            f"Conversation context so far:\n{carried_context}\n\n" if carried_context else ""
        )
        penalized = penalized or {}
        flagged = [eid for eid in shortlist if penalized.get(eid, 0) > 0]
        penalty_block = (
            "Previously-failed experts (avoid unless necessary): "
            + ", ".join(flagged)
            + "\n"
            if flagged
            else ""
        )

        system = _ROUTER_SYSTEM.format(max_experts=max_experts)
        user = _ROUTER_USER.format(
            question=question,
            context_block=context_block,
            expert_list=expert_list,
            penalty_block=penalty_block,
            max_experts=max_experts,
        )
        raw = self._generate(system, user, max_new_tokens=512)
        chosen = self._parse_route(raw, shortlist, max_experts)
        logger.info(
            "Routing reasoning:\n%s\n-> experts: %s",
            raw,
            chosen or self._out_of_scope,
        )
        return raw, chosen

    def _parse_route(self, raw: str, shortlist: List[str], max_experts: int) -> List[str]:
        if "assistant" in raw:
            raw = raw.split("assistant")[-1]
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        last_line = lines[-1] if lines else ""

        if last_line.upper() == self._out_of_scope.upper():
            return []

        for prefix in ("experts:", "expert:"):
            if last_line.lower().startswith(prefix):
                last_line = last_line[len(prefix):].strip()
                break

        candidates = [c.strip().rstrip(".") for c in last_line.split(",")]
        matched = [c for c in candidates if c in shortlist]

        if not matched:  # fallback: scan whole trace for any shortlisted name
            matched = [eid for eid in shortlist if eid in raw]

        # de-duplicate while preserving order, then cap
        seen, ordered = set(), []
        for eid in matched:
            if eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        if not ordered:
            logger.warning("Router output matched no shortlisted expert; treating as rejection.")
        return ordered[:max_experts]

    # ------------------------------------------------------------ criticize
    def criticize(
        self, question: str, expert_id: str, description: str, answer: str
    ) -> Tuple[bool, str]:
        """Return (passed, critique_text)."""
        user = _CRITIC_USER.format(
            question=question,
            expert_id=expert_id,
            description=description or "(no description)",
            answer=answer,
        )
        raw = self._generate(_CRITIC_SYSTEM, user, max_new_tokens=512)
        passed = self._parse_verdict(raw)
        logger.info("Critic verdict for '%s': %s", expert_id, "PASS" if passed else "FAIL")
        return passed, raw

    @staticmethod
    def _parse_verdict(raw: str) -> bool:
        if "assistant" in raw:
            raw = raw.split("assistant")[-1]
        matches = re.findall(r"VERDICT\s*:\s*(PASS|FAIL)", raw, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().upper() == "PASS"
        # No explicit verdict line: be conservative and treat as failure.
        logger.warning("Critic produced no parseable verdict; treating as FAIL.")
        return False

    # ------------------------------------------------------------- aggregate
    def aggregate(self, question: str, labeled_answers: List[Tuple[str, str]]) -> str:
        """Merge (expert_id, answer) pairs into one cohesive answer."""
        if len(labeled_answers) == 1:
            return labeled_answers[0][1]
        answers = "\n\n".join(f"[{eid}]\n{ans}" for eid, ans in labeled_answers)
        user = _AGGREGATOR_USER.format(question=question, answers=answers)
        return self._generate(
            _AGGREGATOR_SYSTEM, user, max_new_tokens=CONFIG["generation"]["max_new_tokens"]
        )

    # -------------------------------------------------------------- compress
    def compress(self, answer: str) -> str:
        """Compress an answer into compact carried context."""
        max_tokens = int(self._routing.get("compression_max_tokens", 256))
        system = _COMPRESSOR_SYSTEM.format(max_words=max_tokens)
        user = _COMPRESSOR_USER.format(answer=answer)
        return self._generate(system, user, max_new_tokens=max_tokens)
