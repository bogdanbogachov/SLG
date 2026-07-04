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

from inference.slg.generation import generate, generate_batch


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
    "{competence_block}"
    "\nReason about which expert(s) (at most {max_experts}) can best answer this "
    "question, then on the final line output the expert name(s) or NONE."
)

_CRITIC_SYSTEM = (
    "You are a strict, domain-grounded quality critic for an engineering "
    "question-answering system. You are given a user question, the specialist "
    "expert that produced an answer (with its domain description), and the answer "
    "itself. Engineering answers carry real risk: a wrong number, an inconsistent "
    "unit, or a violated physical constraint is worse than a fluent but empty "
    "reply. Check whether the answer:\n"
    "1. directly answers the user's question\n"
    "2. uses the selected expert's domain appropriately\n"
    "3. avoids unsupported claims\n"
    "4. mentions uncertainty when needed\n"
    "5. follows the requested format\n"
    "6. is complete but concise\n"
    "7. does not contradict the expert's known limitations\n"
    "8. is numerically and dimensionally sound — quantities carry consistent "
    "units, magnitudes are physically plausible, and no calculation contradicts "
    "itself\n"
    "Briefly note any problems you find. Then output TWO final lines, each on its "
    "own line and nothing after them:\n"
    "CONFIDENCE: <integer 0-100, your confidence that the answer is correct and safe to return>\n"
    "VERDICT: PASS   (if acceptable on all critical points) or VERDICT: FAIL (otherwise)"
)

_CRITIC_USER = (
    "Question: {question}\n\n"
    "Expert: {expert_id} — {description}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluate the answer against the eight criteria, then output the CONFIDENCE "
    "and VERDICT lines."
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

    def _generate_many(self, prompts: List[Tuple[str, str]], max_new_tokens: int) -> List[str]:
        """Batched counterpart of :meth:`_generate` — decode every (system, user)
        prompt, filling the 8B GPU ``reasoner_batch_size`` prompts at a time."""
        if not prompts:
            return []
        if self.model is None:
            self.load()
        messages_list = [
            [create_system_message(system), create_user_message(user)]
            for system, user in prompts
        ]
        batch_size = int(CONFIG["generation"].get("reasoner_batch_size", 1))
        return generate_batch(
            messages_list, self.model, self.tokenizer, max_new_tokens, batch_size
        )

    # --------------------------------------------------------------- route
    def route(
        self,
        question: str,
        shortlist: List[str],
        descriptions: Dict[str, str],
        max_experts: int,
        adjustments: Optional[Dict[str, float]] = None,
        carried_context: str = "",
    ) -> Tuple[str, List[str]]:
        """Return (reasoning_trace, chosen_expert_ids) restricted to the shortlist.

        ``adjustments`` are the signed online-competence deltas for this query
        region (see :mod:`inference.slg.competence`): positive means the expert
        has proven reliable on similar questions, negative means it has failed.
        They are surfaced to the router as soft guidance — the model still
        reasons over the descriptions and makes the final choice.
        """
        if not shortlist:
            return "", []

        system, user = self._build_route_prompt(
            question, shortlist, descriptions, max_experts, adjustments, carried_context
        )
        raw = self._generate(system, user, max_new_tokens=512)
        chosen = self._parse_route(raw, shortlist, max_experts)
        logger.info(
            "Routing reasoning:\n%s\n-> experts: %s",
            raw,
            chosen or self._out_of_scope,
        )
        return raw, chosen

    def _build_route_prompt(
        self,
        question: str,
        shortlist: List[str],
        descriptions: Dict[str, str],
        max_experts: int,
        adjustments: Optional[Dict[str, float]] = None,
        carried_context: str = "",
    ) -> Tuple[str, str]:
        """Render the (system, user) router prompt for one question."""
        expert_list = "\n".join(
            f"- {eid}: {descriptions.get(eid, '(no description)')}" for eid in shortlist
        )
        context_block = (
            f"Conversation context so far:\n{carried_context}\n\n" if carried_context else ""
        )
        adjustments = adjustments or {}
        proven = [eid for eid in shortlist if adjustments.get(eid, 0.0) > 1e-6]
        struggling = [eid for eid in shortlist if adjustments.get(eid, 0.0) < -1e-6]
        competence_block = ""
        if proven:
            competence_block += (
                "Experts that have answered similar questions well (prefer): "
                + ", ".join(proven) + "\n"
            )
        if struggling:
            competence_block += (
                "Experts that have failed similar questions (avoid unless necessary): "
                + ", ".join(struggling) + "\n"
            )

        system = _ROUTER_SYSTEM.format(max_experts=max_experts)
        user = _ROUTER_USER.format(
            question=question,
            context_block=context_block,
            expert_list=expert_list,
            competence_block=competence_block,
            max_experts=max_experts,
        )
        return system, user

    def route_batch(
        self,
        questions: List[str],
        shortlists: List[List[str]],
        descriptions: Dict[str, str],
        max_experts: int,
        adjustments_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[Tuple[str, List[str]]]:
        """Route several questions in one batched 8B pass.

        Each question's shortlist and competence ``adjustments`` are computed by
        the caller from the *round-start* session state (read-only during the
        route phase), so batching does not change what any question sees — it
        only decodes the router traces together. Returns ``(trace, chosen)`` per
        question, in input order.
        """
        n = len(questions)
        adjustments_list = adjustments_list or [None] * n
        results: List[Tuple[str, List[str]]] = [("", []) for _ in range(n)]

        prompts: List[Tuple[str, str]] = []
        idx_map: List[int] = []
        for i in range(n):
            if not shortlists[i]:
                continue
            prompts.append(
                self._build_route_prompt(
                    questions[i], shortlists[i], descriptions, max_experts,
                    adjustments_list[i],
                )
            )
            idx_map.append(i)

        raws = self._generate_many(prompts, max_new_tokens=512)
        for i, raw in zip(idx_map, raws):
            chosen = self._parse_route(raw, shortlists[i], max_experts)
            results[i] = (raw, chosen)
            logger.info(
                "Routing reasoning:\n%s\n-> experts: %s", raw, chosen or self._out_of_scope
            )
        return results

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
    ) -> Tuple[bool, float, str]:
        """Return (passed, confidence, critique_text).

        ``confidence`` is the critic's self-reported probability (in [0, 1]) that
        the answer is correct and safe to return. It is the LLM half of the
        domain verifier and feeds the abstention calibrator.
        """
        user = self._build_critic_user(question, expert_id, description, answer)
        raw = self._generate(_CRITIC_SYSTEM, user, max_new_tokens=512)
        passed = self._parse_verdict(raw)
        confidence = self._parse_confidence(raw, passed)
        logger.info(
            "Critic verdict for '%s': %s (confidence=%.2f)",
            expert_id, "PASS" if passed else "FAIL", confidence,
        )
        return passed, confidence, raw

    @staticmethod
    def _build_critic_user(question: str, expert_id: str, description: str, answer: str) -> str:
        return _CRITIC_USER.format(
            question=question,
            expert_id=expert_id,
            description=description or "(no description)",
            answer=answer,
        )

    def criticize_batch(
        self, items: List[Tuple[str, str, str, str]]
    ) -> List[Tuple[bool, float, str]]:
        """Batched critic. ``items`` are ``(question, expert_id, description,
        answer)`` tuples; returns ``(passed, confidence, critique)`` per item in
        order. The critic judges each answer independently, so batching only
        decodes the verdicts together — it does not couple them."""
        prompts = [
            (_CRITIC_SYSTEM, self._build_critic_user(q, eid, desc, ans))
            for q, eid, desc, ans in items
        ]
        raws = self._generate_many(prompts, max_new_tokens=512)
        out: List[Tuple[bool, float, str]] = []
        for (q, eid, desc, ans), raw in zip(items, raws):
            passed = self._parse_verdict(raw)
            confidence = self._parse_confidence(raw, passed)
            logger.info(
                "Critic verdict for '%s': %s (confidence=%.2f)",
                eid, "PASS" if passed else "FAIL", confidence,
            )
            out.append((passed, confidence, raw))
        return out

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

    @staticmethod
    def _parse_confidence(raw: str, passed: bool) -> float:
        """Parse the 'CONFIDENCE: <0-100>' line into [0, 1]."""
        if "assistant" in raw:
            raw = raw.split("assistant")[-1]
        matches = re.findall(r"CONFIDENCE\s*:\s*(\d{1,3})", raw, flags=re.IGNORECASE)
        if matches:
            value = max(0, min(100, int(matches[-1]))) / 100.0
            return value
        # No parseable confidence: fall back to a verdict-consistent default so a
        # PASS is not silently treated as low confidence (and vice-versa).
        return 0.6 if passed else 0.4

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
