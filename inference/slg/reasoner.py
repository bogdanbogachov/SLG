"""A reasoning LLM that plays several prompt-selected roles in the pipeline.

The same class backs two *different* model instances (see ``model_key``):

* the **reasoner** (Qwen-3B by default): ``route`` (router **tiebreaker** only —
  the classifier is the primary router), ``aggregate`` (merge expert answers),
  ``compress`` (shrink an answer into carried chat context).
* the **critic** (Llama-3.1-8B by default, a *different family* from the experts
  to avoid self-preference bias): ``criticize`` — the LLM half of the domain
  verifier (B).

Each instance loads its model on demand and reuses it across calls.
"""

from typing import Dict, List, Optional, Tuple

from config import CONFIG
from logging_config import logger
from utils.model_loader import cleanup_model_memory, load_base_model_and_tokenizer
from utils.prompt_utils import apply_chat_template, create_system_message, create_user_message

from inference.slg.generation import choice_probs, generate, generate_batch

# The critic now writes a short assessment; the verdict is read from logits.
_CRITIQUE_MAX_TOKENS = 192


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

# The critic judges the answer against the *question* only. The producing expert's
# advertised remit is deliberately withheld: naming it invites the critic to reject
# a correct answer for being outside that remit, which is a routing error, not an
# answer defect — and that would confound mechanism (B) with the router.
_CRITIC_SYSTEM = (
    "You are a fair quality checker for an engineering question-answering system. "
    "You are given a user question and a candidate answer. Your job is to catch "
    "answers that are genuinely wrong or useless — NOT to demand perfection. Real, "
    "helpful answers vary in style, length, and completeness; accept them.\n"
    "An answer is acceptable if ALL of these hold:\n"
    "1. it is on-topic and addresses the question (even if partially),\n"
    "2. it is factually plausible — no clearly false statements or made-up terms,\n"
    "3. it is coherent and not degenerate (not empty, not looping/repeated text),\n"
    "4. any numbers/units it gives are not absurd or self-contradictory.\n"
    "It is unacceptable only if it is off-topic, clearly incorrect, nonsensical, "
    "degenerate, or empty. Do not reject an answer merely for being brief, informal, "
    "missing a caveat, or omitting detail you would have liked. When in doubt, accept.\n"
    "Reply with one or two sentences naming any real problems you find, or stating "
    "that the answer is sound. Do not state a verdict — you will be asked for it "
    "separately."
)

_CRITIC_USER = (
    "Question: {question}\n\n"
    "Answer:\n{answer}\n\n"
    "Briefly assess this answer."
)

# Appended to the critic's own critique to read the verdict off the next-token
# distribution rather than parsing generated text. The two continuations must
# differ in their first token.
_VERDICT_CUE = "\nVERDICT:"
_VERDICT_CHOICES = (" PASS", " FAIL")

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
    """Holds one reasoning model (selected by ``model_key``) and its roles.

    ``model_key`` is a key of ``paths.models`` (e.g. ``qwen_3b`` for the reasoner,
    ``3_1_8b`` for the critic). Defaults to the 8B for backward compatibility.
    """

    def __init__(self, model_key: str = "3_1_8b"):
        import os

        paths_cfg = CONFIG["paths"]
        self._model_key = model_key
        self._model_path = os.path.join(
            paths_cfg["downloaded_models"], paths_cfg["models"][model_key]
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

    @staticmethod
    def _decode_guards() -> Tuple[float, int]:
        """Anti-repetition settings for the structured-output roles.

        Every role on this class must reproduce tokens its prompt already contains
        — the critic's ``VERDICT: PASS``/``VERDICT: FAIL``, the router's expert
        names. The expert-answer guards would forbid exactly that, so they are
        disabled here (see ``generation.reasoner_*`` in config.yaml).
        """
        gen = CONFIG["generation"]
        return (
            float(gen.get("reasoner_repetition_penalty", 1.0)),
            int(gen.get("reasoner_no_repeat_ngram_size", 0)),
        )

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        if self.model is None:
            self.load()
        messages = [create_system_message(system), create_user_message(user)]
        rep_penalty, ngram_size = self._decode_guards()
        return generate(
            messages, self.model, self.tokenizer, max_new_tokens,
            repetition_penalty=rep_penalty, no_repeat_ngram_size=ngram_size,
        )

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
        rep_penalty, ngram_size = self._decode_guards()
        return generate_batch(
            messages_list, self.model, self.tokenizer, max_new_tokens, batch_size,
            repetition_penalty=rep_penalty, no_repeat_ngram_size=ngram_size,
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
    def criticize(self, question: str, answer: str, expert_id: str = "") -> Tuple[bool, float, str]:
        """Return ``(passed, p_pass, critique_text)`` for one answer.

        ``expert_id`` is for logging only — it is never shown to the critic.
        """
        return self.criticize_batch([(question, expert_id, answer)])[0]

    @staticmethod
    def _build_critic_user(question: str, answer: str) -> str:
        return _CRITIC_USER.format(question=question, answer=answer)

    def criticize_batch(
        self, items: List[Tuple[str, str, str]]
    ) -> List[Tuple[bool, float, str]]:
        """Batched critic. ``items`` are ``(question, expert_id, answer)`` tuples;
        returns ``(passed, p_pass, critique)`` per item in order.

        Two steps. The critic first writes a short free-text assessment. We then
        append ``VERDICT:`` to its own words and read ``P(PASS)`` off the
        next-token distribution (one extra forward pass, no decoding). That
        probability *is* the confidence: it is continuous, always defined, and
        needs no parsing — where a self-reported ``CONFIDENCE:`` integer clumps on
        a handful of values (and vanishes entirely when the model omits the line),
        leaving the abstention calibrator (C) nothing to threshold on.

        The critic judges each answer independently, so batching only decodes them
        together — it does not couple them.
        """
        if not items:
            return []
        if self.model is None:
            self.load()

        messages_list = [
            [create_system_message(_CRITIC_SYSTEM),
             create_user_message(self._build_critic_user(q, ans))]
            for q, _eid, ans in items
        ]
        prompts = [(_CRITIC_SYSTEM, self._build_critic_user(q, ans)) for q, _eid, ans in items]
        critiques = self._generate_many(prompts, max_new_tokens=_CRITIQUE_MAX_TOKENS)

        # Re-render each prompt with the critic's own assessment appended, then cue
        # the verdict token so the logits answer "given what you just said, PASS?"
        scoring_prompts = [
            apply_chat_template(messages, self.tokenizer, add_generation_prompt=True)
            + critique + _VERDICT_CUE
            for messages, critique in zip(messages_list, critiques)
        ]
        batch_size = int(CONFIG["generation"].get("reasoner_batch_size", 1))
        probs = choice_probs(
            scoring_prompts, list(_VERDICT_CHOICES), self.model, self.tokenizer, batch_size
        )

        out: List[Tuple[bool, float, str]] = []
        for (_q, eid, _ans), critique, (p_pass, _p_fail) in zip(items, critiques, probs):
            passed = p_pass >= 0.5
            logger.info(
                "Critic verdict for '%s': %s (P(PASS)=%.3f)",
                eid, "PASS" if passed else "FAIL", p_pass,
            )
            out.append((passed, float(p_pass), critique))
        return out

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
