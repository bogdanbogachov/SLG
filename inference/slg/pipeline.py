"""Small Language Router orchestrator.

Per question the pipeline runs: cosine shortlist -> reasoning router -> expert
answer -> domain verifier -> (reroute on failure) -> aggregate -> compress, with
three online mechanisms learning across the whole run/session:

* **(A) competence router** — every verifier verdict updates a per-expert,
  per-query-region reliability estimate that signs the cosine ranking, so the
  router learns who to trust without labels or retraining.
* **(B) domain verifier** — deterministic engineering checks (numeric/unit
  sanity, format) combined with the 8B critic; emits a pass/fail plus a
  confidence used downstream.
* **(C) calibrated abstention** — a self-supervised confidence threshold decides
  when to answer vs. withhold; an answer the critic passes but whose confidence
  is below the calibrated bar is *not* returned.

Two entry points share the same building blocks:

* :meth:`ask`  — automated batch inference over a QA file. Exactly one expert
  per question, single answer, no multi-turn. To keep model load/unload churn
  low under tight VRAM, questions are processed in *rounds*: route all pending
  questions, answer them grouped by expert, verify them all, then reroute the
  failures into the next round.
* :meth:`chat` — interactive multi-turn session that surfaces the router
  reasoning and each verifier verdict to the user.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from config import CONFIG
from logging_config import logger
from utils.path_utils import (
    ensure_dir,
    get_answers_root,
    get_slg_descriptions_path,
    get_slg_path,
    validate_dir_exists,
    validate_file_exists,
)

from inference.slg.ablation import AblationConfig
from inference.slg.experts import ExpertRunner
from inference.slg.reasoner import Reasoner
from inference.slg.retriever import ExpertRetriever
from inference.slg.session import SessionState
from inference.slg.verifier import DomainVerifier

def list_valid_experts(experiment: str, experts_location: str = None) -> set:
    """Routable experts (LoRA adapter on disk ∩ has a description), from the
    filesystem only — no model loading. Mirrors
    ``SmallLanguageRouter._resolve_valid_experts`` so planners (e.g. the
    scalability sweep) can compute the expert pool on CPU / in the parent
    process without touching a GPU."""
    experiments_dir = CONFIG["paths"]["experiments"]
    slg_path = get_slg_path(experts_location or experiment, experiments_dir)
    adapters = {
        name
        for name in os.listdir(slg_path)
        if os.path.isdir(os.path.join(slg_path, name))
    }
    with open(get_slg_descriptions_path(experiment), "r", encoding="utf-8") as f:
        descriptions = set(json.load(f))
    return adapters & descriptions


# Per-question terminal states
PENDING = "pending"
RESOLVED = "resolved"
REJECTED = "rejected"      # router found no suitable expert
EXHAUSTED = "exhausted"    # verifier rejected every attempt
ABSTAINED = "abstained"    # an answer passed but never cleared the confidence bar


class SmallLanguageRouter:
    def __init__(
        self,
        experts_location: str,
        experiment: str,
        ablation: AblationConfig = None,
        expert_subset=None,
    ):
        self.experiment = experiment
        # Ablation condition for this run (full system by default). Non-full runs
        # write under answers/<experiment><suffix>/ so they never clobber the
        # full run's outputs.
        self.ablation = ablation or AblationConfig()
        self._output_label = experiment + self.ablation.suffix
        self._routing = CONFIG["routing"]
        self._top_k = int(self._routing["top_k_cosine"])
        self._max_reroutes = int(self._routing["max_reroutes"])
        self._rejection_message = self._routing["rejection_message"]
        self._exhausted_message = self._routing["exhausted_message"]
        self._low_confidence_message = self._routing.get(
            "low_confidence_message",
            "No expert could answer this with sufficient confidence.",
        )

        experiments_dir = CONFIG["paths"]["experiments"]
        self._slg_path = get_slg_path(experts_location, experiments_dir)
        validate_dir_exists(
            self._slg_path,
            error_message=(
                f"SLG expert adapters directory not found: {self._slg_path}. "
                "Train SLG experts before running inference."
            ),
        )

        descriptions_path = get_slg_descriptions_path(experiment)
        validate_file_exists(
            descriptions_path,
            error_message=(
                f"Expert descriptions not found: {descriptions_path}. "
                "Run --slg_descriptions before inference."
            ),
        )
        with open(descriptions_path, "r", encoding="utf-8") as f:
            self._descriptions: Dict[str, str] = json.load(f)

        self._valid_experts = self._resolve_valid_experts()
        if expert_subset is not None:
            # Scalability harness: restrict routing to a fixed pool of experts.
            self._valid_experts &= set(expert_subset)
        if not self._valid_experts:
            raise ValueError(
                "No routable experts: need both a LoRA adapter on disk and a description."
            )

        self._retriever = ExpertRetriever(experiment, allowed_experts=self._valid_experts)
        self._runner = ExpertRunner(self._slg_path)
        self._reasoner = Reasoner()
        self._verifier = DomainVerifier(
            self._reasoner,
            require_units=bool(self._routing.get("verifier_require_units", True)),
            deterministic=self.ablation.deterministic,
        )

    # ------------------------------------------------------------- setup
    def _resolve_valid_experts(self) -> set:
        adapters = {
            name
            for name in os.listdir(self._slg_path)
            if os.path.isdir(os.path.join(self._slg_path, name))
        }
        desc = set(self._descriptions)
        if desc - adapters:
            logger.warning("Descriptions without adapters (not routable): %s", sorted(desc - adapters))
        if adapters - desc:
            logger.warning("Adapters without descriptions (not routable): %s", sorted(adapters - desc))
        return adapters & desc

    def _diagnostics_dir(self) -> str:
        d = os.path.join(get_answers_root(self.experiment), self._output_label, "slg_diagnostics")
        ensure_dir(d)
        return d

    # =============================================================== batch
    def ask(self, file: str) -> None:
        validate_file_exists(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_dir = os.path.join(get_answers_root(self.experiment), self._output_label)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "slg.json")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == len(data):
                logger.info("All %d questions already answered in %s; nothing to do.", len(data), output_path)
                return
            logger.info("Existing slg.json is incomplete (%d/%d); recomputing run.", len(existing), len(data))

        questions = [item["question"] for item in data]

        # Embed every question once (Jina), then release it before loading the 8B.
        logger.info("Embedding %d questions for cosine routing...", len(questions))
        q_emb = [self._retriever.embed_query(q) for q in questions]

        session = SessionState(ablation=self.ablation)
        state = self._run_rounds(questions, q_emb, session)

        # Assemble answers in original order for evaluation alignment.
        answers_list = []
        for i, item in enumerate(data):
            answers_list.append({
                "chapter": item.get("chapter"),
                "title": item.get("title"),
                "question": item["question"],
                "experts": state["history"][i],
                "answer": state["answer"][i],
                "status": state["status"][i],
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers_list, f, indent=4)

        self._write_diagnostics(state, session)
        self._log_summary(state)

    def _run_rounds(self, questions: List[str], q_emb: List[np.ndarray], session: SessionState) -> Dict:
        n = len(questions)
        state = {
            "status": [PENDING] * n,
            "answer": [None] * n,
            "history": [[] for _ in range(n)],     # experts tried, in order
            "route_traces": [[] for _ in range(n)],
            "critic_log": [[] for _ in range(n)],
            "best_lowconf": [None] * n,            # (answer, confidence) passed-but-withheld
        }

        for attempt in range(self._max_reroutes):
            pending = [i for i in range(n) if state["status"][i] == PENDING]
            if not pending:
                break
            logger.info("=== Routing round %d/%d — %d pending question(s) ===",
                        attempt + 1, self._max_reroutes, len(pending))

            # --- Route phase (8B resident, online-competence adjusted) ---
            self._reasoner.load()
            assignment: Dict[int, str] = {}
            for i in pending:
                adjustments = session.routing_adjustments(q_emb[i])
                shortlist = [eid for eid, _ in self._retriever.shortlist(q_emb[i], self._top_k, adjustments)]
                trace, chosen = self._reasoner.route(
                    questions[i], shortlist, self._descriptions, max_experts=1,
                    adjustments=adjustments,
                )
                state["route_traces"][i].append(trace)
                if not chosen:
                    # No expert: terminal rejection only if we never chose one
                    # before; otherwise the question has run out of alternatives
                    # and is classified after the loop.
                    if not state["history"][i]:
                        state["status"][i] = REJECTED
                        state["answer"][i] = self._rejection_message
                else:
                    assignment[i] = chosen[0]
                    state["history"][i].append(chosen[0])
            self._reasoner.unload()

            # --- Answer phase (1B adapters, grouped by expert) ---
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, expert in assignment.items():
                groups[expert].append(i)
            round_answers: Dict[int, str] = {}
            for expert, idxs in groups.items():
                outs = self._runner.answer_batch(expert, [questions[i] for i in idxs])
                for i, out in zip(idxs, outs):
                    round_answers[i] = out

            # --- Verify phase (8B resident: domain verifier B + competence A + calibration C) ---
            self._reasoner.load()
            for i, out in round_answers.items():
                expert = assignment[i]
                verdict = self._verifier.verify(
                    questions[i], expert, self._descriptions.get(expert, ""), out
                )
                session.observe_verdict(expert, q_emb[i], verdict)
                entry = {"expert": expert, **verdict.to_dict()}
                state["critic_log"][i].append(entry)

                if verdict.passed and session.accept(verdict.confidence):
                    state["status"][i] = RESOLVED
                    state["answer"][i] = out
                elif verdict.passed:
                    # Passed the critic but below the calibrated confidence bar:
                    # keep the best such answer in case every attempt is withheld.
                    best = state["best_lowconf"][i]
                    if best is None or verdict.confidence > best[1]:
                        state["best_lowconf"][i] = (out, verdict.confidence)
                    logger.info("Q%d expert '%s' passed but below confidence bar (%.2f).",
                                i + 1, expert, verdict.confidence)
                else:
                    logger.info("Verifier FAIL on Q%d expert '%s' (confidence=%.2f).",
                                i + 1, expert, verdict.confidence)
            self._reasoner.unload()

        # Classify anything still pending after the reroute budget is spent.
        for i in range(n):
            if state["status"][i] != PENDING:
                continue
            if state["best_lowconf"][i] is not None:
                state["status"][i] = ABSTAINED
                state["answer"][i] = self._low_confidence_message
            elif state["history"][i]:
                state["status"][i] = EXHAUSTED
                state["answer"][i] = self._exhausted_message
            else:
                state["status"][i] = REJECTED
                state["answer"][i] = self._rejection_message
        return state

    # -------------------------------------------------------- diagnostics
    def _write_diagnostics(self, state: Dict, session: SessionState) -> None:
        d = self._diagnostics_dir()
        with open(os.path.join(d, "slg_routes.json"), "w", encoding="utf-8") as f:
            json.dump(state["history"], f, indent=2)
        with open(os.path.join(d, "critic_log.json"), "w", encoding="utf-8") as f:
            json.dump(state["critic_log"], f, indent=2)
        with open(os.path.join(d, "route_traces.json"), "w", encoding="utf-8") as f:
            json.dump(state["route_traces"], f, indent=2)
        # (A) online-competence learning signal and (C) calibration trace — the
        # data behind the paper's "routing improves online" and reliability figures.
        with open(os.path.join(d, "competence_log.json"), "w", encoding="utf-8") as f:
            json.dump(session.competence.log, f, indent=2)
        with open(os.path.join(d, "calibration_log.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_error": session.calibrator.target_error,
                    "final_threshold": session.calibrator.threshold(),
                    "coverage": session.calibrator.coverage(),
                    "threshold_history": session.calibrator.threshold_history,
                },
                f, indent=2,
            )

    def _log_summary(self, state: Dict) -> None:
        from collections import Counter
        counts = Counter(state["status"])
        logger.info(
            "SLG batch complete — resolved=%d rejected=%d exhausted=%d abstained=%d",
            counts.get(RESOLVED, 0), counts.get(REJECTED, 0),
            counts.get(EXHAUSTED, 0), counts.get(ABSTAINED, 0),
        )

    # ================================================================ chat
    def chat(self, input_fn=input, output_fn=print) -> None:
        """Interactive multi-turn session.

        The router may select multiple experts (when ``interactive_multi_expert``
        is set); each answer is verified, failures update the competence model
        and trigger rerouting, the surviving answers are aggregated, and a
        compressed form is carried into the next turn. The router's reasoning and
        each verifier verdict (with confidence) are surfaced to the user, and the
        system abstains when no answer clears the calibrated confidence bar.
        """
        multi = bool(self._routing.get("interactive_multi_expert", True))
        session = SessionState(ablation=self.ablation)
        output_fn(
            "SLG interactive session. Type your engineering question "
            "('exit' or 'quit' to leave).\n"
        )
        while True:
            try:
                question = input_fn("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("\nEnding session.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                output_fn("Ending session.")
                break
            self._chat_turn(question, session, multi, output_fn)

    def _chat_turn(self, question: str, session: SessionState, multi: bool, output_fn) -> None:
        q_emb = self._retriever.embed_query(question)
        accepted: List[tuple] = []   # (expert_id, answer)
        best_lowconf = None          # (expert_id, answer, confidence)
        ever_chose = False

        for attempt in range(self._max_reroutes):
            adjustments = session.routing_adjustments(q_emb)
            taken = {e for e, _ in accepted}
            shortlist = [
                e for e, _ in self._retriever.shortlist(q_emb, self._top_k, adjustments)
                if e not in taken
            ]
            if not shortlist:
                break
            max_experts = len(shortlist) if multi else 1

            self._reasoner.load()
            trace, chosen = self._reasoner.route(
                question, shortlist, self._descriptions, max_experts,
                adjustments=adjustments, carried_context=session.carried_context,
            )
            self._reasoner.unload()
            output_fn(f"\n[router reasoning]\n{trace}\n[router chose] {chosen or 'NONE'}")

            if not chosen:
                break
            ever_chose = True

            # Answer each chosen expert (carrying compressed context).
            answers = {
                e: self._runner.answer(e, question, session.carried_context) for e in chosen
            }

            # Verify each answer; surface verdict; update competence + calibration.
            self._reasoner.load()
            for expert in chosen:
                verdict = self._verifier.verify(
                    question, expert, self._descriptions.get(expert, ""), answers[expert]
                )
                session.observe_verdict(expert, q_emb, verdict)
                output_fn(
                    f"\n[verifier · {expert}] {'PASS' if verdict.passed else 'FAIL'} "
                    f"(confidence={verdict.confidence:.2f})\n{verdict.critique}"
                )
                if verdict.passed and session.accept(verdict.confidence):
                    accepted.append((expert, answers[expert]))
                elif verdict.passed:
                    output_fn(f"[abstain-guard] '{expert}' passed but below the confidence bar; withholding.")
                    if best_lowconf is None or verdict.confidence > best_lowconf[2]:
                        best_lowconf = (expert, answers[expert], verdict.confidence)
                else:
                    output_fn(f"[competence] '{expert}' demoted for similar questions; will reroute.")
            self._reasoner.unload()

            if accepted:
                # Got at least one confident answer this round; stop rerouting.
                break

        if accepted:
            self._reasoner.load()
            final = self._reasoner.aggregate(question, accepted)
            combined = (session.carried_context + "\n" + final).strip()
            session.set_context(self._reasoner.compress(combined))
            self._reasoner.unload()
            output_fn(f"\nassistant> {final}")
        elif best_lowconf is not None:
            output_fn(f"\nassistant> {self._low_confidence_message}")
        elif not ever_chose:
            output_fn(f"\nassistant> {self._rejection_message}")
        else:
            output_fn(f"\nassistant> {self._exhausted_message}")
