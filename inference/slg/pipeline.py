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
from typing import Dict, List, Optional, Tuple

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
from inference.slg.classifier import ExpertClassifier
from inference.slg.experts import ExpertRunner
from inference.slg.generation import batch_size_for
from inference.slg.reasoner import Reasoner
from inference.slg.retriever import ExpertRetriever
from inference.slg.session import SessionState
from inference.slg.verifier import DomainVerifier, Verdict

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


def _contiguous_shard(n: int, shard_index: int, num_shards: int) -> List[int]:
    """Global indices of contiguous shard ``shard_index`` of ``n`` items split
    into ``num_shards`` near-equal blocks (last block may be shorter/empty)."""
    if num_shards <= 1:
        return list(range(n))
    size = -(-n // num_shards)  # ceil division
    start = min(n, shard_index * size)
    end = min(n, start + size)
    return list(range(start, end))


def merge_sharded_base(experiment: str, qa_file: str, shard_results: List[Dict]) -> None:
    """Merge base-ablation shard results (from :meth:`SmallLanguageRouter.answer_shard`)
    into the single ``__base`` run's ``slg.json`` + diagnostics, in original
    question order. Pure CPU — no model or router construction needed."""
    from collections import Counter

    label = experiment + "__base"
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)

    answers = [None] * n
    history = [None] * n
    traces = [None] * n
    critic = [None] * n
    competence_log: List = []
    threshold_history: List = []
    calib: Dict = {}

    for res in sorted(shard_results, key=lambda r: r["indices"][0] if r["indices"] else -1):
        for local, gi in enumerate(res["indices"]):
            answers[gi] = res["records"][local]
            history[gi] = res["history"][local]
            traces[gi] = res["route_traces"][local]
            critic[gi] = res["critic_log"][local]
        competence_log.extend(res["competence_log"])
        threshold_history.extend(res["calibration"]["threshold_history"])
        calib = res["calibration"]

    out_dir = os.path.join(get_answers_root(experiment), label)
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "slg.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4)

    d = os.path.join(out_dir, "slg_diagnostics")
    ensure_dir(d)
    for name, payload in (
        ("slg_routes.json", history),
        ("critic_log.json", critic),
        ("route_traces.json", traces),
        ("competence_log.json", competence_log),
    ):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    with open(os.path.join(d, "calibration_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "target_error": calib.get("target_error"),
            "final_threshold": calib.get("final_threshold"),
            "coverage": calib.get("coverage"),
            "threshold_history": threshold_history,
        }, f, indent=2)

    counts = Counter(a["status"] for a in answers if a)
    logger.info("SLG base (sharded ×%d) complete — %s", len(shard_results), dict(counts))


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
        output_suffix: str = "",
    ):
        self.experiment = experiment
        # Ablation condition for this run (full system by default). Non-full runs
        # write under answers/<experiment><suffix>/ so they never clobber the
        # full run's outputs. ``output_suffix`` (e.g. "__limit50" for a --limit
        # quick check) isolates outputs further without changing where adapters
        # and descriptions are read from (still keyed by ``experiment``).
        self.ablation = ablation or AblationConfig()
        self._output_label = experiment + self.ablation.suffix + output_suffix
        self._routing = CONFIG["routing"]
        self._top_k = int(self._routing["top_k_cosine"])
        self._max_reroutes = int(self._routing["max_reroutes"])
        # Router (classifier decider + reasoner tiebreaker) knobs.
        _router = self._routing.get("router", {})
        self._router_tie_margin = float(_router.get("tie_margin", 0.15))
        self._router_prob_floor = float(_router.get("prob_floor", 0.10))
        self._router_cosine_floor = float(_router.get("cosine_floor", 0.0))
        self._router_multi_threshold = float(_router.get("multi_threshold", 0.30))
        # Chunk sizes for intra-round checkpointing: a round's answer/verify
        # phases are checkpointed after each chunk, so an interrupted round-1
        # pass over the full test set resumes near where it stopped rather than
        # restarting. Match the generation batch sizes so a checkpoint lands
        # once per decoded batch (no extra generation granularity introduced).
        # Resolved per model, so swapping a bigger model into a role (a 14B expert
        # for the 3B) picks up that model's batch. `_reasoner_batch` chunks the
        # *critic* verify calls below, so it keys off `critic_model`.
        _slg_cfg = CONFIG.get("slg", {})
        self._expert_batch = batch_size_for(
            _slg_cfg.get("expert_model", "qwen_3b"), "expert_batch_size"
        )
        self._reasoner_batch = batch_size_for(
            _slg_cfg.get("critic_model", "3_1_8b"), "reasoner_batch_size"
        )
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
        self._classifier = ExpertClassifier(experiment, allowed_experts=self._valid_experts)
        if self._classifier.available:
            logger.info("Router: trained classifier (%d experts).", len(self._classifier.labels))
        else:
            logger.warning(
                "Router: no trained classifier at experiments/%s/slg_router — "
                "falling back to cosine top-1 (run --finetune_router for the learned router).",
                experiment,
            )
        self._runner = ExpertRunner(self._slg_path)
        # Two distinct reasoning models: the reasoner (Qwen-3B) for the router
        # tiebreaker + aggregate + compress, and the critic (Llama-8B, different
        # family) for the domain verifier (B). Each loads its own weights on demand.
        _slg = CONFIG.get("slg", {})
        self._reasoner = Reasoner(model_key=_slg.get("reasoner_model", "qwen_3b"))
        self._critic = Reasoner(model_key=_slg.get("critic_model", "3_1_8b"))
        self._verifier = DomainVerifier(
            self._critic,
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

    # ------------------------------------------------------------- routing
    def _score_experts(
        self, questions: List[str], q_embs: List[np.ndarray]
    ) -> Tuple[List[Dict[str, float]], str]:
        """Return ``(per-question score dict over the allowed pool, backend)``.

        Backend ``"logits"`` (classifier available) → raw class logits, softmaxed
        over the candidate set in :meth:`_rank`. Backend ``"cosine"`` (fallback)
        → raw cosine similarities, ranked as-is. The two are *not* comparable, so
        each has its own reject floor (see :meth:`_route_pending`).
        """
        if self._classifier.available:
            self._classifier.load()
            scores = self._classifier.logits_batch(questions)
            self._classifier.unload()
            return scores, "logits"
        return [self._retriever.scores(emb) for emb in q_embs], "cosine"

    def _rank(
        self,
        scores: Dict[str, float],
        backend: str,
        adjustments: Optional[Dict[str, float]],
        exclude: set,
    ) -> List[Tuple[str, float]]:
        """Rank the *candidate* experts (allowed minus ``exclude``).

        For the classifier backend the candidate logits are **softmaxed over the
        candidate set** first, so the returned scores are a proper probability
        distribution over exactly the experts still in play — masking a pool
        subset or excluding a failed expert on reroute can no longer strand the
        probability mass on an unavailable expert. The competence delta (A) is
        then added to the score, exactly as it was added to the cosine score
        before (the online mechanism is unchanged).
        """
        cand = {e: s for e, s in scores.items() if e not in exclude}
        if not cand:
            return []
        if backend == "logits":
            keys = list(cand)
            arr = np.asarray([cand[k] for k in keys], dtype="float64")
            arr = arr - arr.max()  # stabilise
            w = np.exp(arr)
            w /= w.sum()
            cand = {k: float(w[i]) for i, k in enumerate(keys)}
        for e, delta in (adjustments or {}).items():
            if e in cand:
                cand[e] += delta
        return sorted(cand.items(), key=lambda kv: kv[1], reverse=True)

    def _route_pending(
        self,
        pending: List[int],
        questions: List[str],
        q_emb: List[np.ndarray],
        session: SessionState,
        state: Dict,
    ) -> Dict[int, str]:
        """Route the pending questions: classifier decides, reasoner breaks ties.

        Returns ``{question_index: chosen_expert}`` and, as a side effect, appends
        to each question's ``history``/``route_traces`` and marks terminal
        ``REJECTED`` for questions with no viable expert on their *first* attempt.
        The Qwen-3B reasoner is loaded only when at least one question is ambiguous
        (top1-top2 score gap < ``tie_margin``), so confident questions never pay
        for it.
        """
        scores_list, backend = self._score_experts(
            [questions[i] for i in pending], [q_emb[i] for i in pending]
        )
        floor = self._router_prob_floor if backend == "logits" else self._router_cosine_floor

        assignment: Dict[int, str] = {}
        tiebreak: List[int] = []
        tb_shortlist: Dict[int, List[str]] = {}
        tb_adj: Dict[int, Dict[str, float]] = {}
        for pos, i in enumerate(pending):
            adjustments = session.routing_adjustments(q_emb[i])
            tried = set(state["history"][i])
            ranked = self._rank(scores_list[pos], backend, adjustments, exclude=tried)
            if not ranked or ranked[0][1] < floor:
                # No viable expert. Terminal rejection only if none was ever chosen;
                # otherwise the reroute budget is simply exhausted (classified later).
                top = f"{ranked[0][0]}={ranked[0][1]:.3f}" if ranked else "none"
                state["route_traces"][i].append(f"[router] no expert above floor (best {top}); rejected.")
                if not state["history"][i]:
                    state["status"][i] = REJECTED
                    state["answer"][i] = self._rejection_message
                continue

            top_e, top_s = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if len(ranked) > 1 and (top_s - second) < self._router_tie_margin:
                tiebreak.append(i)
                tb_shortlist[i] = [e for e, _ in ranked[: self._top_k]]
                tb_adj[i] = adjustments
            else:
                assignment[i] = top_e
                state["history"][i].append(top_e)
                state["route_traces"][i].append(
                    f"[router] classifier chose {top_e} (score={top_s:.3f}, margin={top_s - second:.3f})."
                )

        if tiebreak:
            # Ambiguous questions: let the Qwen-3B reasoner pick among the classifier's
            # top candidates. If it declines (NONE) or picks off-list, keep the
            # classifier's top-1 rather than wasting the attempt.
            self._reasoner.load()
            results = self._reasoner.route_batch(
                [questions[i] for i in tiebreak],
                [tb_shortlist[i] for i in tiebreak],
                self._descriptions, max_experts=1,
                adjustments_list=[tb_adj[i] for i in tiebreak],
            )
            self._reasoner.unload()
            for pos, i in enumerate(tiebreak):
                trace, chosen = results[pos]
                pick = chosen[0] if chosen else tb_shortlist[i][0]
                assignment[i] = pick
                state["history"][i].append(pick)
                state["route_traces"][i].append(f"[router] tiebreak among {tb_shortlist[i]} -> {pick}\n{trace}")
        # ``assignment`` insertion order is confident picks (in ``pending`` order)
        # then tiebreak picks — this is the order _run_rounds groups by, and the
        # sharded path must reconstruct it globally, so return the tiebreak set.
        return assignment, tiebreak

    # =============================================================== batch
    def ask(self, file: str) -> None:
        validate_file_exists(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_dir = os.path.join(get_answers_root(self.experiment), self._output_label)
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, "slg.json")
        checkpoint_path = os.path.join(output_dir, "slg_checkpoint.json")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == len(data):
                logger.info("All %d questions already answered in %s; nothing to do.", len(data), output_path)
                self._clear_checkpoint(checkpoint_path)
                return
            logger.info("Existing slg.json is incomplete (%d/%d); recomputing run.", len(existing), len(data))

        questions = [item["question"] for item in data]

        # Embed every question once (Jina), then release it before loading the LLMs.
        # Embeddings are deterministic, so a resumed run recomputes them rather
        # than persisting them in the checkpoint.
        logger.info("Embedding %d questions for cosine routing...", len(questions))
        q_emb = [self._retriever.embed_query(q) for q in questions]

        # Round-granular resume: if a checkpoint from an interrupted run survives,
        # restore the answer state + online A/C session and continue from the next
        # unfinished round. Rounds are the online-update boundary, so this yields
        # the same final result as an uninterrupted run.
        session = SessionState(ablation=self.ablation)
        state, start_attempt, round_progress = self._load_checkpoint(checkpoint_path, session, len(data))
        state = self._run_rounds(
            questions, q_emb, session,
            state=state, start_attempt=start_attempt, checkpoint_path=checkpoint_path,
            round_progress=round_progress,
        )

        self._write_outputs(data, state, session)
        # Run finished and persisted — the checkpoint is no longer needed.
        self._clear_checkpoint(checkpoint_path)

    def _write_outputs(self, data: List[dict], state: Dict, session: SessionState) -> None:
        """Assemble slg.json (original order) + diagnostics + summary."""
        output_dir = os.path.join(get_answers_root(self.experiment), self._output_label)
        ensure_dir(output_dir)
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
        with open(os.path.join(output_dir, "slg.json"), "w", encoding="utf-8") as f:
            json.dump(answers_list, f, indent=4)
        self._write_diagnostics(state, session)
        self._log_summary(state)

    # ---------------------------------------------------- sharded (base only)
    def answer_shard(self, file: str, shard_index: int, num_shards: int) -> Dict:
        """Process one contiguous shard of ``file`` and return picklable partial
        results keyed by **global** question index.

        Only valid for an **order-independent** run: the ``base`` ablation, where
        competence (A) contributes no routing adjustment and abstention (C)
        accepts everything, so a question's answer never depends on the session
        state built by other questions. That makes the per-question answers
        identical whether the run is one stream or sharded across GPUs — only the
        inert A/C diagnostic logs differ in ordering. Do **not** shard any run
        with A or C active.
        """
        validate_file_exists(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        shard = _contiguous_shard(len(data), shard_index, num_shards)

        questions = [data[i]["question"] for i in shard]
        logger.info("Base shard %d/%d: %d question(s).", shard_index + 1, num_shards, len(questions))
        q_emb = [self._retriever.embed_query(q) for q in questions]
        session = SessionState(ablation=self.ablation)
        state = self._run_rounds(questions, q_emb, session)

        records = []
        for local, gi in enumerate(shard):
            item = data[gi]
            records.append({
                "chapter": item.get("chapter"),
                "title": item.get("title"),
                "question": item["question"],
                "experts": state["history"][local],
                "answer": state["answer"][local],
                "status": state["status"][local],
            })
        return {
            "indices": shard,
            "records": records,
            "history": state["history"],
            "route_traces": state["route_traces"],
            "critic_log": state["critic_log"],
            "competence_log": session.competence.log,
            "calibration": {
                "target_error": session.calibrator.target_error,
                "final_threshold": session.calibrator.threshold(),
                "coverage": session.calibrator.coverage(),
                "threshold_history": session.calibrator.threshold_history,
            },
        }

    @staticmethod
    def _new_state(n: int) -> Dict:
        return {
            "status": [PENDING] * n,
            "answer": [None] * n,
            "history": [[] for _ in range(n)],     # experts tried, in order
            "route_traces": [[] for _ in range(n)],
            "critic_log": [[] for _ in range(n)],
            "best_lowconf": [None] * n,            # (answer, confidence) passed-but-withheld
        }

    # ---------------------------------------- sharded round 1 (full runs)
    def answer_shard_round1(self, file: str, shard_index: int, num_shards: int) -> Dict:
        """Run **round 1 only** (route → answer → verify) over one contiguous shard,
        returning raw per-question results by **global** index *without* applying
        the online A/C state.

        Valid for a ``full`` run because round-1 routing is **A-independent**: on
        the first pass the competence model is empty, so every question routes
        from pure classifier scores (competence deltas are all 0), and each
        answer + critic verdict depends only on its own question — never on the
        session state. The parent (:meth:`finish_from_round1`) merges the shards,
        replays A and C in the single-stream **canonical order** to reproduce the
        exact statuses + learning curves, then runs the (small) reroute rounds
        sequentially. This parallelises the expensive first pass across GPUs
        without fragmenting the A/C novelty the way independent full shards would.
        """
        validate_file_exists(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        shard = _contiguous_shard(len(data), shard_index, num_shards)
        if not shard:
            # Empty shard (num_shards > #questions / tiny smoke runs): return early
            # so we never load the classifier/critic just to do nothing.
            logger.info("Full round-1 shard %d/%d: empty; skipping model loads.",
                        shard_index + 1, num_shards)
            return {"results": {}}
        questions = [data[i]["question"] for i in shard]
        logger.info("Full round-1 shard %d/%d: %d question(s).",
                    shard_index + 1, num_shards, len(questions))
        q_emb = [self._retriever.embed_query(q) for q in questions]

        # EMPTY session: round-1 routing must see no competence signal (matches the
        # single-stream first pass). We never call observe_verdict/accept here.
        session = SessionState(ablation=self.ablation)
        n = len(shard)
        state = self._new_state(n)

        # Route (A-independent) — may mark REJECTED for below-floor questions.
        assignment, tiebreak = self._route_pending(list(range(n)), questions, q_emb, session, state)
        tiebreak_set = set(tiebreak)

        # Answer, grouped by expert (same batching as _run_rounds).
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, expert in assignment.items():
            groups[expert].append(i)
        answered: Dict[int, str] = {}
        for expert, idxs in groups.items():
            for c in range(0, len(idxs), self._expert_batch):
                chunk = idxs[c:c + self._expert_batch]
                outs = self._runner.answer_batch(expert, [questions[i] for i in chunk])
                for i, out in zip(chunk, outs):
                    answered[i] = out

        # Verify (critic) — collect verdicts, do NOT apply A/C.
        order = [i for _, idxs in groups.items() for i in idxs]
        self._critic.load()
        verdicts: Dict[int, Verdict] = {}
        for c in range(0, len(order), self._reasoner_batch):
            chunk = order[c:c + self._reasoner_batch]
            vs = self._verifier.verify_batch([
                (questions[i], assignment[i], answered[i]) for i in chunk
            ])
            for i, v in zip(chunk, vs):
                verdicts[i] = v
        self._critic.unload()

        results: Dict[int, dict] = {}
        for local in range(n):
            gi = shard[local]
            if local in assignment:
                v = verdicts[local]
                results[gi] = {
                    "expert": assignment[local],
                    # ``tiebreak`` lets the parent reconstruct the single-stream
                    # A/C order (confident picks first, then tiebreaks).
                    "tiebreak": local in tiebreak_set,
                    "answer": answered[local],
                    "route_traces": state["route_traces"][local],
                    # RAW (unrounded) verdict fields — the parent replays A/C from
                    # these for exact-match state; to_dict() rounding is diagnostics-only.
                    "verdict": {
                        "passed": bool(v.passed),
                        "confidence": float(v.confidence),
                        "det_score": float(v.det_score),
                        "llm_confidence": float(v.llm_confidence),
                        "det_ok": bool(v.det_ok),
                        "checks": v.checks,
                        "critique": v.critique,
                    },
                }
            else:
                # Rejected in round 1 (classifier below floor) — A-independent.
                results[gi] = {
                    "expert": None,
                    "answer": state["answer"][local],
                    "route_traces": state["route_traces"][local],
                    "status": state["status"][local],
                }
        return {"results": results}

    def _checkpoint_path(self) -> str:
        return os.path.join(
            get_answers_root(self.experiment), self._output_label, "slg_checkpoint.json"
        )

    def finish_from_round1(self, file: str, round1: Dict[int, dict]) -> None:
        """Replay merged round-1 shard results into a single-stream state, then run
        the reroute rounds sequentially and write outputs. ``round1`` maps every
        global index to the raw result from :meth:`answer_shard_round1`.

        A and C are applied here in the **canonical order** the single-stream
        route/verify phase uses — confident picks (ascending index) then tiebreak
        picks (ascending index), grouped by expert — so the resulting
        competence/calibration state, statuses, and diagnostics match a one-GPU
        ``full`` run (bar batched-decode float noise, the same caveat as batching).

        If a reroute-tail checkpoint from an interrupted finish already exists it is
        resumed directly (round 1 is not re-sharded), so a crash in the sequential
        tail does not redo the expensive first pass.
        """
        checkpoint_path = self._checkpoint_path()

        # Resume an interrupted reroute tail without re-running round 1.
        probe = SessionState(ablation=self.ablation)
        with open(file, "r", encoding="utf-8") as f:
            n = len(json.load(f))
        state, start_attempt, round_progress = self._load_checkpoint(checkpoint_path, probe, n)
        if state is not None:
            logger.info("Resuming reroute tail from checkpoint (skipping round-1 shards).")
            self._finish_tail(file, state, probe, start_attempt, round_progress, checkpoint_path)
            return

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = SessionState(ablation=self.ablation)
        state = self._new_state(n)

        # Seed routing outcomes; rejected questions are already terminal.
        for i in range(n):
            r = round1[i]
            state["route_traces"][i] = list(r.get("route_traces", []))
            if r["expert"] is None:
                state["status"][i] = r.get("status", REJECTED)
                state["answer"][i] = r["answer"]
            else:
                state["history"][i] = [r["expert"]]

        # Reconstruct the single-stream ``assignment.items()`` order: confident
        # picks first (ascending index), then tiebreak picks (ascending index) —
        # _route_pending appends tiebreaks after all confident picks — then group
        # by expert (first-appearance order). This is exactly the order _run_rounds
        # replays A/C in, so the online state matches regardless of how shards split.
        assigned = [i for i in range(n) if round1[i]["expert"] is not None]
        confident = [i for i in assigned if not round1[i].get("tiebreak")]
        tiebreaked = [i for i in assigned if round1[i].get("tiebreak")]
        groups: Dict[str, List[int]] = defaultdict(list)
        for i in confident + tiebreaked:
            groups[round1[i]["expert"]].append(i)
        order = [i for _, idxs in groups.items() for i in idxs]

        questions = [item["question"] for item in data]
        q_emb = [self._retriever.embed_query(q) for q in questions]
        for i in order:
            r = round1[i]
            vd = r["verdict"]
            verdict = Verdict(
                passed=bool(vd["passed"]),
                confidence=float(vd["confidence"]),       # raw, unrounded
                critique=vd.get("critique", ""),
                checks=vd.get("checks", {}),
                det_score=float(vd.get("det_score", 1.0)),
                llm_confidence=float(vd.get("llm_confidence", 0.5)),
                det_ok=bool(vd.get("det_ok", True)),
            )
            expert = r["expert"]
            session.observe_verdict(expert, q_emb[i], verdict)
            # critic_log stores the rounded to_dict() — matches the single-stream log.
            state["critic_log"][i].append({"expert": expert, "answer": r["answer"], **verdict.to_dict()})
            if verdict.passed and session.accept(verdict.confidence):
                state["status"][i] = RESOLVED
                state["answer"][i] = r["answer"]
            elif verdict.passed:
                state["best_lowconf"][i] = (r["answer"], verdict.confidence)
            # else: stays PENDING → handled by the reroute rounds below.

        self._finish_tail(file, state, session, start_attempt=1,
                          round_progress=None, checkpoint_path=checkpoint_path)

    def _finish_tail(self, file, state, session, start_attempt, round_progress, checkpoint_path):
        """Run the reroute rounds (attempt >= 1) sequentially, then write outputs.

        Shared by a fresh finish (after the round-1 replay, ``start_attempt=1``)
        and by a checkpoint resume. Reroute rounds *do* depend on A (competence is
        now populated), exactly like a single stream."""
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [item["question"] for item in data]
        q_emb = [self._retriever.embed_query(q) for q in questions]
        state = self._run_rounds(
            questions, q_emb, session, state=state, start_attempt=start_attempt,
            checkpoint_path=checkpoint_path, round_progress=round_progress,
        )
        self._write_outputs(data, state, session)
        self._clear_checkpoint(checkpoint_path)

    def _run_rounds(
        self,
        questions: List[str],
        q_emb: List[np.ndarray],
        session: SessionState,
        state: Optional[Dict] = None,
        start_attempt: int = 0,
        checkpoint_path: Optional[str] = None,
        round_progress: Optional[Dict] = None,
    ) -> Dict:
        n = len(questions)
        if state is None:
            state = self._new_state(n)

        for attempt in range(start_attempt, self._max_reroutes):
            pending = [i for i in range(n) if state["status"][i] == PENDING]
            if not pending:
                break
            logger.info("=== Routing round %d/%d — %d pending question(s) ===",
                        attempt + 1, self._max_reroutes, len(pending))

            # Resume the *in-progress* round from an intra-round checkpoint: its
            # route phase already ran (state reflects it), so restore the expert
            # assignment plus any partial answer/verify progress and skip ahead.
            resuming = round_progress is not None and round_progress.get("attempt") == attempt
            if resuming:
                assignment = {int(i): e for i, e in round_progress["assignment"]}
                answered = {int(i): a for i, a in round_progress["answers"]}
                verified = {int(i) for i in round_progress["verified"]}
            else:
                # --- Route phase (classifier decides; reasoner breaks ties) ---
                # Competence adjustments are read from the round-start session
                # state (untouched until the verify phase), so every pending
                # question sees exactly what it would in a one-at-a-time run. The
                # classifier scores the whole round in one batched forward pass and
                # the Qwen-3B reasoner is loaded only for the ambiguous subset.
                assignment, _ = self._route_pending(pending, questions, q_emb, session, state)
                answered, verified = {}, set()
                # Checkpoint immediately after routing so a crash in the (long)
                # answer phase resumes here instead of re-routing the round.
                round_progress = {"attempt": attempt,
                                  "assignment": [[i, e] for i, e in assignment.items()],
                                  "answers": [], "verified": []}
                self._ckpt(checkpoint_path, state, session, attempt, n, round_progress)

            # Canonical answer/verify order: grouped by expert, in assignment
            # order. Both phases walk this order and checkpoint per chunk, so an
            # interrupted round-1 pass over the full test set resumes near where
            # it stopped. The order is what fixes the A/C update sequence, so it
            # is recomputed identically on resume from the restored assignment.
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, expert in assignment.items():
                groups[expert].append(i)
            order = [i for _, idxs in groups.items() for i in idxs]

            # --- Answer phase (1B adapters, grouped by expert, chunked) ---
            for expert, idxs in groups.items():
                todo = [i for i in idxs if i not in answered]
                for c in range(0, len(todo), self._expert_batch):
                    chunk = todo[c:c + self._expert_batch]
                    outs = self._runner.answer_batch(expert, [questions[i] for i in chunk])
                    for i, out in zip(chunk, outs):
                        answered[i] = out
                    round_progress["answers"] = [[i, answered[i]] for i in order if i in answered]
                    self._ckpt(checkpoint_path, state, session, attempt, n, round_progress)

            # --- Verify phase (8B critic: domain verifier B + competence A + calibration C) ---
            # Verdicts are decoded a batch at a time and applied to the online
            # state in the *canonical order* above. Because A (competence) and C
            # (calibration) update sequentially, preserving that order — across a
            # resume too, since verified questions are a prefix of it — keeps the
            # learned state and the results consistent with a single-GPU run.
            self._critic.load()
            to_verify = [i for i in order if i not in verified]
            for c in range(0, len(to_verify), self._reasoner_batch):
                chunk = to_verify[c:c + self._reasoner_batch]
                verdicts = self._verifier.verify_batch([
                    (questions[i], assignment[i], answered[i]) for i in chunk
                ])
                for i, verdict in zip(chunk, verdicts):
                    expert = assignment[i]
                    session.observe_verdict(expert, q_emb[i], verdict)
                    entry = {"expert": expert, "answer": answered[i], **verdict.to_dict()}
                    state["critic_log"][i].append(entry)

                    if verdict.passed and session.accept(verdict.confidence):
                        state["status"][i] = RESOLVED
                        state["answer"][i] = answered[i]
                    elif verdict.passed:
                        # Passed the critic but below the calibrated confidence bar:
                        # keep the best such answer in case every attempt is withheld.
                        best = state["best_lowconf"][i]
                        if best is None or verdict.confidence > best[1]:
                            state["best_lowconf"][i] = (answered[i], verdict.confidence)
                        logger.info("Q%d expert '%s' passed but below confidence bar (%.2f).",
                                    i + 1, expert, verdict.confidence)
                    else:
                        logger.info("Verifier FAIL on Q%d expert '%s' (confidence=%.2f).",
                                    i + 1, expert, verdict.confidence)
                    verified.add(i)
                round_progress["verified"] = [i for i in order if i in verified]
                self._ckpt(checkpoint_path, state, session, attempt, n, round_progress)
            self._critic.unload()

            # Round boundary: this round's answers and A/C updates are all
            # committed. Clear the intra-round progress and checkpoint the
            # boundary (completed_attempts advances, so the reroute budget holds).
            round_progress = None
            self._ckpt(checkpoint_path, state, session, attempt + 1, n, None)

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

    # -------------------------------------------------------- checkpointing
    _CHECKPOINT_VERSION = 2

    def _ckpt(
        self, path: Optional[str], state: Dict, session: SessionState,
        completed_attempts: int, n: int, round_progress: Optional[Dict],
    ) -> None:
        """No-op when checkpointing is disabled (``path`` is None), else save."""
        if path is not None:
            self._save_checkpoint(path, state, session, completed_attempts, n, round_progress)

    def _save_checkpoint(
        self, path: str, state: Dict, session: SessionState, completed_attempts: int, n: int,
        round_progress: Optional[Dict] = None,
    ) -> None:
        """Atomically persist a resume snapshot.

        Written both at each round boundary and *within* a round after every
        answer/verify chunk. Captures the per-question answer state, the number
        of rounds fully finished (so the reroute budget is not reset on resume),
        the online A/C session state, and — for an interrupted round —
        ``round_progress``: the expert assignment plus which questions have been
        answered/verified so far this round, so a long round-1 pass over the full
        test set resumes near where it stopped instead of restarting.
        """
        payload = {
            "version": self._CHECKPOINT_VERSION,
            "label": self._output_label,      # guard: only resume the same run
            "n": n,                           # guard: only resume the same question set
            "completed_attempts": completed_attempts,
            "state": state,
            "session": session.state_dict(),
            "round_progress": round_progress,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the checkpoint

    def _load_checkpoint(
        self, path: str, session: SessionState, n: int
    ) -> Tuple[Optional[Dict], int, Optional[Dict]]:
        """Restore a checkpoint into ``session``; return (state, start_attempt, round_progress).

        Returns ``(None, 0, None)`` — a fresh start — when no valid, matching
        checkpoint exists. A checkpoint is only honoured if it targets the same
        run label, question count, and schema version; otherwise it is ignored
        (the stale file is left in place and overwritten by the first round of
        the new run). ``round_progress`` is non-None only when the run was
        interrupted *mid-round*, and drives the in-round resume.
        """
        if not os.path.exists(path):
            return None, 0, None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ignoring unreadable checkpoint %s (%s); restarting run.", path, e)
            return None, 0, None

        if (
            payload.get("version") != self._CHECKPOINT_VERSION
            or payload.get("label") != self._output_label
            or payload.get("n") != n
        ):
            logger.warning(
                "Checkpoint %s does not match this run (label/size/version); restarting run.",
                path,
            )
            return None, 0, None

        session.load_state_dict(payload["session"])
        start_attempt = int(payload["completed_attempts"])
        state = payload["state"]
        round_progress = payload.get("round_progress")
        remaining = sum(1 for s in state["status"] if s == PENDING)
        in_round = ""
        if round_progress is not None:
            done = len(round_progress.get("verified", []))
            total = len(round_progress.get("assignment", []))
            in_round = f", mid-round {start_attempt + 1} ({done}/{total} verified)"
        logger.info(
            "Resuming SLG run from checkpoint: %d/%d round(s) done, %d question(s) still pending%s.",
            start_attempt, self._max_reroutes, remaining, in_round,
        )
        return state, start_attempt, round_progress

    def _clear_checkpoint(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

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
        tried: set = set()           # every expert already chosen this turn

        for attempt in range(self._max_reroutes):
            adjustments = session.routing_adjustments(q_emb)
            # Classifier scores the question; exclude experts already tried this
            # turn so a reroute proposes genuine alternatives, never a repeat.
            scores_list, backend = self._score_experts([question], [q_emb])
            floor = self._router_prob_floor if backend == "logits" else self._router_cosine_floor
            ranked = self._rank(scores_list[0], backend, adjustments, exclude=tried)
            if not ranked or ranked[0][1] < floor:
                output_fn("\n[router] no suitable expert for this question.")
                break

            top_e, top_s = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            ambiguous = len(ranked) > 1 and (top_s - second) < self._router_tie_margin
            if multi:
                # Interactive multi-expert: the top expert plus any others the
                # classifier scores highly (question likely spans domains).
                chosen = [top_e] + [e for e, s in ranked[1:] if s >= self._router_multi_threshold]
            elif ambiguous:
                # Close call: let the Qwen-3B reasoner decide among the top candidates.
                shortlist = [e for e, _ in ranked[: self._top_k]]
                self._reasoner.load()
                trace, picked = self._reasoner.route(
                    question, shortlist, self._descriptions, 1,
                    adjustments=adjustments, carried_context=session.carried_context,
                )
                self._reasoner.unload()
                output_fn(f"\n[router reasoning]\n{trace}")
                chosen = picked or [top_e]
            else:
                chosen = [top_e]
            output_fn(f"\n[router chose] {', '.join(chosen)} (top score {top_s:.3f})")

            ever_chose = True
            tried.update(chosen)

            # Answer each chosen expert (carrying compressed context).
            answers = {
                e: self._runner.answer(e, question, session.carried_context) for e in chosen
            }

            # Verify each answer; surface verdict; update competence + calibration.
            self._critic.load()
            for expert in chosen:
                verdict = self._verifier.verify(question, expert, answers[expert])
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
            self._critic.unload()

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
