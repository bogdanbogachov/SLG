"""Session-scoped state shared by every question in one run / conversation.

Bundles the three online mechanisms that learn over the lifetime of a single
pipeline invocation:

* **(A)** :class:`~inference.slg.competence.CompetenceModel` — the online,
  label-free estimate of which expert to trust per query region, updated from
  every critic verdict and used to adjust the cosine ranking.
* **(C)** :class:`~inference.slg.abstention.AbstentionCalibrator` — the
  self-supervised confidence threshold that decides when to answer vs. abstain.
* the compressed **carried context** threaded across chat turns.

Lifetime is a single invocation: the whole automated run in batch mode, or one
open conversation in interactive chat mode. Verdicts produced by the
domain verifier **(B)** are folded into both A and C through
:meth:`observe_verdict`.
"""

import numpy as np

from config import CONFIG

from inference.slg.ablation import AblationConfig
from inference.slg.abstention import AbstentionCalibrator
from inference.slg.competence import CompetenceModel
from inference.slg.verifier import Verdict


class SessionState:
    """Online competence + abstention calibration + carried chat context."""

    def __init__(self, carried_context: str = "", ablation: AblationConfig = None):
        self.ablation = ablation or AblationConfig()
        routing = CONFIG["routing"]
        threshold = float(
            routing.get(
                "competence_similarity_threshold",
                routing.get("penalty_similarity_threshold", 0.85),
            )
        )
        self.competence = CompetenceModel(
            threshold=threshold,
            weight=float(routing.get("competence_weight", 0.3)),
        )
        self.calibrator = AbstentionCalibrator(
            target_error=float(routing.get("abstention_target_error", 0.10)),
            confidence_floor=float(routing.get("abstention_confidence_floor", 0.5)),
            min_calibration=int(routing.get("abstention_min_calibration", 20)),
        )
        self.carried_context = carried_context

    # ------------------------------------------------------- observe (B->A,C)
    def observe_verdict(self, expert_id: str, q_emb: np.ndarray, verdict: Verdict) -> None:
        """Fold a verifier verdict into both the competence model and calibrator.

        The calibrator is fed the critic's *raw* ``llm_confidence`` as the score,
        never ``verdict.confidence``: the latter is zeroed on a deterministic veto,
        which would make the score a function of the label and reintroduce the
        circularity that :mod:`inference.slg.abstention` exists to avoid. The two
        agree on every answer that is not vetoed.
        """
        self.competence.observe(expert_id, q_emb, verdict.passed)
        self.calibrator.observe(verdict.llm_confidence, verdict.det_ok)

    # ----------------------------------------------------------- routing (A)
    def routing_adjustments(self, q_emb: np.ndarray):
        """Signed cosine adjustments for experts seen near this question.

        Returns no adjustments when competence (A) is ablated, leaving the
        ranking to raw cosine similarity. Verdicts are still observed so the
        competence log is comparable across conditions.
        """
        if not self.ablation.competence:
            return {}
        return self.competence.adjustments(q_emb)

    # -------------------------------------------------------- abstention (C)
    def accept(self, confidence: float) -> bool:
        """Whether an answer of this confidence clears the calibrated threshold.

        With abstention (C) ablated, every critic-passed answer is accepted.
        """
        if not self.ablation.abstention:
            return True
        return self.calibrator.accept(confidence)

    # ---------------------------------------------------------------- context
    def set_context(self, text: str) -> None:
        self.carried_context = text or ""

    # ----------------------------------------------------- (de)serialization
    def state_dict(self) -> dict:
        """JSON-serializable snapshot of the online A/C state for checkpoint/resume.

        The ablation config is *not* persisted — it is re-supplied by the caller
        (from the run's ablation preset) when a fresh ``SessionState`` is built
        before restore, so a checkpoint can only be resumed under the same preset.
        """
        return {
            "competence": self.competence.state_dict(),
            "calibrator": self.calibrator.state_dict(),
            "carried_context": self.carried_context,
        }

    def load_state_dict(self, d: dict) -> None:
        self.competence.load_state_dict(d["competence"])
        self.calibrator.load_state_dict(d["calibrator"])
        self.carried_context = d.get("carried_context", "")
