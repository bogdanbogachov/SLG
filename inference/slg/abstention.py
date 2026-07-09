"""(C) Calibrated abstention.

The system must know when *not* to answer — a wrong engineering answer is worse
than an honest "I can't answer this reliably."

We treat the critic's ``P(PASS)`` as a nonconformity score and the deterministic
verifier's ``det_ok`` as a *self-supervised* label: no ground truth, no cloud
oracle, nothing leaves the machine (consistent with the on-prem constraint).
From the stream of ``(score, label)`` observations accumulated during the
run/session, a split-conformal-style calibrator maintains a threshold ``tau``
such that, among answers it would accept (score >= tau), the empirical fraction
violating the engineering rules stays at or below a target error rate. Answers
below ``tau`` are withheld and the system abstains.

**The score and the label must come from different sources.** Labelling with the
critic's own PASS/FAIL — which is exactly ``P(PASS) >= 0.5`` — made the label a
deterministic function of the score: every candidate ``tau >= 0.5`` then had zero
empirical error, so the scan below always walked ``tau`` down to at most the
lowest passing score. Since the pipeline only consults ``accept`` on a verdict
that already passed, *every* answer it asked about cleared ``tau``: abstention was
unreachable for any data, and the ``ABSTAINED`` terminal state was dead code. The
rules the critic never sees are what break that circularity.

The calibration set grows online; until it is large enough to be trustworthy
(``min_calibration``) the calibrator falls back to a fixed confidence floor.
The threshold history is exported to diagnostics for the paper's reliability /
coverage analysis.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AbstentionCalibrator:
    """Confidence threshold that controls the accepted-answer error rate."""

    target_error: float = 0.10
    confidence_floor: float = 0.5
    min_calibration: int = 20
    # (score, label) observations — the self-supervised calibration set. `score` is
    # the critic's P(PASS); `label` is the deterministic verifier's det_ok.
    _obs: List[Tuple[float, bool]] = field(default_factory=list)
    # (n_observations, threshold) after each update, for diagnostics.
    threshold_history: List[Tuple[int, float]] = field(default_factory=list)

    def observe(self, score: float, label: bool) -> None:
        self._obs.append((float(score), bool(label)))
        self.threshold_history.append((len(self._obs), self.threshold()))

    def threshold(self) -> float:
        """Smallest score whose acceptance set keeps error <= target_error.

        Scanning candidate thresholds from high to low grows the acceptance set
        (score >= tau) and therefore coverage; we take the lowest tau that
        still satisfies the error budget, maximising coverage. The scan stops at
        the first violation rather than continuing to look for a lower tau that
        happens to satisfy the budget again: empirical error is not monotone in
        tau on a finite sample, and honouring a later dip would let a lucky run
        of low-score-but-valid answers drag the threshold into the tail.

        With too little data we cannot trust the estimate, so we fall back to the
        floor.
        """
        if len(self._obs) < self.min_calibration:
            return self.confidence_floor

        candidates = sorted({c for c, _ in self._obs}, reverse=True)
        best = None
        for tau in candidates:
            accepted = [label for c, label in self._obs if c >= tau]
            if not accepted:
                continue
            error = sum(1 for label in accepted if not label) / len(accepted)
            if error <= self.target_error:
                best = tau  # keep lowering tau while the budget holds
            else:
                break  # lowering further only adds more rule violations
        # If even the strictest threshold violates the budget, abstain widely by
        # demanding more than any observed score.
        if best is None:
            return min(1.0, max(candidates) + 1e-6)
        return best

    def accept(self, confidence: float) -> bool:
        """Whether an answer with this confidence clears the current threshold."""
        return float(confidence) >= self.threshold()

    def coverage(self) -> float:
        """Fraction of observed answers that would currently be accepted."""
        if not self._obs:
            return 0.0
        tau = self.threshold()
        return sum(1 for c, _ in self._obs if c >= tau) / len(self._obs)

    # ----------------------------------------------------- (de)serialization
    def state_dict(self) -> dict:
        """JSON-serializable snapshot for mid-run checkpoint/resume.

        Persists the full ``(score, label)`` calibration set — not just the
        derived threshold — so a resumed run re-derives an identical ``tau``.
        """
        return {
            "target_error": self.target_error,
            "confidence_floor": self.confidence_floor,
            "min_calibration": self.min_calibration,
            "obs": [[float(c), bool(p)] for c, p in self._obs],
            "threshold_history": [[int(n), float(t)] for n, t in self.threshold_history],
        }

    def load_state_dict(self, d: dict) -> None:
        self.target_error = float(d["target_error"])
        self.confidence_floor = float(d["confidence_floor"])
        self.min_calibration = int(d["min_calibration"])
        self._obs = [(float(c), bool(p)) for c, p in d["obs"]]
        self.threshold_history = [(int(n), float(t)) for n, t in d["threshold_history"]]
