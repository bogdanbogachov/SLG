"""(C) Calibrated abstention.

The system must know when *not* to answer — a wrong engineering answer is worse
than an honest "I can't answer this reliably."

We treat the verifier confidence as a nonconformity score and the critic
verdict as a *self-supervised* label: no ground truth, no cloud oracle, nothing
leaves the machine (consistent with the on-prem constraint). From the stream of
``(confidence, passed)`` observations accumulated during the run/session, a
split-conformal-style calibrator maintains a confidence threshold ``tau`` such
that, among answers it would accept (confidence >= tau), the empirical fraction
that the critic rejected stays at or below a target error rate. Answers below
``tau`` are withheld and the system abstains.

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
    # (confidence, passed) observations — the self-supervised calibration set.
    _obs: List[Tuple[float, bool]] = field(default_factory=list)
    # (n_observations, threshold) after each update, for diagnostics.
    threshold_history: List[Tuple[int, float]] = field(default_factory=list)

    def observe(self, confidence: float, passed: bool) -> None:
        self._obs.append((float(confidence), bool(passed)))
        self.threshold_history.append((len(self._obs), self.threshold()))

    def threshold(self) -> float:
        """Smallest confidence whose acceptance set keeps error <= target_error.

        Scanning candidate thresholds from high to low grows the acceptance set
        (confidence >= tau) and therefore coverage; we take the lowest tau that
        still satisfies the error budget, maximising coverage. With too little
        data we cannot trust the estimate, so we fall back to the floor.
        """
        if len(self._obs) < self.min_calibration:
            return self.confidence_floor

        candidates = sorted({c for c, _ in self._obs}, reverse=True)
        best = None
        for tau in candidates:
            accepted = [passed for c, passed in self._obs if c >= tau]
            if not accepted:
                continue
            error = sum(1 for passed in accepted if not passed) / len(accepted)
            if error <= self.target_error:
                best = tau  # keep lowering tau while the budget holds
            else:
                break  # lowering further only adds more failures
        # If even the strictest threshold violates the budget, abstain widely by
        # demanding more than any observed confidence.
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
