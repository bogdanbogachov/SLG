"""(A) Online expert-competence estimation.

A router that *learns which expert to trust* from the verifier signal — with no
labels and no retraining.

The earlier design only ever *penalised* an expert after a critic rejection.
This generalises that one-sided heuristic into a two-sided, online competence
model. Every critic verdict (pass **or** fail) updates a Beta-Bernoulli
posterior over the expert's reliability in the *region* of query space around
the question. The router then shifts the cosine ranking by how far each
expert's estimated reliability sits from the uninformed prior — boosting
experts that have proven competent on similar questions and demoting those that
have failed.

Estimation is *local*: every observation is attached to a region keyed by the
question embedding, so competence learned on one question only transfers to
cosine-similar questions. A single expert can therefore be reliable in one part
of the query space and unreliable in another.

Lifetime matches the pipeline invocation (the whole automated run, or one open
chat session). The ordered observation log is also the data behind the paper's
"routing accuracy improves online" figure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Beta(1, 1) — uninformed prior. Posterior mean of a brand-new expert is 0.5,
# i.e. no boost and no penalty until evidence accumulates.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1.0
_PRIOR_MEAN = _PRIOR_ALPHA / (_PRIOR_ALPHA + _PRIOR_BETA)


@dataclass
class _Region:
    """Competence evidence for one expert in one neighbourhood of query space."""

    expert_id: str
    centroid: np.ndarray  # L2-normalized question embedding
    passes: int = 0
    fails: int = 0

    @property
    def alpha(self) -> float:
        return _PRIOR_ALPHA + self.passes

    @property
    def beta(self) -> float:
        return _PRIOR_BETA + self.fails

    @property
    def n(self) -> int:
        return self.passes + self.fails

    @property
    def mean(self) -> float:
        """Posterior mean reliability (used for the routing adjustment)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def lower_bound(self) -> float:
        """Conservative (lower-confidence) reliability estimate.

        Posterior mean minus one standard deviation of the Beta posterior. Used
        as a secondary, evidence-aware signal for abstention: an expert with one
        lucky pass should not look as reliable as one with ten.
        """
        a, b = self.alpha, self.beta
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        return max(0.0, self.mean - float(np.sqrt(var)))


@dataclass
class CompetenceModel:
    """Online, label-free estimate of per-expert reliability by query region."""

    threshold: float = 0.85
    weight: float = 0.3
    _regions: List[_Region] = field(default_factory=list)
    # Ordered (step, expert, region_size_before, reliability_before, passed)
    # records — the learning signal exported to diagnostics.
    log: List[dict] = field(default_factory=list)

    def _match(self, expert_id: str, q_emb: np.ndarray) -> Optional[_Region]:
        for region in self._regions:
            if region.expert_id == expert_id and self._similar(region.centroid, q_emb):
                return region
        return None

    def _similar(self, a: np.ndarray, b: np.ndarray) -> bool:
        return float(np.dot(a, b)) >= self.threshold

    # ------------------------------------------------------------- update
    def observe(self, expert_id: str, q_emb: np.ndarray, passed: bool) -> float:
        """Fold one critic verdict into the model. Returns reliability after."""
        region = self._match(expert_id, q_emb)
        reliability_before = region.mean if region is not None else _PRIOR_MEAN
        n_before = region.n if region is not None else 0
        if region is None:
            region = _Region(expert_id=expert_id, centroid=np.asarray(q_emb, dtype="float32"))
            self._regions.append(region)
        if passed:
            region.passes += 1
        else:
            region.fails += 1
        self.log.append(
            {
                "step": len(self.log),
                "expert": expert_id,
                "n_before": n_before,
                "reliability_before": round(reliability_before, 4),
                "reliability_after": round(region.mean, 4),
                "passed": bool(passed),
            }
        )
        return region.mean

    # -------------------------------------------------------------- query
    def reliability(self, expert_id: str, q_emb: np.ndarray) -> Tuple[float, float, int]:
        """Return (posterior_mean, lower_bound, n_observations) for the region."""
        region = self._match(expert_id, q_emb)
        if region is None:
            return _PRIOR_MEAN, _PRIOR_MEAN, 0
        return region.mean, region.lower_bound, region.n

    def adjustments(self, q_emb: np.ndarray) -> Dict[str, float]:
        """Signed cosine adjustments for the experts seen near this question.

        ``delta = weight * (reliability - prior)`` — positive for experts that
        have over-performed the prior on similar questions, negative for those
        that have under-performed. Experts with no local evidence get 0 (their
        ranking is left to raw cosine similarity).
        """
        deltas: Dict[str, float] = {}
        for region in self._regions:
            if self._similar(region.centroid, q_emb):
                delta = self.weight * (region.mean - _PRIOR_MEAN)
                deltas[region.expert_id] = deltas.get(region.expert_id, 0.0) + delta
        return deltas

    # ----------------------------------------------------- (de)serialization
    def state_dict(self) -> dict:
        """JSON-serializable snapshot for mid-run checkpoint/resume."""
        return {
            "threshold": self.threshold,
            "weight": self.weight,
            "regions": [
                {
                    "expert_id": r.expert_id,
                    "centroid": r.centroid.tolist(),
                    "passes": r.passes,
                    "fails": r.fails,
                }
                for r in self._regions
            ],
            "log": self.log,
        }

    def load_state_dict(self, d: dict) -> None:
        self.threshold = float(d["threshold"])
        self.weight = float(d["weight"])
        self._regions = [
            _Region(
                expert_id=r["expert_id"],
                centroid=np.asarray(r["centroid"], dtype="float32"),
                passes=int(r["passes"]),
                fails=int(r["fails"]),
            )
            for r in d["regions"]
        ]
        self.log = list(d["log"])
