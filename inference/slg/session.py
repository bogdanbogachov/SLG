"""Session-scoped state: escalating expert penalties + carried chat context.

A penalty is recorded whenever the critic rejects an expert's answer. Each
penalty is keyed by the *question embedding*, so a rejected expert is avoided
not only while rerouting the current question but also for any later question
in the same session that is cosine-similar to it. Re-failing the same expert on
a similar question escalates its penalty (the count grows), pushing it further
down the cosine ranking.

Lifetime is a single pipeline invocation: the whole automated run in batch mode,
or one open conversation in interactive chat mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import CONFIG


@dataclass
class _PenaltyEntry:
    expert_id: str
    question_embedding: np.ndarray  # L2-normalized
    count: int = 1


@dataclass
class SessionState:
    """Penalty memory and compressed carried context for one session/run."""

    _entries: List[_PenaltyEntry] = field(default_factory=list)
    carried_context: str = ""

    def __post_init__(self):
        routing = CONFIG["routing"]
        self._threshold = float(routing.get("penalty_similarity_threshold", 0.85))
        self._weight = float(routing.get("penalty_weight", 0.2))

    def penalize(self, expert_id: str, question_embedding: np.ndarray) -> int:
        """Record/escalate a penalty for ``expert_id`` on this question.

        Returns the new accumulated count for the matched (expert, question).
        """
        for entry in self._entries:
            if entry.expert_id == expert_id and self._similar(
                entry.question_embedding, question_embedding
            ):
                entry.count += 1
                return entry.count
        self._entries.append(_PenaltyEntry(expert_id, question_embedding, count=1))
        return 1

    def penalty_weights(self, question_embedding: np.ndarray) -> Dict[str, float]:
        """Down-weight per expert for a question (count * configured weight)."""
        counts: Dict[str, int] = {}
        for entry in self._entries:
            if self._similar(entry.question_embedding, question_embedding):
                counts[entry.expert_id] = counts.get(entry.expert_id, 0) + entry.count
        return {eid: c * self._weight for eid, c in counts.items()}

    def _similar(self, a: np.ndarray, b: np.ndarray) -> bool:
        return float(np.dot(a, b)) >= self._threshold

    # ----------------------------------------------------------- context
    def set_context(self, text: str) -> None:
        self.carried_context = text or ""

    def reset_penalties(self) -> None:
        """Drop all penalties (used to isolate questions when desired)."""
        self._entries.clear()
