"""(B) Domain-grounded answer verification.

Generic LLM self-critique (Reflexion-style) checks fluency and coherence. For
engineering QA that is not enough: a wrong number or an answer whose quantities
carry no units is worse than a fluent-but-empty paragraph. This verifier pairs
two complementary signals:

* **Deterministic, domain-grounded checks** (no model, no GPU): numeric sanity,
  presence of units on quantities when the question is quantitative, format
  adherence, and degenerate/contradictory output. These are cheap, explainable,
  and can *veto* a pass outright (e.g. non-finite numbers).
* **The 8B critic's engineering-aware judgement** and its self-reported
  confidence (see :meth:`Reasoner.criticize`).

The two are combined into a single structured :class:`Verdict` — a pass/fail
decision plus a calibrated-ready confidence in ``[0, 1]`` and a per-check
breakdown. The confidence is what the abstention calibrator consumes.

The checks are deliberately domain-general (no aerospace- or dataset-specific
rules) so the verifier transfers to any engineering corpus.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# A number, optionally signed / decimal / exponential.
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
# A number immediately followed by a unit-like token (mm, kg, N, m/s, %, °C ...).
_NUMBER_WITH_UNIT = re.compile(
    r"[-+]?\d+(?:[.,]\d+)?\s*"
    r"(?:%|°[CF]?|[a-zA-Zµ]+(?:\s*[/·]\s*[a-zA-Z]+)?(?:\^?\d)?)"
)
# Heuristic: the question asks for a quantity / measurement / calculation.
_QUANTITATIVE_Q = re.compile(
    r"\b(how (?:many|much|long|wide|thick|far|fast|heavy)|what (?:is|are) the "
    r"(?:value|size|length|width|diameter|thickness|mass|weight|force|pressure|"
    r"temperature|tolerance|number|amount|quantity)|calculate|compute|"
    r"dimension|tolerance|diameter|how big)\b",
    re.IGNORECASE,
)
# Heuristic: the question asks for an enumerated / structured answer.
_LIST_Q = re.compile(r"\b(list|steps|procedure|enumerate|stages|sequence|what are the)\b", re.IGNORECASE)
# Degenerate / non-answers.
_NON_ANSWER = re.compile(
    r"\b(i (?:do not|don't|cannot|can't) (?:know|answer|help)|no information|"
    r"unable to (?:answer|determine)|as an ai)\b",
    re.IGNORECASE,
)

# Absurd magnitude guard for the numeric-sanity check (unit-agnostic).
_ABSURD = 1e12


@dataclass
class Verdict:
    """Structured outcome of verifying a single expert answer."""

    passed: bool
    confidence: float                 # in [0, 1], consumed by the calibrator
    critique: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    det_score: float = 1.0            # deterministic-check score in [0, 1]
    llm_confidence: float = 0.5       # critic self-reported confidence

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "det_score": round(self.det_score, 4),
            "llm_confidence": round(self.llm_confidence, 4),
            "checks": self.checks,
            "critique": self.critique,
        }


class DomainVerifier:
    """Combine deterministic engineering checks with the LLM critic."""

    def __init__(self, reasoner, require_units: bool = True, deterministic: bool = True):
        self._reasoner = reasoner
        self._require_units = require_units
        # When False (the -B ablation) the deterministic engineering layer is
        # skipped and verification falls back to the generic 8B critic alone.
        self._deterministic_enabled = deterministic

    # ---------------------------------------------------- deterministic
    def _deterministic(self, question: str, answer: str) -> Tuple[float, bool, Dict[str, bool]]:
        """Return (det_score in [0,1], hard_veto, per-check booleans).

        A ``hard_veto`` forces a FAIL regardless of the LLM critic — reserved for
        unambiguous defects (empty/degenerate answer, non-finite numbers).
        """
        text = (answer or "").strip()
        checks: Dict[str, bool] = {}
        veto = False

        # 1. Non-empty, non-degenerate answer.
        non_degenerate = bool(text) and not _NON_ANSWER.search(text)
        checks["non_degenerate"] = non_degenerate
        if not text:
            veto = True

        # 2. Numeric sanity: every parsed number is finite and not absurd.
        numbers: List[float] = []
        for tok in _NUMBER.findall(text):
            try:
                numbers.append(float(tok.replace(",", ".")))
            except ValueError:
                continue
        numeric_sane = all(np_isfinite(x) and abs(x) < _ABSURD for x in numbers)
        checks["numeric_sane"] = numeric_sane
        if numbers and not numeric_sane:
            veto = True

        # 3. Units present when the question is quantitative.
        if self._require_units and _QUANTITATIVE_Q.search(question or ""):
            has_quantity = bool(_NUMBER_WITH_UNIT.search(text))
            checks["units_present"] = has_quantity
        # else: not applicable — omitted from the score.

        # 4. Format adherence: enumerated questions get structured answers.
        if _LIST_Q.search(question or ""):
            structured = bool(re.search(r"(\n\s*[-*\d]|;|,\s)", text))
            checks["format_adherence"] = structured

        applicable = [v for v in checks.values()]
        det_score = (sum(1 for v in applicable if v) / len(applicable)) if applicable else 1.0
        return det_score, veto, checks

    # ----------------------------------------------------------- verify
    def verify(self, question: str, expert_id: str, description: str, answer: str) -> Verdict:
        if self._deterministic_enabled:
            det_score, veto, checks = self._deterministic(question, answer)
        else:
            det_score, veto, checks = 1.0, False, {}
        llm_passed, llm_conf, critique = self._reasoner.criticize(
            question, expert_id, description, answer
        )

        passed = bool(llm_passed and not veto)
        # Confidence blends the critic's self-report with the deterministic
        # evidence (geometric mean keeps it conservative — a low factor on either
        # side drags the result down). A hard veto zeroes it.
        confidence = 0.0 if veto else float((max(llm_conf, 1e-6) * max(det_score, 1e-6)) ** 0.5)

        return Verdict(
            passed=passed,
            confidence=confidence,
            critique=critique,
            checks=checks,
            det_score=det_score,
            llm_confidence=llm_conf,
        )


def np_isfinite(x: float) -> bool:
    """Local finite check that avoids importing numpy for one predicate."""
    return x == x and x not in (float("inf"), float("-inf"))
