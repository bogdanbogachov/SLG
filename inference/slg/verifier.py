"""(B) Domain-grounded answer verification.

Generic LLM self-critique (Reflexion-style) checks fluency and coherence. For
engineering QA that is not enough: a wrong number or an answer whose quantities
carry no units is worse than a fluent-but-empty paragraph. This verifier pairs
two complementary signals:

* **Deterministic, domain-grounded checks** (no model, no GPU): numeric sanity,
  presence of units on quantities when the question is quantitative, format
  adherence, and degenerate/contradictory output. These are cheap, explainable,
  and act as a *hard constraint*: a veto forces FAIL and zeroes the confidence.
* **The 8B critic's engineering-aware judgement**, whose ``P(PASS)`` supplies the
  confidence score (see :meth:`Reasoner.criticize`).

The two are combined into a single structured :class:`Verdict` — a pass/fail
decision plus a calibrated-ready confidence in ``[0, 1]`` and a per-check
breakdown.

The two halves play *different* roles rather than being averaged. The critic
scores; the rules gate. Blending them (the old ``sqrt(llm_conf * det_score)``)
was unsound: ``det_score`` is a rubric fraction, not a probability, so the
geometric mean degenerated to ``sqrt(llm_conf)`` — pushing a critic's 0.4 up to
0.63 and compressing every score into a narrow band the calibrator could not
separate.

The rules also supply the **self-supervised label** ``det_ok`` that the
abstention calibrator (C) conditions on. That split is what makes C well-posed.
Calibrating the critic's ``P(PASS)`` against the critic's own PASS/FAIL made the
score and the label the same random variable thresholded at 0.5 — every
candidate threshold above 0.5 had exactly zero empirical error, so ``tau``
always collapsed to (at most) the lowest passing confidence and no answer could
ever be abstained on. ``det_ok`` is produced by rules the critic never sees, so
the calibration set carries real signal.

The checks are deliberately domain-general (no aerospace- or dataset-specific
rules) so the verifier transfers to any engineering corpus. The lexical bounds
below are set from the reference-answer distribution of the training corpus
(42,579 answers of >= 60 words): MATTR-50 has p0.1 = 0.35 and p99.9 = 0.92, and
the band [0.30, 0.95] excludes 0.058% of real answers.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import CONFIG

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
# A run of six or more identical punctuation marks ("....................").
_PUNCTUATION_RUN = re.compile(r"([.\-_!?*=~])\1{5,}")
# Word tokens, for the lexical-diversity measure.
_WORD = re.compile(r"[a-zA-Z']+")
# An answer that ends on a sentence terminator, closing bracket/quote, or fence.
_TERMINATED = re.compile(r"(?:```|[.!?;:)\]}\"'`>*_])\s*$")

# Absurd magnitude guard, applied only to numbers carrying a unit — i.e. to
# *quantities*, not to identifiers. Applied to every number the regex finds, it
# vetoed 0.37% of real corpus answers on binary and hex literals ("2147483651...",
# "111011001001010100") and float-mantissa fragments, which say nothing about
# engineering plausibility. Scoped to unit-adjacent numbers the false-veto rate on
# the corpus is 0.067%.
_ABSURD = 1e12

# A refusal phrase only means the answer is a non-answer when it *is* the answer.
# Real corpus answers say "I don't know of a cleaner way, but ..." and then answer;
# none shorter than this contains a refusal phrase at all, so the veto is gated on
# length rather than on the phrase alone (which vetoed 2.3% of real answers).
_MAX_WORDS_FOR_NON_ANSWER = 60

# --- lexical-diversity guard -------------------------------------------------
# Raw type-token ratio falls with length (Heaps' law), so a flat bound would be a
# length test in disguise. MATTR (moving-average TTR over a fixed window) is
# length-invariant by construction. Bounds from the training corpus, see module
# docstring. Both tails are degeneracy: below the floor the model is looping,
# above the ceiling it is refusing to reuse any wording and drifts into synonym
# cascades ("...endlessly indefinitely infinitely ad infinitum...").
_MATTR_WINDOW = 50
_MATTR_MIN = 0.30
_MATTR_MAX = 0.95
# Below this the window statistic is too noisy to judge; the check is skipped.
_MIN_WORDS_FOR_LEXICAL = 60

# The completeness check only applies to answers long enough that hitting the
# decoder's token budget is a plausible explanation for an unterminated ending.
# ~0.75 words per token, at 80% of the budget. Below this length an unterminated
# answer is more likely to be a real answer that just ends on a URL or code.
_TRUNCATION_WORDS = int(0.8 * 0.75 * int(CONFIG["generation"]["max_new_tokens"]))


def _mattr(words: List[str], window: int = _MATTR_WINDOW) -> float:
    """Moving-average type-token ratio: mean distinct-word fraction per window."""
    if len(words) < window:
        return len(set(words)) / max(1, len(words))
    windows = len(words) - window + 1
    return sum(len(set(words[i:i + window])) / window for i in range(windows)) / windows


@dataclass
class Verdict:
    """Structured outcome of verifying a single expert answer."""

    passed: bool
    confidence: float                 # in [0, 1]; what the accept/abstain gate compares to tau
    critique: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    det_score: float = 1.0            # fraction of applicable deterministic checks passed
    llm_confidence: float = 0.5       # critic's P(PASS) from its verdict-token logits
    det_ok: bool = True               # every applicable rule held — the calibrator's label

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "det_score": round(self.det_score, 4),
            "llm_confidence": round(self.llm_confidence, 4),
            "det_ok": self.det_ok,
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
        words = _WORD.findall(text.lower())
        checks: Dict[str, bool] = {}
        veto = False

        # 1. Non-empty, non-degenerate answer. A degenerate answer (empty, or a
        #    short refusal like "I don't know" / "as an AI") is a hard veto: it
        #    forces a FAIL regardless of the critic, per the verifier spec.
        refusal = bool(_NON_ANSWER.search(text)) and len(words) < _MAX_WORDS_FOR_NON_ANSWER
        non_degenerate = bool(text) and not refusal
        checks["non_degenerate"] = non_degenerate
        if not non_degenerate:
            veto = True

        # 2. Lexical plausibility (hard veto). Catches the two failure modes of a
        #    small model decoding a long answer: repetition loops (MATTR far below
        #    the corpus floor) and synonym cascades (MATTR above anything a human
        #    writes). This is the check that makes "degenerate" mean something
        #    beyond an "I don't know" regex.
        if len(words) >= _MIN_WORDS_FOR_LEXICAL:
            mattr = _mattr(words)
            lexically_plausible = _MATTR_MIN <= mattr <= _MATTR_MAX
            checks["lexically_plausible"] = lexically_plausible
            if not lexically_plausible:
                veto = True
        # else: too short for a stable window statistic — not applicable.

        # 3. Numeric sanity: every *quantity* (a number carrying a unit) is finite
        #    and of a plausible magnitude. Bare numbers are skipped — in this corpus
        #    they are as often binary literals or array indices as measurements.
        quantities: List[float] = []
        for match in _NUMBER_WITH_UNIT.finditer(text):
            num = _NUMBER.match(match.group(0))
            if num is None:
                continue
            try:
                quantities.append(float(num.group(0).replace(",", ".")))
            except ValueError:
                continue
        numeric_sane = all(np_isfinite(x) and abs(x) < _ABSURD for x in quantities)
        checks["numeric_sane"] = numeric_sane
        if quantities and not numeric_sane:
            veto = True

        # 4. No filler punctuation runs ("hope my rambling helps somewhat......").
        checks["no_punctuation_run"] = not _PUNCTUATION_RUN.search(text)

        # 5. Completeness: a long answer that stops without a terminator has run
        #    into the decoder's token budget mid-sentence.
        if len(words) >= _TRUNCATION_WORDS:
            checks["complete"] = bool(_TERMINATED.search(text))
        # else: not applicable — a short unterminated answer is usually fine.

        # 6. Units present when the question is quantitative.
        if self._require_units and _QUANTITATIVE_Q.search(question or ""):
            has_quantity = bool(_NUMBER_WITH_UNIT.search(text))
            checks["units_present"] = has_quantity
        # else: not applicable — omitted from the score.

        # 7. Format adherence: enumerated questions get structured answers.
        if _LIST_Q.search(question or ""):
            structured = bool(re.search(r"(\n\s*[-*\d]|;|,\s)", text))
            checks["format_adherence"] = structured

        applicable = [v for v in checks.values()]
        det_score = (sum(1 for v in applicable if v) / len(applicable)) if applicable else 1.0
        return det_score, veto, checks

    # ----------------------------------------------------------- verify
    def verify(self, question: str, expert_id: str, answer: str) -> Verdict:
        return self.verify_batch([(question, expert_id, answer)])[0]

    # ------------------------------------------------------- verify (batched)
    def verify_batch(self, items: List[Tuple[str, str, str]]) -> List[Verdict]:
        """Verify a batch of answers, decoding all critic verdicts together.

        ``items`` are ``(question, expert_id, answer)`` tuples. The deterministic
        layer is pure CPU and stays per-item; only the 8B critic is batched. Each
        answer is judged independently, so the produced verdicts match the
        per-item path (bar the batched-decode float noise). The caller is
        responsible for applying the verdicts to the online
        competence/calibration state in question order.

        ``expert_id`` is carried for logging only; the critic never sees which
        expert produced the answer.
        """
        if self._deterministic_enabled:
            det = [self._deterministic(q, a) for (q, _e, a) in items]
        else:
            det = [(1.0, False, {}) for _ in items]

        crit = self._reasoner.criticize_batch(items)

        verdicts: List[Verdict] = []
        for (det_score, veto, checks), (llm_passed, p_pass, critique) in zip(det, crit):
            # The critic scores, the rules gate: confidence is the critic's P(PASS),
            # and a deterministic veto forces FAIL and zeroes it outright.
            passed = bool(llm_passed and not veto)
            confidence = 0.0 if veto else float(p_pass)
            # The calibrator's label: did every applicable rule hold? Independent of
            # `confidence` by construction — the critic never sees the checks. With
            # the deterministic layer ablated (-B) there is no such label, so it is
            # constant True and C degrades to its confidence floor: that collapse is
            # the measured cost of removing (B), not a bug.
            det_ok = (not veto) and det_score >= 1.0
            verdicts.append(Verdict(
                passed=passed,
                confidence=confidence,
                critique=critique,
                checks=checks,
                det_score=det_score,
                llm_confidence=p_pass,
                det_ok=det_ok,
            ))
        return verdicts


def np_isfinite(x: float) -> bool:
    """Local finite check that avoids importing numpy for one predicate."""
    return x == x and x not in (float("inf"), float("-inf"))
