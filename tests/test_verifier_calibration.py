"""Regression tests for the (B) deterministic checks and the (C) calibrator.

Three failures observed on the `test_3` smoke run are covered.

1. The deterministic layer vetoed nothing: `non_degenerate` was an "I don't know"
   regex, so word-salad and truncated answers scored det_score 1.0 and (B) had no
   effect beyond the LLM critic. The lexical/punctuation/completeness checks must
   fire on those answers — and must stay quiet on real corpus answers.

2. The calibrator's score and label were the same random variable: `confidence`
   was the critic's P(PASS) and `passed` was `P(PASS) >= 0.5`. Any tau >= 0.5 then
   had zero empirical error, so tau never exceeded the lowest passing confidence
   and no passing answer was ever abstained on. `observe` must now take an
   independent label.

3. Consequently, `ABSTAINED` was unreachable. With an independent label a
   low-confidence answer must fall below tau.

GPU-free. Run:
  PYTHONPATH=. OPENAI_API_KEY=dummy CUDA_VISIBLE_DEVICES="" python3 tests/test_verifier_calibration.py
"""
import json
import os
import random

from inference.slg.abstention import AbstentionCalibrator
from inference.slg.verifier import DomainVerifier, _mattr

# Verbatim tails of the test_3 answers, each exhibiting one defect.
_WORD_SALAD = (
    "The bed heats up and the timer resets. " + " ".join(
        "endlessly unending eternal perpetual ongoing continuous uninterrupted seamless "
        "smooth flowing fluidic motionless static stationary immobile stagnant unmoving "
        "inert inactive dormant asleep slumbering hibernation suspended conserve preserve "
        "prolong maintain ensure guarantee assure confirm verify validate inspect examine "
        "evaluate assess measure quantify tally compute derive infer deduce ascertain "
        "establish prove disprove refute reject accept approve authorize grant permit "
        "license certify accredit endorse ratify sanction validate legitimize warrant".split()
    )
)
_PUNCT_RUN = "Use a wider trace for the higher current. Anyway, hope my rambling helps somewhat................."
_LOOPING = "The trim wheel is a backup. " + "It is a manual backup for the pitch trim. " * 40
_GOOD = (
    "The 737 keeps a mechanical trim wheel because its pitch trim is a jackscrew driven "
    "by an electric motor, and the wheel gives the crew a direct cable path to that "
    "jackscrew when the motor is cut out. Later Boeing types use fly-by-wire trim with "
    "redundant electrical channels, so a manual wheel would add mass without adding a "
    "failure path that is not already covered. The wheel survives on the 737 largely "
    "because the airframe is certified as a derivative of the 1967 design and the "
    "mechanical reversion path is part of that certification basis."
)


class _NullReasoner:
    """Stands in for the 8B critic; returns a fixed P(PASS) per item."""

    def __init__(self, p_pass=0.9):
        self.p_pass = p_pass

    def criticize_batch(self, items):
        return [(self.p_pass >= 0.5, self.p_pass, "stub") for _ in items]


def checks_for(answer, question="Why is this so?"):
    v = DomainVerifier(_NullReasoner())
    return v.verify(question, "expert", answer)


if __name__ == "__main__":
    # --- 1. the checks fire on the observed defects -------------------------
    salad = checks_for(_WORD_SALAD)
    assert salad.checks["lexically_plausible"] is False, salad.checks
    assert salad.passed is False and salad.confidence == 0.0, salad
    assert salad.det_ok is False
    print(f"word salad: MATTR={_mattr(_WORD_SALAD.lower().split()):.3f} -> vetoed")

    loop = checks_for(_LOOPING)
    assert loop.checks["lexically_plausible"] is False, loop.checks
    assert loop.passed is False, loop
    print(f"repetition loop: MATTR={_mattr(_LOOPING.lower().split()):.3f} -> vetoed")

    punct = checks_for(_PUNCT_RUN)
    assert punct.checks["no_punctuation_run"] is False, punct.checks
    # A soft check: it lowers det_score and flips the calibration label, but the
    # critic still decides pass/fail.
    assert punct.passed is True and punct.det_ok is False, punct
    assert punct.det_score < 1.0, punct
    print(f"punctuation run: det_score={punct.det_score:.2f}, det_ok=False, passed=True (soft)")

    good = checks_for(_GOOD)
    assert good.det_ok is True and good.passed is True, good
    assert good.det_score == 1.0, good
    print(f"real answer: MATTR={_mattr(_GOOD.lower().split()):.3f}, det_ok=True")

    # Truncation: a long answer stopping mid-sentence.
    truncated = _GOOD + " " + " ".join(["the jackscrew carries the load through"] * 60) + " and then"
    assert checks_for(truncated).checks["complete"] is False
    print("truncated long answer: complete=False")

    # --- 1b. false-positive rate on the real corpus -------------------------
    qa_path = "question_answer/qa_train.json"
    if os.path.exists(qa_path):
        corpus = json.load(open(qa_path))
        sample = random.Random(0).sample(corpus, min(400, len(corpus)))
        vetoed = sum(1 for r in sample if not checks_for(r["answer"], r["question"]).passed)
        rate = vetoed / len(sample)
        assert rate <= 0.02, f"deterministic layer vetoes {rate:.1%} of real answers"
        print(f"corpus false-veto rate: {vetoed}/{len(sample)} = {rate:.2%}")
    else:
        print(f"(skipped corpus false-positive check: no {qa_path})")

    # --- 2. score and label are independent ---------------------------------
    # The old bug, reproduced: label == (score >= 0.5) makes every tau >= 0.5
    # error-free, so the scan always walks tau down to at most the lowest passing
    # score. Every answer the critic passed therefore cleared tau, whatever the
    # data — the pipeline only calls accept() on a passing verdict, so abstention
    # could never trigger. (Coverage over *all* observations can still be < 1: the
    # failures below tau are simply never offered to accept().)
    scores = [i / 40 for i in range(40)]
    tautological = AbstentionCalibrator(target_error=0.10, min_calibration=20)
    for s in scores:
        tautological.observe(s, s >= 0.5)
    lowest_passing = min(s for s in scores if s >= 0.5)
    tau_taut = tautological.threshold()
    assert tau_taut <= lowest_passing, tau_taut
    assert all(tautological.accept(s) for s in scores if s >= 0.5), (
        "under tautological labels every passing answer must be accepted"
    )
    print(f"tautological labels: tau={tau_taut:.3f} <= lowest passing score "
          f"{lowest_passing:.3f}; no passing answer can be abstained on  <- the bug")

    # With an independent label, high scores are reliable and low scores are not,
    # so tau must rise above the floor and coverage must drop below 1.
    cal = AbstentionCalibrator(target_error=0.10, min_calibration=20)
    rng = random.Random(1)
    for _ in range(200):
        score = rng.random()
        # P(rules hold) grows with the critic's confidence, but is not determined by it.
        label = rng.random() < score ** 2
        cal.observe(score, label)
    tau = cal.threshold()
    assert tau > 0.5, f"tau should exceed the floor once low scores are shown to be wrong: {tau}"
    assert cal.coverage() < 1.0, f"coverage should drop below 1: {cal.coverage()}"
    accepted = [lab for s, lab in cal._obs if s >= tau]
    err = sum(1 for lab in accepted if not lab) / len(accepted)
    assert err <= 0.10 + 1e-9, f"accepted-set error {err:.3f} exceeds the target"
    print(f"independent labels: tau={tau:.3f}, coverage={cal.coverage():.2f}, accepted error={err:.3f}")

    # --- 3. abstention is now reachable -------------------------------------
    assert not cal.accept(0.55), "a low-confidence answer must be abstained on"
    assert cal.accept(0.99), "a high-confidence answer must still be returned"
    print("abstention reachable: accept(0.55)=False, accept(0.99)=True")

    print("\nVERIFIER + CALIBRATION TESTS PASSED")
