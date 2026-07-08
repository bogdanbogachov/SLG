"""End-to-end batch-path test with the new classifier router: full run +
checkpoint/resume equivalence, GPU-free (fake models).

Confirms _run_rounds works with _route_pending, produces multiple rounds
(reroute on failures), writes intra-round checkpoints, and that crashing after
every checkpoint and resuming reproduces the uninterrupted reference exactly.

Run: PYTHONPATH=. OPENAI_API_KEY=dummy CUDA_VISIBLE_DEVICES="" python3 tests/test_resume_router.py
"""
import copy, json, os, tempfile
import numpy as np
from inference.slg.pipeline import SmallLanguageRouter, RESOLVED, EXHAUSTED
from inference.slg.session import SessionState
from inference.slg.ablation import AblationConfig

EXPERTS = ["a", "b", "c"]
N = 25


class FakeClassifier:
    available = True
    labels = EXPERTS
    def load(self): return self
    def unload(self): pass
    def logits_batch(self, questions):
        # deterministic per-question LOGITS keyed by the question index encoded in
        # "q<idx>". Well-separated so attempt-1 is a confident pick and the reroute
        # (after excluding the failed expert) cleanly picks the runner-up.
        out = []
        for q in questions:
            i = int(q[1:])
            base = {e: -5.0 for e in EXPERTS}
            base[EXPERTS[i % 3]] = 5.0
            base[EXPERTS[(i + 1) % 3]] = 2.0
            out.append(base)
        return out


class FakeReasoner:
    def load(self): return self
    def unload(self): pass
    def route_batch(self, questions, shortlists, descriptions, max_experts, adjustments_list=None):
        return [("tiebreak-trace", [shortlists[k][0]]) for k in range(len(questions))]


class FakeRunner:
    def answer_batch(self, expert, questions, carried_context=""):
        return [f"ans[{expert}]::{q}" for q in questions]


class FakeVerdict:
    def __init__(self, passed, confidence):
        self.passed = passed; self.confidence = confidence
    def to_dict(self):
        return {"passed": self.passed, "confidence": round(self.confidence, 4),
                "det_score": 1.0, "llm_confidence": self.confidence, "checks": {}, "critique": "x"}


class FakeVerifier:
    def verify_batch(self, items):
        out = []
        for (q, expert, desc, ans) in items:
            i = int(q[1:])
            # fail on expert 'a' the first time it's tried for i%5==0 -> forces reroute
            fail = (i % 5 == 0 and expert == "a")
            out.append(FakeVerdict(passed=not fail, confidence=0.9 if not fail else 0.2))
        return out


def make_router(checkpoint_path=None):
    r = SmallLanguageRouter.__new__(SmallLanguageRouter)
    r._max_reroutes = 3
    r._expert_batch = 8
    r._reasoner_batch = 6
    r._top_k = 5
    r._router_prob_floor = 0.10
    r._router_cosine_floor = 0.0
    r._router_tie_margin = 0.15
    r._router_multi_threshold = 0.30
    r._descriptions = {e: "" for e in EXPERTS}
    r._rejection_message = "REJ"
    r._exhausted_message = "EXH"
    r._low_confidence_message = "LOW"
    r._output_label = "test"
    r._classifier = FakeClassifier()
    r._reasoner = FakeReasoner()
    r._critic = FakeReasoner()   # verify phase loads/unloads the critic model
    r._runner = FakeRunner()
    r._verifier = FakeVerifier()
    return r


def run_reference():
    r = make_router()
    session = SessionState(ablation=AblationConfig())
    q = [f"q{i}" for i in range(N)]
    emb = [np.zeros(4, dtype="float32") for _ in range(N)]
    state = r._run_rounds(q, emb, session)
    return state, session


def signature(state, session):
    return {
        "status": list(state["status"]),
        "answer": list(state["answer"]),
        "history": copy.deepcopy(state["history"]),
        "critic": copy.deepcopy(state["critic_log"]),
        "competence": session.competence.state_dict(),
        "calibrator": session.calibrator.state_dict(),
    }


def count_checkpoints(tmp):
    """Run once, checkpoint to disk, count how many distinct saves occur by
    wrapping _save_checkpoint."""
    r = make_router()
    ckpt = os.path.join(tmp, "ck.json")
    saves = {"n": 0}
    orig = r._save_checkpoint
    def wrapped(path, *a, **k):
        saves["n"] += 1
        return orig(path, *a, **k)
    r._save_checkpoint = wrapped
    session = SessionState(ablation=AblationConfig())
    q = [f"q{i}" for i in range(N)]; emb = [np.zeros(4, dtype="float32") for _ in range(N)]
    r._run_rounds(q, emb, session, checkpoint_path=ckpt)
    return saves["n"]


def resume_after_k_saves(tmp, k):
    """Run, but raise on the k-th checkpoint save (simulating a crash right after
    it is persisted), then resume from the saved checkpoint and finish."""
    ckpt = os.path.join(tmp, f"ck{k}.json")

    class Boom(Exception): pass

    r = make_router()
    saves = {"n": 0}
    orig = r._save_checkpoint
    def wrapped(path, *a, **kw):
        orig(path, *a, **kw)
        saves["n"] += 1
        if saves["n"] == k:
            raise Boom()
    r._save_checkpoint = wrapped
    session = SessionState(ablation=AblationConfig())
    q = [f"q{i}" for i in range(N)]; emb = [np.zeros(4, dtype="float32") for _ in range(N)]
    try:
        r._run_rounds(q, emb, session, checkpoint_path=ckpt)
        return None  # k beyond total saves; nothing to resume
    except Boom:
        pass

    # Resume: fresh router + session, load checkpoint, continue.
    r2 = make_router()
    session2 = SessionState(ablation=AblationConfig())
    state2, start_attempt, round_progress = r2._load_checkpoint(ckpt, session2, N)
    assert state2 is not None, f"checkpoint {k} did not load"
    state2 = r2._run_rounds(q, emb, session2, state=state2, start_attempt=start_attempt,
                            checkpoint_path=ckpt, round_progress=round_progress)
    return signature(state2, session2)


if __name__ == "__main__":
    ref_state, ref_session = run_reference()
    ref = signature(ref_state, ref_session)
    resolved = ref["status"].count(RESOLVED)
    print(f"reference: {resolved}/{N} resolved, statuses={set(ref['status'])}")
    assert resolved == N, "expected all resolved after reroutes"

    with tempfile.TemporaryDirectory() as tmp:
        total = count_checkpoints(tmp)
        print(f"checkpoints written across the run: {total}")
        assert total > 3, "expected intra-round checkpoints (more than round boundaries)"

        mismatches = 0
        for k in range(1, total + 1):
            sig = resume_after_k_saves(tmp, k)
            if sig is None:
                continue
            if sig != ref:
                mismatches += 1
                print(f"  MISMATCH resuming after checkpoint {k}")
        assert mismatches == 0, f"{mismatches} resume points diverged"
        print(f"resume-equivalence OK: all {total} crash points reproduce the reference exactly")

    print("\nALL RESUME+ROUTER TESTS PASSED")
