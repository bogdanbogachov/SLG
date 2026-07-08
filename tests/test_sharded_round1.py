"""Equivalence test for the round-1-sharded `full` path.

Asserts that sharding round 1 across N shards, then replaying A/C in canonical
order and running the reroute rounds (SmallLanguageRouter.answer_shard_round1 +
finish_from_round1), reproduces the single-stream _run_rounds result exactly —
status, answers, history, critic log, and both the competence (A) and calibrator
(C) state. GPU-free (fake models).

Run: PYTHONPATH=. OPENAI_API_KEY=dummy CUDA_VISIBLE_DEVICES="" python3 tests/test_sharded_round1.py
"""
import copy, json, os, tempfile
import numpy as np

from config import CONFIG
from inference.slg.pipeline import SmallLanguageRouter, RESOLVED
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
        out = []
        for q in questions:
            i = int(q[1:])
            base = {e: -5.0 for e in EXPERTS}
            top = EXPERTS[i % 3]
            second = EXPERTS[(i + 1) % 3]
            if i % 4 == 0:            # near-tie -> routed via the reasoner tiebreaker
                base[top] = 1.0; base[second] = 0.9
            else:                     # confident
                base[top] = 5.0; base[second] = 2.0
            out.append(base)
        return out


class FakeReasoner:
    def load(self): return self
    def unload(self): pass
    def route_batch(self, questions, shortlists, descriptions, max_experts, adjustments_list=None):
        # deterministic: keep the classifier's top candidate
        return [("tb", [shortlists[k][0]]) for k in range(len(questions))]


class FakeRunner:
    def answer_batch(self, expert, questions, carried_context=""):
        return [f"ans[{expert}]::{q}" for q in questions]


class FakeVerdict:
    # confidence == llm_confidence here (det_score 1.0); to_dict rounds like the
    # real Verdict.to_dict so the single-stream critic_log matches the replayed one.
    def __init__(self, passed, confidence):
        self.passed = passed; self.confidence = confidence
        self.det_score = 1.0; self.llm_confidence = confidence
        self.checks = {}; self.critique = "x"
    def to_dict(self):
        return {"passed": self.passed, "confidence": round(self.confidence, 4),
                "det_score": round(self.det_score, 4),
                "llm_confidence": round(self.llm_confidence, 4), "checks": {}, "critique": "x"}


class FakeVerifier:
    def verify_batch(self, items):
        out = []
        for (q, expert, desc, ans) in items:
            i = int(q[1:])
            fail = (i % 5 == 0 and expert == "a")
            # Non-round-clean confidences: a regression that stored the rounded
            # to_dict() value for A/C replay would diverge from single-stream here.
            out.append(FakeVerdict(passed=not fail, confidence=0.234567 if fail else 0.876543))
        return out


def make_router(output_label="t"):
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
    r.experiment = "t"
    r._output_label = output_label
    r.ablation = AblationConfig()
    r._classifier = FakeClassifier()
    r._reasoner = FakeReasoner()
    r._critic = FakeReasoner()
    r._runner = FakeRunner()
    r._verifier = FakeVerifier()
    # embed_query only needs to be deterministic; competence regions use cosine but
    # zeros are fine for an equivalence check (both paths see the same vectors).
    class _Ret:
        def embed_query(self, text): return np.zeros(4, dtype="float32")
    r._retriever = _Ret()
    return r


def signature(state, session):
    return {
        "status": list(state["status"]),
        "answer": list(state["answer"]),
        "history": copy.deepcopy(state["history"]),
        "critic": copy.deepcopy(state["critic_log"]),
        "competence": session.competence.state_dict(),
        "calibrator": session.calibrator.state_dict(),
    }


def run_reference(data):
    r = make_router()
    session = SessionState(ablation=AblationConfig())
    q = [d["question"] for d in data]
    emb = [np.zeros(4, dtype="float32") for _ in range(len(data))]
    state = r._run_rounds(q, emb, session)
    return signature(state, session)


def run_sharded(qa_file, n_shards, tmp):
    # Shard round 1.
    round1 = {}
    for k in range(n_shards):
        res = make_router().answer_shard_round1(qa_file, k, n_shards)
        round1.update(res["results"])
    assert len(round1) == N, len(round1)
    # Finish (replay A/C + reroutes) on one router; capture state+session instead
    # of writing files.
    fin = make_router()
    captured = {}
    fin._write_outputs = lambda data, state, session: captured.update(state=state, session=session)
    CONFIG["paths"]["answers"] = tmp                    # keep checkpoint writes in tmp
    os.makedirs(os.path.join(tmp, "t", "t"), exist_ok=True)
    fin.finish_from_round1(qa_file, round1)
    return signature(captured["state"], captured["session"])


if __name__ == "__main__":
    data = [{"question": f"q{i}", "title": "T", "chapter": "C"} for i in range(N)]
    ref = run_reference(data)
    assert ref["status"].count(RESOLVED) == N, ("reference not all resolved", set(ref["status"]))
    print(f"reference: {N}/{N} resolved")

    with tempfile.TemporaryDirectory() as tmp:
        qa_file = os.path.join(tmp, "qa.json")
        with open(qa_file, "w") as f:
            json.dump(data, f)
        for n_shards in (2, 3, 4, 8, 30):   # 30 > N exercises empty shards
            sig = run_sharded(qa_file, n_shards, tmp)
            assert sig == ref, f"sharded ({n_shards}) diverged from single-stream"
            print(f"sharded round-1 with {n_shards} shards == single-stream: OK")

    print("\nALL SHARDED-ROUND1 EQUIVALENCE TESTS PASSED")
