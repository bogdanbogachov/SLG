"""GPU-free unit test of the classifier-router logic in pipeline._route_pending.

Builds a bare SmallLanguageRouter via __new__ (no models loaded) and drives
_route_pending with fake classifier/reasoner/session to check:
  1. confident pick (no reasoner load), 2. tie -> reasoner tiebreak,
  3. reject below floor, 4. reroute excludes already-tried experts,
  5. cosine fallback when classifier unavailable.

Run: PYTHONPATH=. OPENAI_API_KEY=dummy CUDA_VISIBLE_DEVICES="" python3 tests/test_router.py
(also collectable by pytest: `PYTHONPATH=. pytest tests/test_router.py`)
"""
import numpy as np
from inference.slg.pipeline import SmallLanguageRouter, REJECTED, PENDING


class FakeClassifier:
    def __init__(self, rows, available=True):
        self.available = available
        self.labels = ["a", "b", "c"]
        self._rows = rows            # list of dict{expert: LOGIT} per question
        self.loaded = 0
    def load(self): self.loaded += 1; return self
    def unload(self): pass
    def logits_batch(self, questions): return self._rows


class FakeRetriever:
    def __init__(self, rows): self._rows = rows; self.i = 0
    def scores(self, emb):
        r = self._rows[self.i]; self.i += 1; return r


class FakeReasoner:
    def __init__(self, picks): self.picks = picks; self.loads = 0
    def load(self): self.loads += 1; return self
    def unload(self): pass
    def route_batch(self, questions, shortlists, descriptions, max_experts, adjustments_list=None):
        # pick per position from self.picks (list of chosen-lists)
        return [("trace", self.picks[k]) for k in range(len(questions))]


class FakeSession:
    def __init__(self, adj=None): self._adj = adj or {}
    def routing_adjustments(self, emb): return dict(self._adj)


def make_router(classifier, reasoner, retriever=None, cosine_floor=0.0):
    r = SmallLanguageRouter.__new__(SmallLanguageRouter)
    r._router_prob_floor = 0.10
    r._router_cosine_floor = cosine_floor
    r._router_tie_margin = 0.15
    r._top_k = 5
    r._descriptions = {"a": "", "b": "", "c": ""}
    r._rejection_message = "REJECTED_MSG"
    r._classifier = classifier
    r._reasoner = reasoner
    r._retriever = retriever
    return r


def fresh_state(n, history=None):
    return {
        "status": [PENDING] * n,
        "answer": [None] * n,
        "history": history or [[] for _ in range(n)],
        "route_traces": [[] for _ in range(n)],
        "critic_log": [[] for _ in range(n)],
        "best_lowconf": [None] * n,
    }


def test_confident():
    clf = FakeClassifier([{"a": 5.0, "b": 0.0, "c": 0.0}])   # softmax(a) ~= 0.99
    rea = FakeReasoner([])
    r = make_router(clf, rea)
    st = fresh_state(1)
    a, _ = r._route_pending([0], ["q"], [None], FakeSession(), st)
    assert a == {0: "a"}, a
    assert rea.loads == 0, "reasoner must not load for a confident pick"
    assert st["history"][0] == ["a"]
    print("1 confident pick: OK (no reasoner load)")


def test_tie():
    clf = FakeClassifier([{"a": 1.0, "b": 0.9, "c": -5.0}])  # softmax a~0.52 b~0.47 -> gap<0.15
    rea = FakeReasoner([["b"]])            # reasoner resolves tie -> b
    r = make_router(clf, rea)
    st = fresh_state(1)
    a, _ = r._route_pending([0], ["q"], [None], FakeSession(), st)
    assert a == {0: "b"}, a
    assert rea.loads == 1, "reasoner should load for the tie"
    print("2 tie -> reasoner tiebreak: OK")


def test_tie_reasoner_declines():
    clf = FakeClassifier([{"a": 1.0, "b": 0.9, "c": -5.0}])
    rea = FakeReasoner([[]])               # reasoner NONE -> keep classifier top-1
    r = make_router(clf, rea)
    st = fresh_state(1)
    a, _ = r._route_pending([0], ["q"], [None], FakeSession(), st)
    assert a == {0: "a"}, a
    print("3 tie + reasoner declines -> classifier top-1: OK")


def test_reject_cosine_floor():
    # Fallback (no classifier): raw cosines all below the cosine floor -> REJECTED.
    clf = FakeClassifier([], available=False)
    rea = FakeReasoner([])
    ret = FakeRetriever([{"a": 0.06, "b": 0.05, "c": 0.04}])
    r = make_router(clf, rea, retriever=ret, cosine_floor=0.10)
    st = fresh_state(1)
    a, _ = r._route_pending([0], ["q"], [np.zeros(4, dtype="float32")], FakeSession(), st)
    assert a == {} and st["status"][0] == REJECTED, (a, st["status"][0])
    print("4 cosine below floor -> REJECTED: OK")


def test_reroute_excludes_tried():
    # 'a' already tried; must not be re-picked even though it scores highest. After
    # excluding it, softmax renormalizes over {b,c} so b is a strong, valid pick
    # (the post-softmax-masking bug would have left b with a tiny stranded prob).
    clf = FakeClassifier([{"a": 5.0, "b": 2.0, "c": -5.0}])
    rea = FakeReasoner([])
    r = make_router(clf, rea)
    st = fresh_state(1, history=[["a"]])
    a, _ = r._route_pending([0], ["q"], [None], FakeSession(), st)
    assert a == {0: "b"}, a
    assert st["history"][0] == ["a", "b"]
    print("5 reroute excludes tried; renormalized pick survives floor: OK")


def test_competence_adjustment():
    # near-tie logits, but competence boosts 'b' over 'a' past the tie margin.
    clf = FakeClassifier([{"a": 1.0, "b": 0.9, "c": -5.0}])
    rea = FakeReasoner([])
    r = make_router(clf, rea)
    st = fresh_state(1)
    sess = FakeSession(adj={"b": 0.3, "a": -0.2})
    a, _ = r._route_pending([0], ["q"], [None], sess, st)
    assert a == {0: "b"}, a
    assert rea.loads == 0
    print("6 competence adjustment flips + widens margin: OK")


def test_fallback_cosine():
    clf = FakeClassifier([], available=False)
    rea = FakeReasoner([])
    ret = FakeRetriever([{"a": 0.8, "b": 0.1, "c": 0.05}])
    r = make_router(clf, rea, retriever=ret)   # default cosine_floor 0.0
    st = fresh_state(1)
    a, _ = r._route_pending([0], ["q"], [np.zeros(4, dtype="float32")], FakeSession(), st)
    assert a == {0: "a"}, a
    print("7 classifier-unavailable -> cosine fallback: OK")


if __name__ == "__main__":
    test_confident()
    test_tie()
    test_tie_reasoner_declines()
    test_reject_cosine_floor()
    test_reroute_excludes_tried()
    test_competence_adjustment()
    test_fallback_cosine()
    print("\nALL ROUTER TESTS PASSED")
