"""Regression tests for critic decoding: guards, and the verdict-token readout.

Two independent failures are covered.

1. `no_repeat_ngram_size` bans n-grams found anywhere in the sequence, *prompt
   included*. Every Reasoner role must reproduce tokens its prompt dictates — the
   router echoes expert names, and the critic once had to echo "VERDICT: PASS".
   With the expert-answer guard on, those completions are masked to -inf and the
   role cannot say what it was asked to say. Reasoner._decode_guards() must
   therefore disable both guards.

2. The critic's confidence is now read from the next-token distribution over
   " PASS" / " FAIL" rather than parsed out of generated text. choice_probs must
   read the *last real* position under left padding and renormalise over exactly
   those two tokens.

GPU-free. Run:
  PYTHONPATH=. OPENAI_API_KEY=dummy CUDA_VISIBLE_DEVICES="" python3 tests/test_critic_decode_guards.py
"""
import torch
from transformers import AutoTokenizer, NoRepeatNGramLogitsProcessor

from config import CONFIG
from inference.slg import generation
from inference.slg.reasoner import Reasoner, _VERDICT_CHOICES

TOKENIZER_PATH = "downloaded_models/downloaded_qwen2_5_3b"

# The n-gram size the trap is demonstrated at, independent of what config.yaml
# currently sets for expert answers.
_TRAP_NGRAM = 3

# Stands in for any structured-output prompt that dictates the model's own tokens
# (the critic's old verdict lines; the router's "Expert: <name>" last line).
_DICTATED_PROMPT = (
    "Reply with VERDICT: PASS if the answer is acceptable, "
    "or VERDICT: FAIL if it is not.\n"
    "The answer looks sound.\nVERDICT:"
)


def dictated_token_scores(tok, ngram_size):
    """Score the tokens the prompt told the model to emit, under a given guard."""
    ids = tok(_DICTATED_PROMPT, return_tensors="pt").input_ids
    scores = torch.zeros((1, len(tok)), dtype=torch.float)
    if ngram_size > 0:
        scores = NoRepeatNGramLogitsProcessor(ngram_size)(ids, scores)
    return [scores[0, tok(c, add_special_tokens=False).input_ids[0]].item()
            for c in _VERDICT_CHOICES]


class FakeModel:
    """Returns a fixed next-token logit vector, keyed by the last real token id.

    Lets us assert choice_probs reads the correct position under left padding: if
    it read a pad position instead, the scores would come from the wrong row.
    """
    def __init__(self, vocab, pass_id, fail_id):
        self.vocab, self.pass_id, self.fail_id = vocab, pass_id, fail_id

    def __call__(self, input_ids, attention_mask=None, **kw):
        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab), -10.0)
        for i in range(b):
            # Row i's final position votes PASS iff its last real token is even.
            favours_pass = int(input_ids[i, -1]) % 2 == 0
            logits[i, -1, self.pass_id] = 2.0 if favours_pass else 0.0
            logits[i, -1, self.fail_id] = 0.0 if favours_pass else 2.0
        return type("Out", (), {"logits": logits})()


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    # --- 1. the guard trap -------------------------------------------------
    # Demonstrated at a fixed n rather than the configured one: the expert-answer
    # guards are currently off (see config.yaml), but Reasoner must disable them
    # unconditionally so that re-enabling them can never silence a structured role.
    p, f = dictated_token_scores(tok, _TRAP_NGRAM)
    assert p == float("-inf") and f == float("-inf"), (
        f"expected dictated tokens banned under no_repeat_ngram_size={_TRAP_NGRAM}, got {p}, {f}"
    )
    print(f"no_repeat_ngram_size={_TRAP_NGRAM}: dictated tokens {_VERDICT_CHOICES} -> -inf  <- the trap")

    rep_penalty, ngram_size = Reasoner._decode_guards()
    p, f = dictated_token_scores(tok, ngram_size)
    assert p > float("-inf") and f > float("-inf"), (
        f"Reasoner roles still cannot emit dictated tokens (ngram={ngram_size})"
    )
    assert rep_penalty == 1.0, f"repetition_penalty must not skew these roles, got {rep_penalty}"
    print(f"Reasoner guards (rep={rep_penalty}, ngram={ngram_size}): reachable  <- fixed")

    # generate() must read the expert-answer settings from config, and let an
    # explicit override (what Reasoner passes) win.
    assert generation._guards(None, None) == (
        float(CONFIG["generation"]["repetition_penalty"]),
        int(CONFIG["generation"]["no_repeat_ngram_size"]),
    ), "expert answers no longer honour the configured guards"
    assert generation._guards(1.0, 0) == (1.0, 0), "explicit guards must override config"
    print("expert-answer defaults read from config; explicit overrides honoured")

    # --- 2. the verdict-token readout --------------------------------------
    pass_id = tok(_VERDICT_CHOICES[0], add_special_tokens=False).input_ids[0]
    fail_id = tok(_VERDICT_CHOICES[1], add_special_tokens=False).input_ids[0]
    assert pass_id != fail_id, "PASS/FAIL must differ in their first token"

    model = FakeModel(len(tok), pass_id, fail_id)

    def last_id(text):
        return tok(text).input_ids[-1]

    # A short prompt and a long one, ending on token ids of opposite parity, so a
    # reader that grabbed a pad position (or the wrong row) would flip a verdict.
    # The short prompt is the one that gets left-padded.
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    short = next(w for w in words if last_id(w) % 2 == 0)
    long_word = next(w for w in words if last_id(w) % 2 == 1)
    prompts = [short, " ".join([long_word] * 4)]
    assert last_id(prompts[0]) % 2 == 0 and last_id(prompts[1]) % 2 == 1

    probs = generation.choice_probs(
        prompts, list(_VERDICT_CHOICES), model, tok, batch_size=2, device="cpu"
    )

    assert len(probs) == 2 and all(abs(sum(p) - 1.0) < 1e-5 for p in probs), probs
    # Row 0 ends on an even token id -> PASS favoured; row 1 on odd -> FAIL favoured.
    assert probs[0][0] > 0.5 and probs[1][0] < 0.5, (
        f"choice_probs read the wrong position under left padding: {probs}"
    )
    print(f"choice_probs: P(PASS) = {probs[0][0]:.3f} (short, left-padded) / "
          f"{probs[1][0]:.3f} (long)  <- correct row, normalised over PASS/FAIL")

    print("\nCRITIC DECODE + VERDICT-READOUT TESTS PASSED")
