# SLG — Small Language Router

A retrieval and inference pipeline for **trustworthy, privacy-constrained
engineering question answering on commodity hardware** (≤24 GB VRAM). Everything
runs on-premises — no cloud LLM is contacted at inference, so no question or
document ever leaves the machine.

A router directs each question to a topic-specific **retrieval index**; a
**frozen** small model answers *grounded in the retrieved passage*; a cheap
verifier checks the answer; a calibrated threshold decides whether to return it
or **abstain**.

**The problem this solves.** For a small firm that cannot use a cloud model, the
binding constraint is not *knowledge* (retrieval supplies that) but **trust**:
they need assurance an answer is correct, and honesty when it is not. Plain RAG
cannot provide this — retrieval always returns a top-k result, but the similarity
score is **not calibrated to correctness**. A high cosine means "closest
document," never "this answers the question," and there is no native "nothing
matches" outcome. So a bare retrieve-and-return system is *silently* wrong on
out-of-scope questions, semantic near-misses, and superseded documents.

Three online mechanisms address that, all label-free and fully on-prem:

- **(A) Online competence-learning router** — learns *which topic index to trust*
  from its own verifier signal, with no labels and no retraining. Every verdict
  updates a per-index, per-query-region reliability estimate that adjusts the
  router's ranking. Routing improves over the lifetime of a run.
- **(B) Domain-grounded verifier** — a **1.4 GB** stack, not an 8B LLM judge:
  a cross-encoder reranker (question↔passage relevance), a small NLI consistency
  model (claim-level passage→answer support), semantic entropy over k samples
  (stability), and deterministic grounding rules. Each covers the others' blind
  spot — only the reranker catches a *wrong passage*; only entropy/consistency
  catch a *drifting answer*.
- **(C) Calibrated abstention** — a **conformal** threshold controlling the error
  rate among answers the system actually *returns* to a target ε, with a
  finite-sample guarantee. A wrong engineering answer is worse than an honest
  "I can't answer this reliably."

**No fine-tuning, no RL — all models frozen.** Domain knowledge enters through the
**context window** (retrieval), not the **weights**. Fine-tuning the 3B experts
damaged their generation (repetition loops: SFT overwrote the base model's
stopping behaviour while memorising a small, repetitive corpus). Not training
anything is also a deployability claim: a small firm with no ML team can run this.

## Architecture

Per query:

1. **Route** — classifier picks the topic index.
2. **(A)** competence delta adjusts the router score (online, from past verdicts).
3. **Retrieve** — top-5 passages from that index.
4. **Rerank** — cross-encoder rescores those 5.
5. **Generate** — frozen Qwen-3B, k=5 sampled answers, grounded in the top passage.
6. **(B) Verify** — three neural features + one deterministic label (table below).
7. **Ensemble** — logistic regression over the features → score `s`.
8. **(C) Abstain** — conformal threshold: `s >= tau` → answer, else abstain.
9. **Update** A and C; reroute to the next index on failure.

| Component | Model | VRAM | Emits |
|---|---|---|---|
| Router | Llama-3.2-1B classifier | <1 GB | topic-index score |
| Embedder | jina-v2-base-en | 0.5 GB | retrieval + top1−top2 margin *(feature)* |
| Reranker | bge-reranker-base | 0.6 GB | question↔passage relevance *(feature)* |
| Generator | Qwen2.5-3B-Instruct **(frozen)** | 6.0 GB | k grounded answers |
| Consistency | MiniCheck / DeBERTa-NLI | 0.8 GB | claim-level support, **min** over claims *(feature)* |
| Semantic entropy | *(reuses generator + NLI)* | 0 | stability over the k samples *(feature)* |
| Grounding rules | regex | 0 | **LABEL**: answer quantities/entities must appear in passage |
| Ensemble | logistic regression | 0 | score `s`, fit online on `(features, label)` |

**Verifier total ≈ 1.4 GB**, versus ~16 GB for an 8B LLM judge.

Two details carry the design:

- **Claim-level checking.** Split the answer into claims, check each against the
  passage, take the **minimum**. A whole-answer score averages a single
  fabricated step away; the weakest link surfaces it — and it lets the system
  show the engineer *which* sentence was unsupported.
- **Score and label must not share a model.** The score comes from the neural
  features, the label from the deterministic rules. Couple them and the label
  becomes a function of the score, the calibrator learns nothing, and abstention
  becomes unreachable.

## Requirements

- Python 3.10+
- CUDA-compatible GPU, ≤24 GB VRAM target (Qwen-3B generator ≈6 GB; verifier
  stack ≈1.4 GB; 1B router classifier and Jina embedder are lighter)
- Optional environment variables: `HF_API_KEY` for gated Hugging Face models;
  `OPENAI_API_KEY` only for legacy cloud baselines/evaluation, not for the paper
  pipeline.

## Installation

```bash
# Create virtual environment on Linux
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install --upgrade --upgrade-strategy eager -r requirements.txt
```

## Configuration

1. Set paths, model names, and hyperparameters in `config.yaml`.
2. Export optional API keys only for the paths that need them:
   ```bash
   export HF_API_KEY='your-key'       # optional, for gated model download
   export OPENAI_API_KEY='your-key'   # legacy/cloud-only commands
   ```
3. Set the experiment name in `config.yaml`: `experiment: 'your_experiment_name'`.
4. Tune the router in the `routing:` block (`router:` sub-block — tie margin,
   reject floor, classifier training; plus reroute budget, competence weight,
   verifier units check, abstention target error).

```bash
# NOTE: the boolean flags take an explicit value — pass `=True` (e.g.
# `--infer_slg=True`), matching job.sh. A bare `--infer_slg` will not parse.

# 1. Download models (Qwen2.5-3B generator, LLaMA 3.2-1B router base,
#    jina-v2-base-en embedder; reranker + NLI once wired)
python main.py --download_models=True

# Print the exact ablation/baseline matrix.
python main.py --list_experiments=True

# 2. Build the Stack Exchange corpus (the only dataset)
python main.py --download_qa=True                      # fetch+extract default communities
python main.py --build_qa=True --qa_cap 5000           # build qa.json (<=5k per topic)
#   (needs a 7z extractor: `pip install py7zr` or a system 7z/7za/7zr)
python main.py --split_qa=True                         # qa_train/qa_test + per-topic split_by_title/
python main.py --split_qa=True --qa_subset 100         # smoke test: 100 pairs, all topics kept
python main.py --build_oos_split=True --oos_topics aviation robotics

# 3. Build per-topic retrieval indices + the out-of-scope split
#    OOS files are written under question_answer/oos/.

# 4. Build/load the router and indices.
#    Target architecture: frozen/local router + per-topic retrieval indices.
#    Current legacy path still has --finetune_router for the old classifier.

# 5. Run inference — routed retrieval + grounded generation + verify + abstain
python main.py --infer_slg=True          # batch
python main.py --chat_slg=True           # interactive multi-turn session
python main.py --infer_slg=True --limit=5   # seeded, title-stratified quick subset

# Baselines: run `python main.py --list_experiments=True` for the paper matrix.
# Legacy cloud baselines remain available but are not part of the on-prem claim:
#   python main.py --infer_baseline=True
#   python main.py --infer_rag=True

# 6. Evaluate behavior locally. `--evaluate=True` still uses legacy OpenAI
# semantic/AI-expert metrics and is optional/off-claim.
python main.py --slg_metrics=True
python main.py --evaluate=True

# 7. Experiments (see "Experiments" below)
python main.py --slg_ablations=True        # mechanism suite: full, -A, -B, -C, base
python main.py --slg_scalability=True      # pool scaling sweep (real-data subset)
python main.py --slg_metrics=True          # routing-curve + selective-prediction metrics
python main.py --slg_all=True              # all of the above, in order, as one job
python main.py --paper_assets=True         # aggregate everything -> LaTeX tables + figures
```
