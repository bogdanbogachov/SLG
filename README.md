# SLG — Small Language Router

A fine-tuning and inference pipeline for **privacy-constrained engineering
question answering on commodity hardware**. Everything runs on-premises — no
cloud LLM is contacted at inference time, so no question or document ever leaves
the machine. The system, **SLG (Small Language Router)**, answers engineering
questions with a pool of small specialist models (one LoRA adapter per topic
split) orchestrated by a reasoning-based router.

Three online mechanisms let the small local models close much of the gap to a
cloud LLM under tight resources:

- **(A) Online competence-learning router** — the router learns *which expert to
  trust* from its own verifier signal, with no labels and no retraining. Every
  verdict updates a per-expert, per-query-region reliability estimate that
  adjusts the cosine ranking. Routing improves over the lifetime of a run.
- **(B) Domain-grounded verifier** — instead of generic self-critique, answers
  are checked for engineering validity (numeric sanity, units on quantities,
  format, contradiction) by deterministic checks **and** the 8B critic, which
  also reports a confidence.
- **(C) Calibrated abstention** — a self-supervised confidence threshold decides
  when to answer and when to withhold, controlling the error rate among answers
  the system actually returns. A wrong engineering answer is worse than an
  honest "I can't answer this reliably."

## Requirements

- Python 3.10+
- CUDA-compatible GPU (the 8B critic needs a GPU large enough to hold it; the
  Qwen-3B experts/reasoner, 1B router classifier, and Jina embedder are lighter)
- Environment variables: `OPENAI_API_KEY`, `HF_API_KEY`, `TOGETHER_AI_API_KEY`

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
2. Export the API keys (used during data generation / baselines, not by local
   inference):
   ```bash
   export OPENAI_API_KEY='your-key'
   export HF_API_KEY='your-key'
   export TOGETHER_AI_API_KEY='your-key'
   ```
3. Set the experiment name in `config.yaml`: `experiment: 'your_experiment_name'`.
4. Tune the router in the `routing:` block (`router:` sub-block — tie margin,
   reject floor, classifier training; plus reroute budget, competence weight,
   verifier units check, abstention target error).

## Usage

### Workflow

```bash
# NOTE: the boolean flags take an explicit value — pass `=True` (e.g.
# `--infer_slg=True`), matching job.sh. A bare `--infer_slg` will not parse.

# 1. Download models (LLaMA 3.2-1B, LLaMA 3.1-8B, Qwen2.5-3B, jina-v2-base-en)
python main.py --download_models=True

# 2a. Real-world QA (Stack Exchange dumps -> question_answer/qa.json)
python main.py --download_qa=True                      # fetch+extract default communities
python main.py --build_qa=True --qa_cap 5000           # build qa.json (<=5k per expert)
#   (needs a 7z extractor: `pip install py7zr` or a system 7z/7za/7zr)

# 2b. …or generate synthetic QA pairs from source documents (scalability set only)
python main.py --create_qa=True
python main.py --combine_all_qa=True
python main.py --inflate_overshadowing=True

# 2c. Split -> full qa_train.json + qa_test.json, then per-expert split_by_title/
python main.py --split_qa=True
python main.py --split_qa=True --qa_subset 100   # smoke test: 100 pairs (80 train/20 test), all experts kept

# 3. Generate expert descriptions (after split_qa, before inference)
python main.py --slg_descriptions=True
#   Base LLaMA 3.1-8B-Instruct reads a random 25-answer sample of each expert's
#   deduplicated answers (an expert may have thousands, which would overflow the
#   context; sample is seeded, experts with <=25 use all) and writes a distinct
#   <=10-word description per expert, iteratively (each prompt sees the prior
#   descriptions) -> experiments/<exp>/slg_descriptions/descriptions.json.
#   Skipped if that file already exists (delete to rebuild).

# 4. Fine-tune one LoRA expert per topic split
python main.py --finetune=True

# 4b. Train the router classifier (Llama-1B seq-classification head) on the
#     training questions -> experiments/<exp>/slg_router/. Cheap single-GPU job;
#     one classifier serves every ablation + scalability pool size. Without it,
#     routing falls back to cosine similarity.
python main.py --finetune_router=True

# 5. Run inference
python main.py --infer_baseline=True     # OpenAI GPT-4.1 (cloud reference only)
python main.py --infer_rag=True          # RAG baseline
python main.py --infer_finetuned=True    # Single fine-tuned LLaMA
python main.py --infer_slg=True          # SLG — automated batch inference
python main.py --chat_slg=True           # SLG — interactive multi-turn session

# 6. Evaluate (scores the full run + baselines AND every ablation in the umbrella)
python main.py --evaluate
python main.py --evaluate --training_metrics

# 7. Ablation experiments (see "Experiments" below)
python main.py --slg_ablations        # full suite: full, -A, -B, -C, base
python main.py --slg_scalability      # expert-pool scaling sweep (real-data subset)
python main.py --slg_metrics          # routing-curve + selective-prediction metrics
python main.py --slg_all              # all of the above, in order, as one job
python main.py --paper_assets         # aggregate everything -> LaTeX tables + figures
```

## Dataset construction (real-world QA)

Paper-facing methodology for the core (non-synthetic) evaluation set. Modules:
`question_answer/download_stackexchange.py` and `question_answer/build_stackexchange_qa.py`
(exposed as `--download_qa` / `--build_qa`).

**Source & snapshot.** Stack Exchange Data Dump, per-community archives from the
Internet Archive (`https://archive.org/download/stackexchange/<host>.7z`). The dump
has **no DOI**; cite the Internet Archive item (identifier `stackexchange`), the
specific communities, and the **snapshot date** — record the date you downloaded,
since the dump is versioned by date. The most recent snapshots may only be
available via Stack Exchange's own request flow (policy changed ~2024).

**License (report this explicitly).** All content is **CC BY-SA** (Attribution +
ShareAlike). Obligations: (1) attribute each item — the extractor stores
`source_url`, `author` (answer), `question_author` per record; (2) any *published*
derivative dataset must be released under the **same CC BY-SA** (copyleft). State
the exact CC version matching your snapshot.

**Experts = communities.** Each community is one expert; `title` = community label
so `slug(title)` is the expert id (plugs into `split_qa_pairs_by_title`). Default
engineering set (12): Aviation, Engineering, Electrical Engineering (`electronics`),
Physics, Chemistry, Space Exploration, Robotics, 3D Printing, Signal Processing
(`dsp`), Motor Vehicle Maintenance (`mechanics`), Network Engineering,
Computational Science (`scicomp`).

**Answer selection.** From each question's answers, take the **accepted** answer;
if none, the **highest-scored** answer with score ≥ `--min_score` (default 1).
Questions with no qualifying answer are dropped. `question` = question title + body;
`answer` = selected answer body. Both are **HTML-stripped** (tags removed, entities
unescaped, whitespace normalized) and truncated to `--max_q_chars` / `--max_a_chars`
(2000 / 3000).

**Deduplication (leak-free).** Records are deduplicated on a normalized question key
(lowercased alphanumerics), keeping the highest-scored answer — applied within each
community and again globally across communities, so the later 80/20 split cannot
leak a question across train/test. *(This is the fix for the memorization observed
on the synthetic set.)*

**Balancing.** Per expert, keep the top-`--qa_cap` records by answer score
(default 5000; ties shuffled with `--seed`). With the default 12 communities, set
`--qa_cap 2500` for ~30k balanced, or `5000` and let the big sites cap out.

**Splitting.** `--split_qa` → `split_train_test(qa.json)` writes full
`qa_train.json` + `qa_test.json` (80/20, `random_state = CONFIG.seed`, dropping any
`title == answer`), then `split_qa_pairs_by_title(qa_train)` writes one per-expert
file under `question_answer/split_by_title/`. Experts train on `qa_train` only; test
is held out.

**Record schema** (extra provenance keys are ignored by the pipeline, kept for the
paper): `chapter`, `title` (expert), `question`, `answer`, `source_url`, `tags`,
`score`, `author`, `question_author`, `license`.

**Numbers to report.** The build step prints a per-community table (raw → dedup →
capped); record total pairs, #experts, per-expert counts, test-set size and
per-domain test counts (from `qa_test.json`). Target ~10k–30k total across ~12
distinct domains with a ≥1k leak-free test split; run each condition on ≥3 seeds.

**Reproducibility.** Fixed with `--seed` (build cap/shuffle) and `CONFIG.seed`
(train/test split). To reproduce: record the snapshot date, community list,
`--qa_cap`, `--min_score`, and both seeds.

## Pipeline flow

This section explains, in plain terms, exactly what happens to a question from
the moment it arrives to the moment an answer (or an abstention) comes back.
Both entry points — batch (`--infer_slg`) and interactive (`--chat_slg`) — share
the same machinery and differ only in how many experts may answer and whether
the session is multi-turn.

**The components involved:**

- a local **Jina** embedder (`jina-v2-base-en`, 768-dim) — turns text into
  vectors, entirely on-device (used for the online competence memory A and as the
  cosine routing fallback);
- a **router classifier** — **LLaMA 3.2-1B + a LoRA sequence-classification
  head**, fine-tuned on the training questions; this is what **picks the expert**
  (`--finetune_router`);
- a **Qwen-3B reasoner** (`Qwen2.5-3B-Instruct`) that plays three roles —
  **router tiebreaker** (only for ambiguous questions), **aggregator**,
  **compressor**;
- a **Llama-3.1-8B critic** — the LLM half of the verifier (B); a *different
  family* from the Qwen experts (and ≥ their size) so it isn't grading its own
  family;
- a pool of **Qwen2.5-3B + LoRA** experts, one adapter per topic split, loaded
  on demand.

> **Why a classifier router?** On a real run the 8B reasoning router routed only
> ~45% of questions to the correct expert, even though the cosine shortlist
> contained that expert 99.4% of the time — the reasoner was discarding the right
> answer. A small discriminative classifier trained on the questions restores
> routing to ~90%+; the 8B is kept only to break genuine ties.

### Two comparisons that are easy to confuse

Routing (picking the expert) and the competence memory are **two separate
things**. Keeping them apart is the key to understanding the flow:

| | what is compared | what it decides | selection rule |
|---|---|---|---|
| **Routing** | question → **classifier** | which expert answers | top-1 probability (Qwen-3B tiebreak if close) |
| **Competence memory (A)** | question ↔ **past questions** (cosine) | how much an expert's past pass/fail counts here | cosine ≥ **0.85** (a real threshold) |

The hard cosine threshold in the system (`0.85`) lives **only** in the competence
model, where it decides whether a previous question is close enough that its
outcome should reward or punish the expert on the current question. Routing
itself is a classifier decision, not a cosine cut. (When no trained router
exists, routing falls back to ranking experts by cosine similarity of the
question to each expert's mean-*question* embedding.)

### Step by step (one question)

1. **Embed the question.** Jina produces a normalized vector (used by the
   competence memory A, and as the routing fallback). In batch, every question is
   embedded once up front.

2. **Score every expert with the classifier.** The question runs through the
   fine-tuned Llama-1B classification head, giving a probability per expert
   (masked to the allowed pool). Already-tried experts are excluded so a reroute
   never re-picks a failure. *(Fallback with no trained router: cosine of the
   question against each expert's mean-question embedding.)*

3. **Apply the competence adjustment (A).** Each expert's score is nudged by
   `delta = competence_weight × (reliability − 0.5)`. *Reliability* is the
   expert's online pass/fail estimate **in the neighbourhood of this question** —
   built only from past questions whose cosine to the current one is ≥ 0.85. An
   expert with no relevant history gets delta 0. An expert that has been passing
   similar questions is boosted; one that has been failing them is demoted. This
   is the part that **learns online, with no labels and no retraining** — and it
   resets each run/session. (Formerly added to the cosine score; now added to the
   classifier probability — the mechanism is identical.)

4. **Pick the expert (+ Qwen-3B tiebreaker).** Rank experts by `score + delta`.
   If the top score is below `router.prob_floor`, the question ends as **REJECTED**
   ("a suitable expert was not found"). If the top-1/top-2 gap is below
   `router.tie_margin`, the **Qwen-3B reasoner** is loaded to choose among the top
   candidates (reading their descriptions + the question); if it declines
   (`NONE`) the classifier's top-1 stands. Otherwise the top-1 is taken directly
   and no reasoner is loaded for routing. In batch exactly **one** expert is
   picked; in interactive several may be (experts scoring ≥ `router.multi_threshold`).

6. **Answer with the expert(s) (Qwen-3B + LoRA).** The chosen adapter(s) generate
   the answer. In interactive mode any carried context from previous turns is
   prepended.

7. **Verify the answer (B, the domain verifier).** Two checks run together:
   - *Deterministic* (no model): numeric sanity (with a hard veto on
     empty answers or absurd/non-finite numbers), units-on-quantities when the
     question is quantitative, and format adherence for list-type questions.
   - *LLM critic* (Llama-3.1-8B, a different family from the Qwen experts):
     relaxed prompt — PASS if the answer is on-topic, factually plausible, and
     not degenerate; FAIL only genuinely wrong/useless answers. Emits
     `VERDICT: PASS/FAIL` plus `CONFIDENCE: 0–100`.

   The answer **passes** only if the critic says PASS *and* no deterministic veto
   fired. The **confidence** returned is `sqrt(critic_confidence ×
   deterministic_score)` (a conservative blend; 0 if vetoed).

8. **Learn from the verdict (feeds A and C).** The verdict updates two things:
   the expert's competence neighbourhood (reward on pass, punish on fail, local
   to this question — step 3 next time), and the abstention calibrator's
   observation set (step 9).

9. **Accept, abstain, or reroute (C, calibrated abstention).** The calibrator
   maintains a confidence threshold **τ**:
   - `passed` **and** `confidence ≥ τ` → **RESOLVED**; the answer is returned.
   - `passed` but `confidence < τ` → the answer is **withheld** (kept as a
     fallback in case nothing better turns up).
   - `failed` → the expert is demoted (A) and the question is **rerouted** to the
     next-best expert (the failed one is excluded from the next routing pass).

   τ starts at a floor of 0.5 and, once `abstention_min_calibration` (default 20)
   verdicts have accrued, becomes the lowest confidence at which the *critic*
   FAIL-rate among accepted answers stays ≤ `abstention_target_error` (0.10).

10. **Stop conditions.** Rerouting repeats up to `max_reroutes` (default 3).
    After the budget is spent a question ends as **RESOLVED**, **REJECTED**
    (router never chose anyone), **EXHAUSTED** (every attempt failed the critic),
    or **ABSTAINED** (something passed but never cleared τ).

11. **Aggregate and compress (interactive only).** When more than one expert is
    accepted, the Qwen-3B aggregator merges them into one cohesive reply; that reply
    is then compressed and carried forward as context for the next turn.

### The per-question flow at a glance

```
                       ┌─────────────────────────────────────────────┐
   question ──────────▶ 1B classifier router  → prob per expert       │
                       │   + (A) online competence adjustments        │
                       └───────────────┬─────────────────────────────┘
                                       │  top-1 (Qwen-3B tiebreak if close)
                                       ▼
                            chosen expert   ──── below prob_floor ─────┐
                                       │                               ▼
                                       ▼                        "suitable expert
                       Qwen-3B + LoRA expert answer              not found" (REJECTED)
                                       ▼
                    (B) domain verifier  =  deterministic checks
                                            (numbers, units, format)
                                          + 8B critic  → pass/fail + confidence
                                       │
            ┌──────────────────────────┼───────────────────────────────┐
            ▼                          ▼                                ▼
   pass & confidence ≥ τ      pass but confidence < τ            fail (critic)
        ACCEPT                  withhold (abstain-guard)        update (A): demote
            │                          │                         expert in region
            ▼                          ▼                                │
   aggregate → compress       (C) if no answer ever               reroute (≤ max_reroutes)
                              clears τ → ABSTAIN                        │
                                                          budget spent → EXHAUSTED
```

Every verdict feeds **(A)** the competence model (boost on pass, demote on fail,
local to the query region) and **(C)** the abstention calibrator (confidence +
verdict as a self-supervised label, which sets the threshold τ).

### Automated batch inference (`--infer_slg`)

- **Exactly one expert per question**, one answer returned, single turn.
- Processed in **rounds** to minimise model load/unload churn on tight VRAM:
  route every pending question, answer them grouped by expert, verify them all,
  then reroute only the failures into the next round (up to `max_reroutes`).
- Each round phase **batches its generation** to fill the GPU: routing scores the
  round in one classifier forward pass (only ties load the reasoner), expert
  answers are decoded per-expert-group, and all critic verdicts are decoded
  together (batch sizes: `generation.reasoner_batch_size` for the 8B critic / 3B
  reasoner, `generation.expert_batch_size` for the 3B experts). The online A/C updates
  are then applied in a fixed canonical order — grouped by expert (confident picks
  in question order, then tiebreaks) — the same order the unbatched loop walks, so
  the learned state is consistent with a single-stream run. See
  [Multi-GPU execution & batching](#multi-gpu-execution--batching).
- Per-question outcome (`status` in `slg.json`):
  - `resolved` — verified and confident enough to return.
  - `rejected` — the router found no suitable expert.
  - `exhausted` — the verifier rejected every attempt within the reroute budget.
  - `abstained` — an answer passed the critic but never cleared the calibrated
    confidence bar, so the system withholds it.
- Writes `answers/<exp>/<exp>/slg.json` (one record per question, original order)
  plus diagnostics under `answers/<exp>/<exp>/slg_diagnostics/` (routes, route
  traces, verifier log, **competence learning curve**, **calibration trace**).
  Everything for an experiment sits under one umbrella `answers/<exp>/`: the full
  run + baselines in `answers/<exp>/<exp>/`, and each ablation / scalability size
  as a sibling (`answers/<exp>/<exp>__no_competence/`, `.../<exp>__scale10/`, ...).

### Interactive session (`--chat_slg`)

- Multi-turn chatbot. The router may select **multiple experts** per turn when
  `routing.interactive_multi_expert` is set.
- The **router's reasoning trace and every verifier verdict (with confidence)
  are shown to the user**, as is each abstain-guard / competence demotion event.
- Accepted answers are **aggregated** into one cohesive reply, then
  **compressed** and carried as context into the next turn.
- If no answer clears the confidence bar the assistant abstains; if the router
  never picks an expert it reports that none was suitable.

## Scope and assumptions of the online mechanisms (A, B, C)

All three mechanisms learn online, with no labels and no retraining, and **reset
every run / chat session** — nothing persists to disk. That shared design choice
carries assumptions worth stating for each.

### (A) Online competence-learning router

- **Self-supervised signal.** Reliability is updated from the *verifier's*
  verdict, not from ground truth, so a systematically lenient or harsh critic
  biases who the router learns to trust. It inherits the verifier's accuracy.
- **Region by cosine threshold.** "Same region" is a hard cosine cut
  (`competence_similarity_threshold`, 0.85). Near the boundary the assignment is
  brittle; a question just under 0.85 inherits no competence and starts neutral.
- **Cold start / small-N.** A brand-new expert gets delta 0 until it has been
  tried; on short runs each region holds few observations, so the Beta estimate
  is wide and the nudge is small by design.

### (B) Domain-grounded verifier

- **Critic is a different family from the experts.** The deterministic checks are
  independent; the LLM verdict comes from Llama-3.1-8B while the experts are
  Qwen-3B, so it is not grading its own family (and the critic is ≥ the experts'
  size). It is still on-prem self-verification, not an external oracle — its
  agreement with ground truth should be validated on a labelled set.
- **Deterministic checks are heuristics.** Numeric-sanity, units, and format
  rules are regex/range based; only the absurd-magnitude and degenerate-answer
  vetoes are hard. Units/format are soft (confidence-shaping), so a correct but
  unconventionally phrased answer can be down-weighted, not rejected.

### (C) Calibrated abstention

The threshold τ is a **selective-prediction heuristic, not a method with formal
error guarantees.**

- **Self-supervised label.** τ controls the *critic's* FAIL rate among accepted
  answers, not true wrongness — and the same 8B acts as both critic and
  verifier, so it is calibrated against its own judgement.
- **Not conformal in the strict sense.** There is no held-out calibration split
  and the online stream is reused adaptively, so the exchangeability assumptions
  of split-conformal prediction do not hold. τ is *calibrated*, not *guaranteed*.
- **Small-N / per-session reset.** Below `abstention_min_calibration`
  observations τ is just the fixed floor (0.5); on short runs the calibration
  set is small and noisy.

`target_error` (default 0.10) is therefore the tolerated fraction of
critic-FAIL answers above the line, not a guaranteed bound on real error.

## Experiments

The mechanisms are evaluated by ablation. Each run reuses the same batch
pipeline; non-full runs write to `answers/<experiment>/<experiment>__<label>/` so they never
clobber each other.

| Experiment | Command | Proves |
|---|---|---|
| Leave-one-out ablation | `--slg_ablations` | each of A/B/C earns its place |
| Routing-accuracy curve | `--slg_metrics` | A learns online *(headline)* |
| Risk–coverage curve | `--slg_metrics` | C keeps the better answers |
| Scalability sweep | `--slg_scalability` | system scales with the expert pool |
| Paper tables + figures | `--paper_assets` | aggregates all of the above for the paper |

**Leave-one-out (`--slg_ablations`).** Runs five conditions — `full`, `no_competence`
(−A), `no_verifier` (−B, deterministic engineering checks off → critic-only),
`no_abstention` (−C, answer everything), and `base` (all three off). A single
condition can also be run with `--infer_slg --slg_ablation no_competence`.

**Metrics (`--slg_metrics`).** Deterministic post-processing over each finished
run — no models, no API calls. The correctness signal is **routing correctness**:
a question's ground-truth expert is the topic split it came from (`slug(title)`),
and a route is correct when the first chosen expert matches. It writes
`slg_diagnostics/selective_metrics.json` with:
- `routing_curve` — cumulative first-route accuracy vs. #questions; plot `full`
  against `no_competence` to show A improves online.
- `risk_coverage` — selective accuracy as coverage shrinks (answered questions
  ordered by verifier confidence); shows C trades coverage for accuracy.
- `summary` — coverage, overall and selective routing accuracy, status counts.

**Scalability (`--slg_scalability`).** Distractor scaling on a **fixed** question
set. The *core* experts that actually answer the questions are always present;
the pool is grown by adding **distractor** experts the questions never need
(`routing.scalability_sizes` = total pool sizes). Because the task is held
constant and every question stays answerable, this isolates the effect of a
larger pool — latency should stay roughly flat (the classifier does one forward
pass regardless of pool size; one trained router serves every size via an
allow-list mask) and routing accuracy should hold as irrelevant competitors are
added. Results per size go to
`slg_diagnostics/scalability.json`. **Runs on real data by default:**
`files.qa_scalability` → `question_answer/qa_scalability.json`, a fixed
100-question / 4-core-expert subset of `qa_test` (built by
`question_answer/build_scalability_subset.py`); the other 8 real experts are the
distractors, so sizes `[4,6,8,10,12]` need no synthetic data. Point
`files.qa_scalability` at a larger **synthetic** distractor set only to scale the
pool past the real corpus's 12 experts.

**Evaluation sweep (`--evaluate`).** Scores answer quality for **every run** under
the experiment's umbrella — the full run + baselines and each leave-one-out
ablation — writing one `experiments/<exp>/<label>/metrics.json` per run. The eval
outputs mirror the answers umbrella (full at `experiments/<exp>/<exp>/`, ablations
as siblings), so nothing lands at the `experiments/` top level. Scalability runs
(`<exp>__scale*`) are skipped (they repeat the task; no table needs their
quality). Resumable per file and idempotent, so re-running only scores what's new.

**Paper assets (`--paper_assets`).** One aggregator turns every result into
copy-paste-ready deliverables under `paper_assets/<experiment>/`:
- `tables/` — LaTeX `booktabs` floats: `main_quality.tex` (SLG vs. baselines),
  `ablation.tex` (leave-one-out A/B/C routing/selective), `ablation_quality.tex`
  (full answer quality per ablation), `scalability.tex`. Drop in with `\input{}`.
- `figures/` — PDF (for the paper) + PNG (for preview): routing-learning curve,
  risk–coverage curve, scalability, ablation bars. Add with `\includegraphics{}`.
- `README.md` — index of what each asset is and which flag produced it.

Every source is optional, so it degrades gracefully. Run `--paper_assets` **last** —
after `--slg_all` (behaviour metrics) **and** `--evaluate` (answer quality) — so it
captures everything; the quality table and Semantic/AI-Expert columns only appear
once `--evaluate` has produced `metrics.json`.

`--slg_all` chains the experiment runs in one job — ablations → scalability →
metrics (each step guarded) — but does **not** aggregate; call `--paper_assets`
separately at the end. Only the leave-one-out and scalability runs need the 8B
(GPU); metrics and `--paper_assets` are pure CPU and run anywhere, including the
login node.

## Multi-GPU execution & batching

The whole pipeline runs from **one** `job.sh` and scales across multiple GPUs
with no orchestration. Two independent layers of parallelism combine:

1. **Cross-run (across GPUs).** Independent jobs are dispatched one-per-GPU by
   `utils/parallel.py` (`run_parallel`): each LoRA fine-tune, each of the five
   ablation runs, each scalability pool size, and each fine-tuned baseline. One
   worker process per visible GPU (pinned with `CUDA_VISIBLE_DEVICES`, `spawn`
   start method); results are collected in task order. It **auto-scales to the
   GPU count** — raise the GPUs in `job.sh` and more jobs run at once, no code
   change. With ≤1 GPU (or `SLG_DISABLE_PARALLEL=1`) it falls back to in-process
   sequential, identical to before.
2. **Within-run (fills each GPU).** Inside a single run, work is **batched**:
   routing scores a whole round in one classifier forward pass (only the
   ambiguous tiebreak subset ever loads the Qwen-3B reasoner), then generation
   (`inference/slg/generation.py:generate_batch`, left-padded greedy decoding)
   decodes expert answers per expert group and all critic verdicts per round
   together. Batch sizes are `generation.reasoner_batch_size` (8B critic / 3B reasoner,
   also the classifier forward batch) and `generation.expert_batch_size` (1B
   experts), sized to load an 80GB GPU to ~75% without OOM.

**Ablation scheduling.** The suite has 5 runs but a node has 4 GPUs, so a naive
one-per-GPU dispatch leaves one preset running solo in a second wave. Instead,
the four **coupled** presets (`full`, `no_competence`, `no_verifier`,
`no_abstention` — each carries online A/C state that evolves across the question
stream and must run as one ordered process) go one-per-GPU, and **`base`** (A and
C off → questions independent) is **sharded data-parallel across all GPUs** and
merged in original order, bit-identical to a single-stream `base` run, filling
the GPUs that would otherwise idle in wave 2.

**Round-1 parallelism for a full `--infer_slg` run.** A standalone `full` run
can't be naively sharded (A/C are stateful), but its **round 1** — route+answer+
verify over the entire test set, i.e. the dominant cost — *is* sharded across all
GPUs, because round-1 routing is **A-independent** (competence is empty on the
first pass, so every question routes from pure classifier scores). Each shard
returns raw answers+verdicts; the parent then **replays A and C in canonical
order** and runs the small reroute rounds sequentially on one GPU
(`answer_shard_round1` + `finish_from_round1`). The result matches the
single-stream run (statuses + A/C learning curves), so the speed-up costs nothing
scientifically. Falls back to a single stream for ablations, `--limit`
quick-checks, one GPU, or `SLG_DISABLE_PARALLEL`.

**Consistency.** Cross-run jobs are atomic and independent, so results are
identical to a single-GPU run — only *which* GPU runs *which* job changes.
Batching matches the **unbatched round-based** run: batch inference is processed
in *rounds* (route all pending → answer → verify all → reroute), so the (A)
competence and (C) calibration state updates **once per round** from that round's
verdicts — it is round-granular online, not per-question online (this predates
batching; the round structure exists to bound model load/unload churn). Within a
round, A/C updates are applied in a fixed canonical order — grouped by expert
(confident picks in question order, then tiebreaks) — the order the answer/verify
phases walk, and greedy decoding means answers match in practice (padding + float
order can rarely flip a token → consistent, not bit-identical). A round with a single
question falls back to unbatched decoding and stays bit-identical.

**Single node, on purpose.** Parallelism is in-process (worker subprocesses +
in-memory queues), which cannot span nodes, so **all GPUs must be on one node**
(`--nodes=1 --gpus-per-node=h100:N`). Keep `--ntasks=1`. Enabling caches for
concurrent workers: the retriever embeds/caches the **full** expert set once and
applies the routing allow-list at query time, so parallel subset runs share one
read-only cache with no write race (`_warm_expert_cache` builds it before dispatch).

**Training batch sizes** are model-size-aware (`finetune/finetune.py`): the 1B
experts use `training.per_device_train_batch_size`, the 8B baseline the smaller
`training.per_device_train_batch_size_8b` (it fills 80GB sooner), with
`gradient_accumulation_steps=1`. Note this enlarges the effective batch versus a
2×2 setup, so it changes optimisation slightly over the fixed epoch budget —
linear-scale the learning rate if you need to match older runs.

The fine-tuned **baseline** inference (`ask_finetuned`) is batched the same way
(model-size-aware batch size, saved once per batch so it resumes at batch
granularity).

## Project Structure

```
SLG/
├── main.py                     # Entry point
├── config.yaml                 # Configuration (see the routing: block)
├── commands/                   # Command handlers (data, train, inference, eval)
├── inference/
│   ├── baseline.py             # GPT-4.1 / RAG / fine-tuned baselines
│   └── slg/                    # Small Language Router
│       ├── pipeline.py         # Orchestrator: ask() batch + chat() interactive
│       ├── classifier.py       # Router: 1B seq-classification head (the decider) + training
│       ├── retriever.py        # Jina embeddings: competence memory + cosine routing fallback
│       ├── reasoner.py         # Reasoning roles (batched): Qwen-3B route-tiebreak/aggregate/compress + Llama-8B critic
│       ├── generation.py       # Greedy decode helpers (single + left-padded batch)
│       ├── experts.py          # Qwen-3B + LoRA expert answering (batched)
│       ├── competence.py       # (A) online expert-competence model
│       ├── verifier.py         # (B) domain-grounded verifier
│       ├── abstention.py       # (C) calibrated abstention
│       └── session.py          # Per-run state binding A + C + carried context
├── finetune/                   # LoRA fine-tuning (model-size-aware batch sizes)
├── evaluate/                   # Evaluation metrics
├── utils/                      # Model loading, paths, prompts
│   └── parallel.py             # One-worker-per-GPU pool (cross-run parallelism)
└── download_llama/             # Model download utilities
```

## Key Features

- **Fully on-prem**: no cloud LLM at inference time; data never leaves the host.
- **LoRA experts**: one adapter per topic split, loaded on demand.
- **Classifier router** (1B seq-classification head) with an **online competence
  model** layered on top that learns who to trust without labels or retraining;
  the Qwen-3B reasoner breaks ties only.
- **Domain-grounded verification** and **calibrated abstention** for reliable
  engineering answers.
- **Evaluation**: ROUGE, METEOR, Exact Match, semantic similarity, AI Expert.
- **Multi-GPU from one job**: independent runs dispatched one-per-GPU and each
  run's generation batched to fill the GPU; auto-scales with the requested GPU
  count, results consistent with a single-GPU run
  ([details](#multi-gpu-execution--batching)).

## Output Structure

```
experiments/
└── {experiment_name}/
    ├── slg/                       # SLG expert LoRA adapters (one dir per expert)
    ├── slg_index/                 # cached expert routing embeddings
    ├── slg_descriptions/          # descriptions.json (expert -> short description)
    └── metrics.json               # evaluation results

answers/
└── {experiment_name}/
    ├── gpt-4.1-2025-04-14.json
    ├── rag.json
    ├── finetuned_3_2_1b.json
    ├── slg.json                   # SLG predictions (chapter/title/question/experts/answer/status)
    └── slg_diagnostics/
        ├── slg_routes.json        # experts tried per question
        ├── route_traces.json      # router reasoning traces
        ├── critic_log.json        # verifier verdicts (pass/fail, confidence, checks)
        ├── competence_log.json    # (A) online learning signal
        └── calibration_log.json   # (C) threshold history + coverage
```
