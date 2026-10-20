# Paper methods and experiment notes

This note records how the current experiment inputs are constructed and which analyses `submit_all.sh` schedules. The counts below describe the datasets currently present in this checkout; regenerate or verify them with `python main.py --build_datasets` before using them in the paper.

## Dataset construction

Run `python main.py --build_datasets` on the login node before submitting jobs. The builder reads only these inputs:

- Avia: `qa_archive/upd_slg_l/qa.json`.
- CQADupStack: `question_answer/cqadupstack_clustered/qa_min20.json`.
- Confuser sentences: `question_answer/inflating_material.json`.

No MM, SRM, or OM reader is called to create the experiment QA data. For the small Avia subset, the `chapter` metadata already in the Avia source is used to keep maintenance manual chapters 5–12, 20–28, and 30–32. No question generation runs during dataset construction.

### Filtering and deduplication

The builder first removes rows where `title == answer`. It then assigns every row a stable SHA-256 `qid` using the lowercased, whitespace-normalized question and original answer. Deduplication happens **before train/test splitting and before answer inflation**:

1. Remove exact normalized question duplicates globally.
2. Within each original-answer group, process questions in stable `qid` order.
3. Tokenize each question as the set from `re.findall(r"\w+", question.lower())`.
4. Keep a question only when its word Jaccard similarity with every already-kept question for that answer is below 0.8. The threshold is configured as `dataset.min_jaccard_dedup` in `config.yaml`.

The same procedure is applied to CQAD for consistency. Each dataset is split independently with an 80/20 train/test split and `random_state` equal to seed 42, 53, or 64. Train and test rows are sorted by `qid` when written. The resulting split files are shared by all models in a dataset/seed bundle. Each train split also has a `by_title/` directory with exactly one expert training file for each title in that train split.

Current built counts:

| Dataset | Questions | Unique answers | Titles | Questions per answer (min–max) | Train / test per seed |
|---|---:|---:|---:|---:|---:|
| `l_avia` | 22,441 | 974 | 276 | 8–28 | 17,952 / 4,489 |
| `s_avia` | 11,841 | 525 | 139 | 8–28 | 9,472 / 2,369 |
| `s_avia_25` | 11,841 | 525 | 139 | 8–28 | 9,472 / 2,369 |
| `s_avia_50` | 11,841 | 525 | 139 | 8–28 | 9,472 / 2,369 |
| `s_avia_75` | 11,841 | 525 | 139 | 8–28 | 9,472 / 2,369 |
| `s_avia_100` | 11,841 | 525 | 139 | 8–28 | 9,472 / 2,369 |
| `cqad20` | 3,429 | 72 | 11 | 20–260 | 2,743 / 686 |

The Avia input currently has 27,692 rows. The title-equals-answer filter removes 420 rows, leaving 27,272 before near-duplicate removal; the 0.8 deduplication removes 4,831. CQAD currently has 3,435 input rows, of which deduplication removes 6. These are build-specific counts, not guarantees for a later replacement source file.

### Avia subsets and inflation variants

`l_avia` is the deduplicated Avia source. `s_avia` filters those same deduplicated rows to the configured maintenance chapters. It does not use a separate manual-generated QA file. Inflation changes only the answer field; questions and `qid`s remain the same.

The variants `s_avia_25`, `s_avia_50`, `s_avia_75`, and `s_avia_100` use a fixed dataset-build ranking seed (`20261015`), independent of experiment seed. For every title, distinct answers are selected by a stable hash ranking. The selected count is rounded half up per title. Consequently, the realized percentages can differ from the nominal percentages. For this build, the selected shares are 149/525 (28.4%), 299/525 (57.0%), 424/525 (80.8%), and 525/525 (100%). The selected sets are nested, and the builder asserts that nesting and identical test `qid`s across `s_avia` and all inflation levels for each seed.

Confusers are assigned within each title in deterministic `(title frequency, title, answer)` order. An answer keeps the same confuser at every inflation level where it is selected. Confusers are unique among answers within a title and reused across titles. The answer becomes `<confuser>`, a blank line, and the original answer. There are 24 distinct confusers in use at 100% in the current build.

The inflation percentage is applied within each title to its distinct answers, not to individual QA rows. Thus all questions associated with a selected `(title, original answer)` receive the same modified answer. The fixed hash ranking makes the selected answer sets nested across levels and independent of the train/test seed. The qid is based on the original answer, so it is preserved when the displayed/scored answer is inflated. This design permits the split membership and question-level pairing to remain fixed while answer prefixes change.

### Leakage and overlap summaries

For each seed and dataset, the build summary reports the distribution of each test question's maximum word-Jaccard similarity against training questions with the same original answer. It also reports the fraction at thresholds 0.25, 0.5, 0.75, 0.8, and 1.0, plus mean, median, 90th percentile, 95th percentile, and maximum. Since global exact-question deduplication occurs before splitting, there are no exact repeated questions across the resulting train/test partitions in the current build.

The same summary reports answer-prefix overlap at prefix length 5: UF/TF for the full dataset and mean US/TS across per-title train files. These are build diagnostics; use the generated summary for the final source files when reporting them.

The prefix diagnostic uses distinct answer strings, splits each answer after its first five whitespace-delimited tokens, and counts a prefix as an overlap when it maps to more than one continuation. UF and TF are calculated over the complete dataset; US and TS are arithmetic means of the corresponding counts across title-specific training files. These are descriptive overlap measures, not model-performance measures.

## Model and inference design

The default comparison contains four systems per dataset/seed cell:

- **1B baseline:** LoRA fine-tuning of the Llama-3.2-1B-Instruct base model on the cell's complete training split.
- **8B baseline:** LoRA fine-tuning of the Llama-3.1-8B-Instruct base model on the same training split.
- **SLG:** one Llama-3.2-1B LoRA expert per title represented in the training split, plus a fine-tuned Llama-3.2-1B question-to-title classifier router. Expert files are generated from the cell's training split only. Their count is expected to equal the number of training titles. The current training loop trains experts sequentially.
- **RAG:** the unfine-tuned Llama-3.1-8B-Instruct generator, with retrieval over training answers only. Exact duplicate answer strings are removed before chunking; answer text is split into 500-word chunks with 50-word overlap. Jina embeddings (`jinaai/jina-embeddings-v2-base-en`) are indexed with FAISS `IndexFlatL2`; the five nearest chunks are placed in the generation context. RAG has no OpenAI inference calls.

For SLG, a normalized mean embedding of each expert's distinct training answers is used to form the expert similarity graph. The current configuration retains up to three neighbors with cosine similarity at least 0.9. The fine-tuned router selects the main expert; confidence controls whether zero or more of its graph neighbors are also invoked. Oracle routing replaces the selected expert with the question's title-matched expert. Wrong routing selects a seeded random expert other than the title-matched expert, deterministically from the run seed and qid. These are inference-only counterfactuals; they do not retrain experts or the router.

The 1B, 8B, and SLG generators use the shared chat-template utilities and greedy decoding, with a maximum of 750 new tokens in the current configuration. RAG uses its own system prompt with retrieved context and the same maximum generation length. No GPT-4.1 answer-generation baseline is run. The SLG router is fine-tuned from title labels; it uses a stratified holdout drawn from the training rows for early stopping and separately logs results on the bundle's test split. That logged test result is diagnostic and is not used by the main answer-generation metric pipeline.

### Training procedure and reproducibility

The baseline and expert adapters use the shared LoRA configuration (`r=16`, `alpha=128`, dropout 0.05), 10 maximum epochs, learning rate 0.001, effective batch size 4 per device before any multi-device effects, weight decay 0.001, cosine scheduling, 5% warm-up, and early stopping after three unimproved validation evaluations. The training data is tokenized with the model chat template, truncated/padded to 1,024 tokens, and split internally into train/validation subsets for early stopping. The router has a separate LoRA configuration (`r=4`, alpha 16, q/v projections), a 512-token maximum input, and up to 45 epochs with step-based evaluation and early stopping. Report these as code settings; actual stopping epochs and GPU time must come from each run's logs/metrics.

Training checkpoints resume from the latest Hugging Face Trainer checkpoint when one exists. Completed adapters are skipped when both their final adapter artifact and `.done` marker are present. Inference appends JSONL answers by qid and skips qids already written. Evaluation checkpoints per-question progress and can rescore selected metrics. These mechanisms support restart after interruption, but output files from the actual completed runs remain the evidence that every cell finished.

## Evaluation and paper analyses

Each full bundle covers one dataset and one seed. It trains the 1B and 8B baselines plus SLG, then runs inference for 1B, 8B, SLG, and local RAG. It also runs oracle and wrong-expert router inference. Evaluation is a separate login-node step because the Rorqual compute nodes cannot reach the OpenAI API. On the login node, `bash evaluate_all.sh` evaluates every available model answer file for all 7 datasets and 3 seeds; `--dataset` and `--seed` can restrict it. The API key is required only for this scoring step and is read from `OPENAI_API_KEY`.

Evaluation retains per-question score records keyed by `qid`; the eight reported metrics are BLEU, ROUGE-1/2/L, exact match, METEOR, Entailment, and AI_Expert. BLEU uses whitespace tokenization and smoothing method 4. ROUGE uses stemming and F-measures. Exact match compares stripped strings. METEOR uses NLTK tokenization and WordNet resources. “Entailment” is operationalized as cosine similarity between OpenAI `text-embedding-3-small` embeddings, transformed from [-1, 1] to [0, 1]; it is an embedding-similarity proxy rather than a textual entailment classifier. AI_Expert is a binary GPT-4.1-nano judgment using the dataset-specific judge prompt (aerospace-specific for Avia, domain-neutral for CQAD). The API key is read from `OPENAI_API_KEY`; API-based evaluation runs on the login node because the compute nodes used for bundle jobs have no network access.

The paired significance analysis reads only per-question score records and checks that the four primary models have matching qid sets in each cell. For each dataset and each of the eight metrics, it compares SLG with 8B, 1B, and RAG. In each of 10,000 bootstrap replicates it samples answer groups with replacement independently within each seed, computes a question-weighted mean difference for that seed, then averages the three seed means with equal seed weight. It reports the observed mean difference, a percentile 95% interval, a two-sided sign-tail bootstrap p-value, and Holm-adjusted p-values. The bootstrap's fixed RNG seed is 20261015. The comparison of SLG with 8B is treated as equivalence only for ROUGE-L and AI_Expert, using a ±0.01 interval criterion; other metrics have no equivalence declaration. The output includes CSV and LaTeX files.

For `s_avia_100`, the same analysis additionally computes difference-in-differences against `s_avia_25`: `(SLG_100 − SLG_25) − (baseline_100 − baseline_25)`, separately for the 8B and 1B baselines and for all eight metrics. It requires identical qid sets across these inflation levels. Holm correction is applied across all rows produced for that dataset, including these inflation-contrast rows. This is a specific 100%-versus-25% contrast, not a fitted trend over all four inflation levels.

The generated paper quality table first averages question-level scores within each seed and then reports the arithmetic mean and sample standard deviation (`ddof=1`) of the three seed-level means. The deployment table aggregates per-question latency and peak allocated GPU memory from answer records and training GPU-hours from metrics logs. Its current “minimum GPU tested” field is hard-coded to the job resource label (H100 3g.40gb); it is not an empirically estimated minimum hardware requirement. RAG has no fine-tuning GPU-hours, and the SLG parameter-size label refers to the active expert, not the total collection of stored experts plus router.

For the full default matrix, `./submit_all.sh` submits 21 independent train/inference bundle jobs (7 datasets × 3 seeds), named `slg-<dataset>-s<seed>` in Slurm. Once they finish, run `./evaluate_all.sh` on a login node. After evaluation has created the per-question score files, submit the paired significance, paper/deployment tables, and activation patching with `sbatch --job-name=slg-post-analysis job.sh post_analysis`. `submit_all.sh` does not submit evaluation or post-analysis automatically. The `--slg_all` ablation and `--scalability` options currently stop with an explicit “runner not included” error; they are not part of the default experiment matrix.

Activation patching selects at most 200 test questions per cell with 1B ROUGE-L ≤ 0.3 and SLG ROUGE-L ≥ 0.8, taking the first qualifying qids in sorted-qid order. For `s_avia_25`, `s_avia_50`, and `s_avia_75`, it instead reuses the qualifying qids selected at `s_avia_100` for the same seed, enabling a paired inflation comparison. At each of the 16 Llama-3.2-1B layers, it replaces the baseline model's prompt-position residual stream with the corresponding residual from the title-matched SLG expert, and computes teacher-forced log probability for up to the first 16 gold-answer tokens. A control patches residuals from another test question, matching prompt length when possible and otherwise truncating or padding positions. Restoration is `(patched log-probability − baseline log-probability) / (SLG log-probability − baseline log-probability)`; examples with a near-zero denominator are skipped. The plot reports a 95% percentile interval from 2,000 bootstrap resamples of selected questions per layer. This is a selected-subset mechanistic analysis, not a test-set average effect. The full result is generated during post-analysis; the smoke-test path patches one forced example only to verify execution, not to produce a paper result. The planning estimate is about one GPU-hour per cell, but actual runtime depends on qualifying examples and hardware.

### Interpretation and claims to verify

- Near-duplicate removal and split-before-inflation ordering are intended to reduce lexical leakage and hold qid membership constant across inflation conditions. The per-build Jaccard summaries should be reported; deduplication does not guarantee that all near-duplicates across different original answers are removed.
- The router counterfactuals compare expert choice while holding the trained expert set fixed. They can describe sensitivity to using the title-matched expert or a randomly selected wrong expert; they do not by themselves identify a general causal effect of specialization.
- The paired bootstrap treats questions with the same scored gold-answer string as a cluster and treats seed-specific splits as independent strata. In inflated Avia cells, the scored gold answer includes the confuser prefix; verify that these resulting answer groups match the intended original-answer clusters before making cluster-level inferential claims.
- Activation-patching examples are selected using answer outcomes from the evaluated test split. Interpret patching as a post hoc mechanism probe on selected successes/failures, not as an unbiased estimate over the test population. The late-layer restoration curve may be difficult to interpret because late-layer replacement can trivially restore output behavior.
- Numerical statements in this note are from the current local generated data. Before publication, confirm that the cluster used the same source files and record the build summary, software/model revisions, actual seeds, completed cell outputs, and evaluated per-question files. No performance results are asserted here.

## Reproducibility commands

```bash
python main.py --download_models
python main.py --build_datasets
uv venv --python "$(command -v python)" --seed ENV
uv pip sync --python ENV/bin/python --index-url https://pypi.org/simple requirements.txt
bash submit_all.sh
```

After all bundle jobs finish:

```bash
export OPENAI_API_KEY='...'
bash evaluate_all.sh
sbatch --job-name=slg-post-analysis job.sh post_analysis
```

To evaluate only a smoke-test cell after its compute job has completed, use `./evaluate_all.sh --dataset s_avia_100 --seed 42 --smoke`. The `--smoke` mode reads the separate `results/smoke/` answers and `data/_smoke/` test split. It still requires `OPENAI_API_KEY` on the login node. Paired significance testing is not meaningful for this single-cell smoke run.

The dataset builder is deterministic and skips existing outputs unless `--force` is passed. Jobs require prepared split files and do not build or split data themselves.

## Quick pipeline smoke run

`bash smoke_test.sh [dataset] [seed]` submits one isolated Slurm validation job named `slg-smoke-<dataset>-s<seed>`. It defaults to `s_avia_100`, seed 42. The job selects two section titles, 5 train questions per title, and up to 2 test questions per title, then runs both baseline fine-tunes, two SLG experts and the router, four standard inference models, oracle/wrong routing, and one activation-patching example. Smoke artifacts go under `data/_smoke/` and `results/smoke/`; they do not overwrite final experiment cells. The patching output only checks that the code path executes; it is not a paper result. API evaluation is not run in the compute job; use the login-node evaluation workflow for full results. Model weights must already be downloaded.
