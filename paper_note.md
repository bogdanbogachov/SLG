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
| `cqad20_fine` | 3,429 | 72 | 24 | 20–260 | 2,743 / 686 |
| `cqad20_answer` | 3,429 | 72 | 72 | 20–260 | 2,743 / 686 |

The Avia input currently has 27,692 rows. The title-equals-answer filter removes 420 rows, leaving 27,272 before near-duplicate removal; the 0.8 deduplication removes 4,831. CQAD currently has 3,435 input rows, of which deduplication removes 6. These are build-specific counts, not guarantees for a later replacement source file.

### Avia subsets and inflation variants

`l_avia` is the deduplicated Avia source. `s_avia` filters those same deduplicated rows to the configured maintenance chapters. It does not use a separate manual-generated QA file. Inflation changes only the answer field; questions and `qid`s remain the same.

The variants `s_avia_25`, `s_avia_50`, `s_avia_75`, and `s_avia_100` use a fixed dataset-build ranking seed (`20261015`), independent of experiment seed. For every title, distinct answers are selected by a stable hash ranking. The selected count is rounded half up per title. Consequently, the realized percentages can differ from the nominal percentages. For this build, the selected shares are 149/525 (28.4%), 299/525 (57.0%), 424/525 (80.8%), and 525/525 (100%). The selected sets are nested, and the builder asserts that nesting and identical test `qid`s across `s_avia` and all inflation levels for each seed.

Confusers are assigned within each title in deterministic `(title frequency, title, answer)` order. An answer keeps the same confuser at every inflation level where it is selected. Confusers are unique among answers within a title and reused across titles. The answer becomes `<confuser>`, a blank line, and the original answer. There are 24 distinct confusers in use at 100% in the current build.

The inflation percentage is applied within each title to its distinct answers, not to individual QA rows. Thus all questions associated with a selected `(title, original answer)` receive the same modified answer. The fixed hash ranking makes the selected answer sets nested across levels and independent of the train/test seed. The qid is based on the original answer, so it is preserved when the displayed/scored answer is inflated. This design permits the split membership and question-level pairing to remain fixed while answer prefixes change.

### CQAD expert-granularity variants (`cqad20_fine`, `cqad20_answer`)

`cqad20` has only 11 experts (one per StackExchange site), so each expert must cover several unrelated answers. Two variants re-partition the **same** rows into finer experts. Only the `title` (the expert label) changes: questions, answers, qids and train/test splits are identical to `cqad20` for every seed (the builder reuses the `cqad20` split files and asserts identical test qids).

- Grouping uses answer text only (TF-IDF, sublinear tf, cosine), independent of seeds and of any model output.
- Within each site, the distinct answers are ordered by a stable SHA-256 hash and split into `ceil(n / k)` balanced groups (sizes differ by at most one). Each group starts from the first unassigned answer and takes its most similar unassigned answers.
- `cqad20_fine`: at most k = 4 answers per expert → 24 experts. `cqad20_answer`: k = 1 → 72 experts (one expert per answer cluster). Smallest expert training set: 14 questions.
- Expert names are `<site title> <group number>`.

Because the questions and splits are unchanged, the 1B, 8B and RAG baselines are identical to `cqad20` and are copied rather than rerun (`reuse_baselines.sh`, which checks that the test qids match). Only the SLG experts, routers and SLG answers are new. These variants form a granularity series 11 → 24 → 72 experts on fixed data. Framing for the paper: the expert partition is a deployment design choice, and the router makes finer experts usable. The original 11-expert `cqad20` stays in the paper.

### Leakage and overlap summaries

For each seed and dataset, the build summary reports the distribution of each test question's maximum word-Jaccard similarity against training questions with the same original answer. It also reports the fraction at thresholds 0.25, 0.5, 0.75, 0.8, and 1.0, plus mean, median, 90th percentile, 95th percentile, and maximum. Since global exact-question deduplication occurs before splitting, there are no exact repeated questions across the resulting train/test partitions in the current build.

The same summary reports answer-prefix overlap at prefix length 5: UF/TF for the full dataset and mean US/TS across per-title train files. These are build diagnostics; use the generated summary for the final source files when reporting them.

The prefix diagnostic uses distinct answer strings, splits each answer after its first five whitespace-delimited tokens, and counts a prefix as an overlap when it maps to more than one continuation. UF and TF are calculated over the complete dataset; US and TS are arithmetic means of the corresponding counts across title-specific training files. These are descriptive overlap measures, not model-performance measures.

## Model and inference design

The default comparison contains four systems per dataset/seed cell:

- **1B baseline:** LoRA fine-tuning of the Llama-3.2-1B-Instruct base model on the cell's complete training split.
- **8B baseline:** LoRA fine-tuning of the Llama-3.1-8B-Instruct base model on the same training split.
- **SLG (final system):** one Llama-3.2-1B LoRA expert per title represented in the training split, plus the **Mahalanobis hidden-state router** (next section). Expert files are generated from the cell's training split only. Their count is expected to equal the number of training titles. The current training loop trains experts sequentially.
- **SLG-classifier (router ablation):** the same experts with the earlier router, a LoRA-fine-tuned Llama-3.2-1B question-to-title classifier. Answers: `slg_finetuned_router.jsonl`.
- **SLG-oracle (upper bound):** the same experts, always using the title-matched expert. Answers: `slg_finetuned_router_oracle.jsonl`.
- **RAG:** the unfine-tuned Llama-3.1-8B-Instruct generator, with retrieval over training answers only. Exact duplicate answer strings are removed before chunking; answer text is split into 500-word chunks with 50-word overlap. Jina embeddings (`jinaai/jina-embeddings-v2-base-en`) are indexed with FAISS `IndexFlatL2`; the five nearest chunks are placed in the generation context. RAG has no OpenAI inference calls.

For SLG with the fine-tuned or Mahalanobis router, exactly **one** expert answers each question. The router's top-1 expert generates the answer, and no graph neighbours are invoked (`_route_after_expert` goes straight to the aggregator). The expert-similarity graph (normalized mean embedding of each expert's distinct training answers, up to three neighbours with cosine ≥ 0.9, confidence-controlled expansion) is used only by the legacy cosine router, which is not part of the paper's runs. Oracle routing replaces the selected expert with the question's title-matched expert. Wrong routing selects a seeded random expert other than the title-matched expert, deterministically from the run seed and qid. These are inference-only counterfactuals; they do not retrain experts or the router.

The 1B, 8B, and SLG generators use the shared chat-template utilities and greedy decoding, with a maximum of 750 new tokens in the current configuration. RAG uses its own system prompt with retrieved context and the same maximum generation length. No GPT-4.1 answer-generation baseline is run. The SLG router is fine-tuned from title labels; it uses a stratified holdout drawn from the training rows for early stopping and separately logs results on the bundle's test split. That logged test result is diagnostic and is not used by the main answer-generation metric pipeline.

### Training procedure and reproducibility

The baseline and expert adapters use the shared LoRA configuration (`r=16`, `alpha=128`, dropout 0.05), 10 maximum epochs, learning rate 0.001, effective batch size 4 per device before any multi-device effects, weight decay 0.001, cosine scheduling, 5% warm-up, and early stopping after three unimproved validation evaluations. The training data is tokenized with the model chat template, truncated/padded to 1,024 tokens, and split internally into train/validation subsets for early stopping. The router has a separate LoRA configuration (`r=4`, alpha 16, q/v projections), a 512-token maximum input, and up to 45 epochs with step-based evaluation and early stopping. Report these as code settings; actual stopping epochs and GPU time must come from each run's logs/metrics.

Training checkpoints resume from the latest Hugging Face Trainer checkpoint when one exists. Completed adapters are skipped when both their final adapter artifact and `.done` marker are present. Inference appends JSONL answers by qid and skips qids already written. Evaluation checkpoints per-question progress and can rescore selected metrics. These mechanisms support restart after interruption, but output files from the actual completed runs remain the evidence that every cell finished.

## Mahalanobis hidden-state router (final SLG router)

Code: package `hidden_router/` (`features.py`, `router.py`, `probe.py`, `tfidf.py`, `assemble.py`, `pipeline.py`); integrated into `inference/slg.py` as `router_method="mahalanobis"`. Tests: `tests/test_hidden_router.py`.

### Method

1. **Features.** The base Llama-3.2-1B-Instruct model (the same frozen base the experts are LoRA-adapted from, with no adapter) reads the raw question text (no chat template, at most 512 tokens, fp16). For each of the 17 hidden states (index 0 = token embeddings, i = output of transformer block i), the token vectors are mean-pooled over the question, excluding BOS. BOS is excluded because it is an attention sink with a very large norm that is identical for every question.
2. **Classifier.** Gaussian classes with one shared covariance (linear discriminant / Mahalanobis distance), one class per expert:
   - score_e(h) = log p(e) − ½ (h − μ_e)ᵀ S⁻¹ (h − μ_e), and p(e | h) = softmax_e(score_e).
   - Features are first z-scored per dimension using the pooled within-expert standard deviation. LLM hidden states contain a few "massive activation" dimensions with very large variance, which would otherwise dominate the shrinkage.
   - S = (1 − α)·C + α·I, where C is the pooled within-expert correlation matrix.
   - The prior p(e) is the training frequency of each expert.
   - Everything is closed form: class means, one covariance, and a Cholesky solve. There is no gradient training.
3. **Model selection without test data.** Layer ∈ {0..16} and α ∈ {0.05, 0.2, 0.5} are chosen by top-1 accuracy on a stratified 20% validation slice of the **training** questions. Experts with fewer than 3 training questions keep all their questions in the fit part. Ties are broken by top-5 accuracy, then by the earlier layer. The router is then refit on all training questions with the chosen (layer, α). The test split is never used for any choice.
4. **Inference.** One forward pass of the frozen 1B base over the question, then one matrix product (experts × hidden size). The top-1 expert answers.

### Cost

- Feature extraction for all 25,870 distinct questions of the 9 datasets: about 6 min on an H100 1g.10gb MIG slice (10 GB). The features are seed-independent and cached once.
- Fitting with the full layer × α grid: about 20 s per cell on the same slice. The training cost is negligible next to the classifier router's LoRA training run.
- No extra model is stored: the router is the frozen base model plus a (experts × 2048) matrix and a bias vector per cell.

### Routing results (probe, test split, mean ± sample SD over seeds 42/53/64)

From `results/hidden_router/probe_summary.md` and `tfidf_summary.md`. "TF-IDF" is the lexical baseline (next subsection). "L0" is the same Mahalanobis classifier on layer 0 (token embeddings, diagnostic).

| Dataset | Experts | Classifier top-1 | TF-IDF top-1 | L0 top-1 | **Mahalanobis top-1** | Mahalanobis top-5 | Chosen layers |
|---|---:|---|---|---|---|---|---|
| l_avia | 276 | 0.774 ± 0.069 | 0.880 ± 0.002 | 0.929 ± 0.002 | **0.942 ± 0.004** | 0.989 ± 0.001 | 3, 3, 2 |
| s_avia (all inflation levels)* | 139 | 0.862–0.880 | 0.934 ± 0.004 | 0.953 ± 0.003 | **0.961 ± 0.002** | 0.994 ± 0.000 | 3, 1, 2 |
| cqad20 | 11 | 0.994 ± 0.001 | 0.984 ± 0.006 | 0.989 ± 0.005 | **0.998 ± 0.002** | 1.000 ± 0.000 | 7, 6, 5 |
| cqad20_fine | 24 | 0.771 ± 0.017 | 0.901 ± 0.007 | 0.914 ± 0.013 | **0.939 ± 0.009** | 0.998 ± 0.002 | 14, 11, 13 |
| cqad20_answer | 72 | 0.739 ± 0.019 | 0.862 ± 0.006 | 0.870 ± 0.006 | **0.923 ± 0.006** | 0.991 ± 0.003 | 16, 16, 15 |

\* Inflation changes only answers. Router training and test questions are identical across `s_avia` and `s_avia_25/50/75/100`, so the Mahalanobis and TF-IDF results are identical across them; report them once. The classifier router was retrained per inflation level, so its top-1 differs: s_avia 0.875 ± 0.019, _25 0.880 ± 0.006, _50 0.862 ± 0.008, _75 0.876 ± 0.007, _100 0.875 ± 0.020.

Observations:
- **Large gains where routing was the bottleneck:** +17 points on l_avia, about +9 on s_avia, +17 on cqad20_fine and +18 on cqad20_answer. The variance across seeds collapses (l_avia SD 0.069 → 0.004), because a closed-form fit has no optimisation instability.
- **Top-5 ≥ 0.989 everywhere,** so the correct expert is almost always near the top.
- **The best layer depends on expert granularity.** Coarse topic experts (aviation chapters, 11 StackExchange sites) are routed best from early or middle layers (1–7). Fine-grained experts within one site are routed best from late layers: 11–14 for 24 experts and 15–16 for 72 experts. Telling near-duplicate topics apart needs more contextual or semantic representations. Per-layer test curves: `results/stats/routing_layers.csv` and `results/<ds>/seed_<s>/hidden_router/per_layer_test.json`.
- **The layer is chosen automatically and cheaply** on training data. A deployer does not need activation patching or any test labels to set up the router.
- **On aviation the layer curve is flat:** l_avia seed 42 gives 0.924–0.941 test top-1 across all 17 layers. The choice of layer matters there only marginally, but it matters for fine-grained experts.

### Lexical baseline (TF-IDF + same classifier)

`python -m hidden_router tfidf` (CPU, about 7 min for all cells). Per cell, TF-IDF (word 1–2 grams, sublinear tf, min_df 2) and TruncatedSVD to 768 dimensions are fitted on the training questions only. The SVD keeps 49–59% of the variance. Then the same Mahalanobis classifier is fitted, with α selected on the same validation slice. Purpose: to test whether the hidden states add information beyond word statistics.

- Hidden states beat TF-IDF on every dataset: +6.2 points on l_avia, +2.7 on s_avia, +1.4 on cqad20, +3.8 on cqad20_fine and +6.1 on cqad20_answer.
- The gain from depth is largest for fine-grained experts. On cqad20_answer, layer 0 is about equal to TF-IDF (0.870 vs 0.862), and the chosen deep layer adds about 5 points.
- Even TF-IDF + LDA beats the fine-tuned 1B classifier router everywhere except cqad20. The improvement therefore has **two** sources: (i) the closed-form shrunk-covariance classifier with validation-based selection, instead of a fine-tuned classification head; and (ii) contextual hidden states instead of lexical features. Report both.

Outputs: `results/<ds>/seed_<s>/hidden_router/tfidf_baseline.json` and `results/hidden_router/tfidf_summary.{csv,md}`.

### How the Mahalanobis-SLG answers were produced (important for the methods section)

Generation with SLG is deterministic for a given (question, expert) pair:
- one expert answers;
- greedy decoding;
- the same prompt and generation code, `SmallLanguageGraph._tuned_generate_with_confidence`.

So the answer of expert E to question q is the same whichever SLG run produced it. The final SLG answers (`answers/slg_mahalanobis_router.jsonl`) were therefore assembled by `python -m hidden_router assemble`. For each test question, the Mahalanobis router picks an expert, and the answer is taken as follows:

| Router's choice | Answer used |
|---|---|
| The gold expert | The oracle-run answer |
| The classifier router's choice | The SLG-classifier answer |
| The wrong-routing counterfactual's expert | The wrong-routing answer |
| Any other expert | Generated now with that expert, through the same code path |

Each record stores `answer_source`. The premise is verified on real data before assembly: for questions where the classifier router chose the gold expert, its answer must be identical to the oracle answer. The assembly stops if fewer than 95% are identical; the results are in `hidden_router/reuse_check.json`. The share of answers generated fresh (expected about 3–5% from the probe) is in `hidden_router/assembly_report.json` and in the `share_generated` column of `results/stats/routing.csv`.

This is equivalent to running SLG end to end with the Mahalanobis router (`python main.py --bundle` runs it as stage `infer_slg_mahalanobis`, which fits the router at train time if it is missing). For the paper, state that answers are deterministic given (question, expert) and that the reuse identity was checked; quote the identical share from `reuse_check.json`.

Latency caveat: copied answers keep the latency and peak memory recorded by their source run. Those runs use the same expert generation path. They exclude the Mahalanobis routing step, which costs one 1B forward pass over the question (tens of milliseconds, small next to generation of up to 750 tokens). The classifier router's own latency was included in the SLG-classifier records. Mention this when reporting the deployment table.

### Estimated end-task effect (probe estimate; final numbers come from evaluation)

Before generating anything, the probe estimated SLG quality per question. Where an answer from the chosen expert already existed, it used that answer's exact score. For the remaining 0.1–5% of questions, it used the wrong-routing score as a proxy. Values are means over seeds (ROUGE-L / AI_Expert). The baselines are measured values.

| Dataset | SLG-classifier | SLG-Mahalanobis (est.) | 8B | 1B | RAG |
|---|---|---|---|---|---|
| l_avia | 0.788 / 0.807 | 0.917 / 0.928 | 0.885 / 0.903 | 0.640 / 0.636 | 0.429 / 0.925 |
| s_avia | 0.877 / 0.894 | 0.934 / 0.947 | 0.922 / 0.927 | 0.824 / 0.835 | 0.454 / 0.936 |
| s_avia_25 | 0.875 / 0.878 | 0.930 / 0.936 | 0.922 / 0.920 | 0.794 / 0.785 | 0.438 / 0.875 |
| s_avia_50 | 0.867 / 0.854 | 0.935 / 0.929 | 0.916 / 0.906 | 0.783 / 0.756 | 0.430 / 0.828 |
| s_avia_75 | 0.879 / 0.859 | 0.935 / 0.925 | 0.912 / 0.898 | 0.763 / 0.720 | 0.417 / 0.792 |
| s_avia_100 | 0.877 / 0.856 | 0.933 / 0.919 | 0.887 / 0.863 | 0.716 / 0.650 | 0.410 / 0.763 |
| cqad20 | 0.611 / 0.615 | 0.613 / 0.617 | 0.736 / 0.759 | 0.663 / 0.686 | 0.181 / 0.427 |
| cqad20_fine | 0.557 / 0.589 | 0.647 / 0.678 | 0.736 / 0.759 | 0.663 / 0.686 | 0.181 / 0.427 |
| cqad20_answer | 0.674 / 0.702 | 0.824 / 0.865 | 0.736 / 0.759 | 0.663 / 0.686 | 0.181 / 0.427 |

These are planning estimates only. Use `results/stats/paper_quality.csv` and the paired bootstrap after evaluation for every number in the paper.

## Evaluation and paper analyses

Each full bundle covers one dataset and one seed. It trains the 1B and 8B baselines plus SLG, then runs inference for 1B, 8B, SLG, and local RAG. It also runs oracle and wrong-expert router inference. Evaluation is a separate login-node step because the Rorqual compute nodes cannot reach the OpenAI API. On the login node, `bash evaluate_all.sh` evaluates every available model answer file for all 9 datasets and 3 seeds (answer files that already have a complete evaluation checkpoint are skipped, so only new files such as `slg_mahalanobis_router.jsonl` are scored); `--dataset` and `--seed` can restrict it. The API key is required only for this scoring step and is read from `OPENAI_API_KEY`.

Evaluation retains per-question score records keyed by `qid`; the eight reported metrics are BLEU, ROUGE-1/2/L, exact match, METEOR, Entailment, and AI_Expert. BLEU uses whitespace tokenization and smoothing method 4. ROUGE uses stemming and F-measures. Exact match compares stripped strings. METEOR uses NLTK tokenization and WordNet resources. “Entailment” is operationalized as cosine similarity between OpenAI `text-embedding-3-small` embeddings, transformed from [-1, 1] to [0, 1]; it is an embedding-similarity proxy rather than a textual entailment classifier. AI_Expert is a binary GPT-4.1-nano judgment using the dataset-specific judge prompt (aerospace-specific for Avia, domain-neutral for CQAD). The API key is read from `OPENAI_API_KEY`; API-based evaluation runs on the login node because the compute nodes used for bundle jobs have no network access.

The paired significance analysis reads only per-question score records and checks that the four primary models have matching qid sets in each cell. For each dataset and each of the eight metrics, it compares SLG (Mahalanobis router) with 8B, 1B, RAG, SLG-classifier (router ablation) and SLG-oracle (remaining routing gap); all six systems must have identical qid sets. In each of 10,000 bootstrap replicates it samples answer groups with replacement independently within each seed, computes a question-weighted mean difference for that seed, then averages the three seed means with equal seed weight. It reports the observed mean difference, a percentile 95% interval, a two-sided sign-tail bootstrap p-value, and Holm-adjusted p-values. The bootstrap's fixed RNG seed is 20261015. The comparison of SLG with 8B is treated as equivalence only for ROUGE-L and AI_Expert, using a ±0.01 interval criterion; other metrics have no equivalence declaration. The output includes CSV and LaTeX files.

For `s_avia_100`, the same analysis additionally computes difference-in-differences against `s_avia_25`: `(SLG_100 − SLG_25) − (baseline_100 − baseline_25)`, separately for the 8B and 1B baselines and for all eight metrics. It requires identical qid sets across these inflation levels. Holm correction is applied across all rows produced for that dataset, including these inflation-contrast rows (and, for `cqad20_fine`/`cqad20_answer`, the granularity rows below). This is a specific 100%-versus-25% contrast, not a fitted trend over all four inflation levels.

For `cqad20_fine` and `cqad20_answer`, an extra paired comparison `SLG_vs_SLG_cqad20_granularity` (SLG on the variant minus SLG on `cqad20`, same qids, bootstrap RNG seed 20261015 + 2000) measures the effect of expert granularity on fixed data. These variants are included in post-analysis automatically once all three seeds have `scores/slg_mahalanobis_router.jsonl`.

The routing table (`evaluate/routing_table.py`, `results/stats/routing.{csv,tex}`) reports per dataset:
- classifier-router top-1/3/5, TF-IDF top-1/3/5, layer-0 top-1 and Mahalanobis top-1/3/5;
- the chosen layers and α, and the best test layer (diagnostic only);
- the share of freshly generated answers;
- **routing gap closed** = (SLG − SLG-classifier) / (SLG-oracle − SLG-classifier) for ROUGE-L and AI_Expert. This is undefined (left blank) when the classifier is already within 0.01 of the oracle, as on cqad20.

`results/stats/routing_layers.csv` gives the dataset × layer test top-1/top-5 curve (mean over seeds) for a layer figure.

The generated paper quality table first averages question-level scores within each seed and then reports the arithmetic mean and sample standard deviation (`ddof=1`) of the three seed-level means. The deployment table aggregates per-question latency and peak allocated GPU memory from answer records and training GPU-hours from metrics logs. Its current “minimum GPU tested” field is hard-coded to the job resource label (H100 3g.40gb); it is not an empirically estimated minimum hardware requirement. RAG has no fine-tuning GPU-hours. SLG's training GPU-hours count only the experts: the Mahalanobis router is a closed-form fit of seconds. SLG-classifier also counts the classifier router's LoRA training. The deployment table now includes `cqad20_fine` and `cqad20_answer`. The SLG parameter-size label refers to the active expert, not the total collection of stored experts plus router.

For the full default matrix, `./submit_all.sh` submits 21 independent train/inference bundle jobs (7 datasets × 3 seeds), named `slg-<dataset>-s<seed>` in Slurm; the granularity variants are submitted separately with `./submit_all.sh --datasets cqad20_fine,cqad20_answer` after `./reuse_baselines.sh`. Once they finish, run `./evaluate_all.sh` on a login node. After evaluation has created the per-question score files, submit the paired significance, paper/deployment/routing tables, and activation patching with `sbatch --job-name=slg-post-analysis job.sh post_analysis`. `submit_all.sh` does not submit evaluation or post-analysis automatically. The `--slg_all` ablation and `--scalability` options currently stop with an explicit “runner not included” error; they are not part of the default experiment matrix.

Activation patching studies what the **experts** know, independent of routing, and therefore still selects questions with the SLG-classifier score file (`slg_finetuned_router.jsonl`); switching it would invalidate completed patching results without changing what it measures. It selects at most 200 test questions per cell with 1B ROUGE-L ≤ 0.3 and SLG ROUGE-L ≥ 0.8, taking the first qualifying qids in sorted-qid order. For `s_avia_25`, `s_avia_50`, and `s_avia_75`, it instead reuses the qualifying qids selected at `s_avia_100` for the same seed, enabling a paired inflation comparison. At each of the 16 Llama-3.2-1B layers, it replaces the baseline model's prompt-position residual stream with the corresponding residual from the title-matched SLG expert, and computes teacher-forced log probability for up to the first 16 gold-answer tokens. A control patches residuals from another test question, matching prompt length when possible and otherwise truncating or padding positions. Restoration is `(patched log-probability − baseline log-probability) / (SLG log-probability − baseline log-probability)`; examples with a near-zero denominator are skipped. The plot reports a 95% percentile interval from 2,000 bootstrap resamples of selected questions per layer. This is a selected-subset mechanistic analysis, not a test-set average effect. The full result is generated during post-analysis; the smoke-test path patches one forced example only to verify execution, not to produce a paper result. The planning estimate is about one GPU-hour per cell, but actual runtime depends on qualifying examples and hardware.

## Claims and advantages to make in the paper

Each claim is listed with the evidence that supports it. All claims are subject to the final evaluated numbers.

1. **A cheap, training-free router fixes SLG's main bottleneck.** The classifier router was the weak component: top-1 0.77 on l_avia and 0.74–0.77 on fine CQAD, with high variance across seeds. The Mahalanobis router reaches 0.92–0.96 top-1 and ≥ 0.99 top-5 with a closed-form fit of seconds (routing table; probe).
2. **No test-time or test-set tuning.** The layer and shrinkage are selected on a validation slice of the training data only. No activation patching or labelled test data is needed to deploy.
3. **Hidden states carry routing information beyond lexical cues.** The method beats the same classifier on TF-IDF features on all datasets, by +1.4 to +6.2 points (lexical baseline).
4. **The layer adapts to expert granularity.** Early layers win for coarse topics and late layers for fine-grained experts (11 → 24 → 72 experts: middle → late → last layers). This is an interpretable, mechanistically motivated finding. It connects to the activation-patching analysis of where expert knowledge is used: discuss qualitatively, without claiming causality.
5. **SLG with small experts matches or beats much larger models at lower active cost.** On every aviation dataset, the estimate for SLG (1B active parameters) exceeds the fine-tuned 8B on ROUGE-L and AI_Expert, and far exceeds 1B and RAG. On cqad20_answer it exceeds 8B as well. To be confirmed by the paired bootstrap and Holm-corrected p-values (`SLG_vs_8B`, `SLG_vs_1B`, `SLG_vs_RAG`).
6. **Robust to answer inflation.** On s_avia_100 the 8B baseline degrades (RL 0.922 → 0.887 from s_avia to s_avia_100) and 1B degrades more, while SLG stays at about 0.93. Evidence: the difference-in-differences rows `SLG_vs_8B_inflation_DiD_100v25` and `SLG_vs_1B_inflation_DiD_100v25`.
7. **Modularity: finer experts become usable.** With the classifier router, finer partitions hurt because routing collapsed. With the Mahalanobis router, quality rises with granularity on fixed data (cqad20 → cqad20_answer: est. RL 0.613 → 0.824). Evidence: the `SLG_vs_SLG_cqad20_granularity` rows.
8. **Router ablation.** `SLG_vs_SLG-classifier` (paired, Holm) isolates the router's contribution with identical experts. `SLG_vs_SLG-oracle` and the gap-closed ratio show how much routing headroom remains.
9. **Easy to add new knowledge (design argument).** Adding an expert needs only a LoRA adapter for the new document plus the mean feature vector of its training questions. The shared covariance can be updated from sufficient statistics, so the other experts and the router are never retrained. The classifier router would need retraining with a new output class. *Not yet measured empirically; state it as a property of the method, or run a held-out-expert addition experiment before claiming measured numbers.*
10. **Deployable on small hardware.** The router and one active 1B expert run on a 10 GB MIG slice (probe and assembly ran on H100 1g.10gb). Compare peak memory and latency in the deployment table.

Related work to position against: mixture-of-LoRA routing such as LoRAHub, MoLE, Arrow (LoRA-weight-based routing), PHATGOOSE and other zero-shot expert routing; Mahalanobis or Gaussian discriminant classifiers on frozen LM features (e.g., the OOD-detection literature); and classic LDA with shrinkage covariance (Ledoit–Wolf style). The novelty argument is the combination: a training-free, closed-form discriminant on the frozen base model's own hidden states as the router for a library of small domain experts, with data-driven layer selection that tracks expert granularity.

## Caveats and expected reviewer questions

- **cqad20_answer has one answer cluster per expert.** Each expert is close to memorising one answer, so a reviewer may call it retrieval rather than expertise. Present it as the far end of the granularity series and not as the headline result. The headline results are the aviation datasets and the 11/24-expert CQAD.
- **cqad20 (11 experts).** The router was already near perfect (0.994). SLG stays below 8B there because the experts themselves are weak: oracle RL ≈ 0.615 vs 8B 0.736. Routing cannot fix this; report it honestly as the limit of coarse experts.
- **Answer assembly by reuse** (see the section above). Describe it, quote `reuse_check.json`, and note that the full pipeline produces the same answers end to end.
- **Estimated vs measured.** All "est." numbers in this note are probe estimates. The paper must use the evaluated `slg_mahalanobis_router` scores.
- **Layer curve on aviation is flat.** Do not overclaim layer specificity for aviation. Claim it for the granularity trend.
- **Copied answers keep their source run's latency** and exclude the ~tens-of-ms routing forward pass (see above).
- **α = 0.05,** the smallest grid value, was chosen in most cells. A smaller α might gain slightly. The grid is fixed in `config.yaml` (`hidden_router.shrinkage_grid`).

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

Mahalanobis-router workflow, as run for the paper after the bundles had finished with the classifier router:

```bash
# Steps 1-4: features (cached), fit and test routers per cell, compare with the classifier router, estimate scores
sbatch --account=rrg-adml2021 --time=01:30:00 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --mem=32G job.sh hidden_router probe
# Lexical baseline (CPU)
python -m hidden_router tfidf
# Steps 5-6: build answers/slg_mahalanobis_router.jsonl per dataset (reuse check, then copy or generate)
sbatch ... job.sh hidden_router assemble --datasets <ds>
# Then evaluate (login node) and post-analysis as above
```

Outputs: `results/hidden_router/{probe_summary,tfidf_summary}.{csv,md}` and per cell `results/<ds>/seed_<s>/hidden_router/{router.pt, selection.json, test_routes.json, per_layer_test.json, tfidf_baseline.json, reuse_check.json, assembly_report.json}`.

A fresh full run (`./submit_all.sh`) fits the Mahalanobis router during training (`training_components.train_hidden_router`) and runs the bundle stage `infer_slg_mahalanobis`; no assembly step is needed then.

To evaluate only a smoke-test cell after its compute job has completed, use `./evaluate_all.sh --dataset s_avia_100 --seed 42 --smoke`. The `--smoke` mode reads the separate `results/smoke/` answers and `data/_smoke/` test split. It still requires `OPENAI_API_KEY` on the login node. Paired significance testing is not meaningful for this single-cell smoke run.

The dataset builder is deterministic and skips existing outputs unless `--force` is passed. Jobs require prepared split files and do not build or split data themselves.

## Quick pipeline smoke run

`bash smoke_test.sh [dataset] [seed]` submits one isolated Slurm validation job named `slg-smoke-<dataset>-s<seed>`. It defaults to `s_avia_100`, seed 42. The job selects two section titles, 5 train questions per title, and up to 2 test questions per title, then runs both baseline fine-tunes, two SLG experts and the router, four standard inference models, oracle/wrong routing, and one activation-patching example. Smoke artifacts go under `data/_smoke/` and `results/smoke/`; they do not overwrite final experiment cells. The patching output only checks that the code path executes; it is not a paper result. API evaluation is not run in the compute job; use the login-node evaluation workflow for full results. Model weights must already be downloaded.
