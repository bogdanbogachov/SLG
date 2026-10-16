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

### Leakage and overlap summaries

For each seed and dataset, the build summary reports the distribution of each test question's maximum word-Jaccard similarity against training questions with the same original answer. It also reports the fraction at thresholds 0.25, 0.5, 0.75, 0.8, and 1.0, plus mean, median, 90th percentile, 95th percentile, and maximum. Since global exact-question deduplication occurs before splitting, there are no exact repeated questions across the resulting train/test partitions in the current build.

The same summary reports answer-prefix overlap at prefix length 5: UF/TF for the full dataset and mean US/TS across per-title train files. These are build diagnostics; use the generated summary for the final source files when reporting them.

## Evaluation and paper analyses

Each full bundle covers one dataset and one seed. It trains the 1B and 8B baselines plus SLG, then runs inference and evaluation for 1B, 8B, SLG, and local RAG. It also runs oracle and wrong-expert router inference/evaluation for that cell. OpenAI is used by evaluation for Entailment and AI-expert scoring; the API key is supplied through `OPENAI_API_KEY` at job submission. It is not part of dataset building or RAG generation.

Evaluation retains per-question score records keyed by `qid`. The paired significance analysis reads these records, resamples answer groups within each seed, combines the three seed-level estimates, and reports 95% percentile intervals and two-sided bootstrap p-values. Holm correction is applied within dataset. The generated paper quality tables report the sample standard deviation (`ddof=1`) across seeds.

For the full default matrix, `bash submit_all.sh` submits 21 independent bundle jobs (7 datasets × 3 seeds), followed by one Slurm post-analysis job dependent on all 21. That dependent job runs paired significance analysis for every dataset, creates paper quality and deployment tables, and runs activation patching. A restricted dataset/seed submission does not schedule the all-matrix post-analysis job; run the desired analysis explicitly after its inputs exist.

Activation patching selects up to 200 test questions per cell where the 1B baseline has ROUGE-L ≤ 0.3 and SLG has ROUGE-L ≥ 0.8. For the four Avia inflation levels, it selects qids from `s_avia_100` and reuses those qids at each level for the same seed. It measures teacher-forced log probability of the first 16 gold-answer tokens while replacing prompt-position residual states, layer by layer, and includes an unrelated-question control. The analysis uses forward passes only. The documented planning estimate is about one GPU-hour per cell, up to roughly 21 GPU-hours for the full matrix; actual runtime depends on qualifying examples and hardware.

## Reproducibility commands

```bash
python main.py --download_models
python main.py --build_datasets
export OPENAI_API_KEY='...'
bash submit_all.sh
```

The dataset builder is deterministic and skips existing outputs unless `--force` is passed. Jobs require prepared split files and do not build or split data themselves.

## Quick pipeline smoke run

`bash smoke_test.sh [dataset] [seed]` submits one isolated Slurm validation job. It defaults to `s_avia_100`, seed 42. The job selects two section titles, 5 train questions per title, and up to 2 test questions per title, then runs both baseline fine-tunes, two SLG experts and the router, all four standard inference/evaluation models, oracle/wrong routing, one activation-patching example, a reduced 200-replicate paired bootstrap, and the paper table builders. Smoke artifacts go under `data/_smoke/` and `results/smoke/`; they do not overwrite final experiment cells. The bootstrap and one-example patching outputs only check that those code paths execute; they are not paper results. Model weights must already be downloaded, and `OPENAI_API_KEY` must be exported before submitting.
