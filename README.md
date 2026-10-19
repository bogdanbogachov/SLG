# SLG

A fine-tuning and inference pipeline optimized for resource-constrained environments, capable of running on small GPUs (up to 4 GB VRAM) with distributed AI potential. The system implements **SLG (Small Language Graph)**, a multi-expert architecture where each expert is fine-tuned using LoRA adapters.
## Requirements

- Python 3.10+
- CUDA-compatible GPU
- Environment variable for login-node evaluation: `OPENAI_API_KEY`

## Installation

```bash
# Match the Python module and environment directory used by job.sh
module load python/3.11.5
uv venv --python "$(command -v python)" --seed ENV
uv pip sync --python ENV/bin/python --index-url https://pypi.org/simple requirements.txt
```

## Configuration

1. Set model names and hyperparameters in `config.yaml`.
2. Export `OPENAI_API_KEY` before running evaluation on a login node; compute jobs do not call OpenAI:
   ```bash
   export OPENAI_API_KEY='your-key'
   ```
   Model downloads and dataset construction do not require this key.

## Usage

### Full experiment commands

Run these from the project directory on the login node. This assumes the environment, model weights, and prepared datasets are already in place. The loop submits 21 separate Slurm compute jobs. Wait for them to finish before evaluating: Rorqual compute nodes cannot reach the OpenAI API, so evaluation runs directly on the login node.

```bash
# Submit all 7 datasets × all 3 seeds (21 separate Slurm jobs)
./submit_all.sh

# After all 21 jobs have completed, evaluate all answer files on the login node
export OPENAI_API_KEY='your-actual-key'
./evaluate_all.sh

# After evaluation completes, submit paired statistics, paper tables, and activation patching
sbatch job.sh post_analysis
```

`submit_all.sh` is a shell script that runs the `sbatch job.sh <dataset> <seed>` command once for each dataset and seed. Run it directly with `./submit_all.sh` on the login node; do not prefix it with `sbatch`. Slurm runs each submitted `job.sh` as a separate compute job.

## CQADupStack Clustered QA Dataset

The min-20 clustered CQADupStack source used by dataset construction is:

```text
question_answer/cqadupstack_clustered/qa_min20.json
```

It was created by combining BEIR CQADupStack with StackExchange answer dumps:

1. Download BEIR CQADupStack, which provides question text and semantically similar question links in `qrels/test.tsv`.
2. Build semantically similar question clusters from those links.
3. Download the matching StackExchange `Posts.xml` dump for each CQADupStack domain.
4. For each cluster, select one canonical answer from StackExchange, preferring the accepted answer and falling back to the highest-scored answer.
5. Write one QA row per question variant, with all questions in the cluster sharing the selected answer.

Each row follows the local QA schema:

```json
{
  "chapter": "CQADupStack - physics",
  "title": "CQADupStack - physics",
  "question": "semantically similar question variant",
  "answer": "selected StackExchange answer"
}
```

For CQADupStack rows, `title` intentionally mirrors `chapter` so title-based
splitting uses the CQADupStack domain as the class label. The original cluster
source title is preserved in `clusters.csv`.

The min-20 source contains:

```text
3,435 QA rows
72 answer/question clusters
at least 20 questions per answer
```

See [docs/CQADupStack.md](docs/CQADupStack.md) for download and conversion provenance, exact commands, and cleanup guidance. The broader version with at least 4 questions per answer is preserved at:

```text
question_answer/cqadupstack_clustered/qa_min4_full.json
```

## Project Structure

```
eng_llm/
├── main.py                # Entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
├── commands/              # Command handlers
│   ├── data_processing.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── models.py
├── question_answer/       # QA generation from PDFs
├── finetune/              # LoRA fine-tuning
├── inference/             # Baseline, RAG, SLG inference
├── evaluate/              # Evaluation metrics
├── utils/                 # Model loading, paths, prompts
└── download_llama/        # Model download utilities
```

## Key Features

- **LoRA Fine-tuning**: Efficient fine-tuning of LLaMA 3.2-1B and 3.1-8B models
- **RAG**: Retrieval-augmented generation with FAISS vector search
- **SLG**: Small Language Graph with expert routing and multi-model inference
- **SLG router selection**: cosine similarity routing or a fine-tuned LLaMA 3.2 1B classifier router
- **Evaluation**: ROUGE, METEOR, Exact Match, semantic similarity, AI Expert

## Output Structure

```
experiments/
└── {experiment_name}/
    ├── finetuned_3_2_1b/       # Fine-tuned adapter
    ├── finetuned_3_1_8b/       # Fine-tuned adapter
    ├── slg_router_3_2_1b/      # Fine-tuned SLG classifier router adapter
    ├── slg/                    # SLG expert adapters
    └── metrics.json            # Evaluation results

answers/
└── {experiment_name}/
    ├── gpt-4.1-2025-04-14.json
    ├── rag.json
    ├── finetuned_3_2_1b.json
    ├── finetuned_3_1_8b.json
    ├── slg.json
    ├── slg_finetuned_router.json
    └── routing_reports/
        ├── slg_routing_report.json
        └── slg_finetuned_router_routing_report.json
```

SLG routing reports include per-question expected expert, selected expert, correctness,
visited experts, candidate answer confidences, and aggregate routing accuracy. Fine-tuned
router training logs `eval_accuracy` and final `test_accuracy` on `qa_test`; SLG inference
also logs and stores full routing reports. The report includes confusion matrices, top-k
routing accuracy, per-chapter and per-expert accuracy, Wilson confidence intervals,
answer-quality summaries by routing correctness, high-confidence error examples,
latency summaries, and router comparison statistics when both cosine and fine-tuned
router reports exist.
