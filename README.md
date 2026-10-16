# SLG

A fine-tuning and inference pipeline optimized for resource-constrained environments, capable of running on small GPUs (up to 4 GB VRAM) with distributed AI potential. The system implements **SLG (Small Language Graph)**, a multi-expert architecture where each expert is fine-tuned using LoRA adapters.
## Requirements

- Python 3.10+
- CUDA-compatible GPU
- Environment variables: `OPENAI_API_KEY`

## Installation

```bash
# Create virtual environment on Linux
python -m venv venv

# Install dependencies
pip uninstall -y -r <(pip freeze)
pip install --upgrade pip setuptools wheel
pip install --upgrade --upgrade-strategy eager -r requirements.txt
```

## Configuration

1. Set model names and hyperparameters in `config.yaml`.
2. Export `OPENAI_API_KEY` before submitting compute jobs; it is used only for evaluation metrics:
   ```bash
   export OPENAI_API_KEY='your-key'
   ```
   Model downloads and dataset construction do not require this key.

## Usage

### Workflow

```bash
# 1. On the login node, prepare shared model weights and all dataset splits
python main.py --download_models
python main.py --build_datasets

# 2. Export OPENAI_API_KEY, then run a tiny full-pipeline check first
export OPENAI_API_KEY='your-key'
bash smoke_test.sh                 # defaults to s_avia_100, seed 42
# Or: bash smoke_test.sh cqad20 53

# Submit one complete end-to-end bundle
sbatch job.sh l_avia 42

# Submit the default 7 datasets × 3 seeds as 21 independent jobs. Each bundle
# also runs oracle and wrong-router inference/evaluation after the main models.
# A dependent post-analysis job runs paired significance tests and paper tables
# after all 21 bundles finish, followed by activation patching.
# It uses up to 200 selected questions per cell; across the four avia inflation
# levels it reuses the same qids within each seed. Budget about 21 GPU-hours.
bash submit_all.sh

# Restrict the submission matrix
bash submit_all.sh --datasets s_avia_100 --seeds 53

# Run paired, answer-clustered statistics after all three seeds have scores
python main.py --stats --dataset s_avia_100

# Build the paper's latency, memory, and training-cost table after bundle results exist
python main.py --deployment_table

# Build paper quality summaries from per-question scores (means and sample SD across seeds)
python main.py --paper_assets

# Run a router counterfactual without training
python main.py --bundle --dataset s_avia_100 --seed 42 --router_override oracle
```

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
