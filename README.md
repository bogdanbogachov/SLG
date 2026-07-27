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

1. Copy `config.yaml` and set paths, model names, and hyperparameters
2. Set environment variables:
   ```bash
   export OPENAI_API_KEY='your-key'
   ```
3. Set experiment name in `config.yaml`: `experiment: 'your_experiment_name'`

## Usage

### Workflow

```bash
# 1. Download models
python main.py --download_models

# 2. Generate QA pairs from PDFs
python main.py --create_qa
python main.py --combine_all_qa
python main.py --inflate_overshadowing --inflation_percentage 25
python main.py --split_qa

# 3. Optional: Check data overlap
python main.py --data_overlap_check

# 4. Fine-tune models
python main.py --finetune
# Enable training_components.train_slg_router in config.yaml to train the SLG classifier router.

# 5. Run inference
python main.py --infer_baseline       # OpenAI GPT-4.1
python main.py --infer_rag            # RAG with GPT-4.1-nano
python main.py --infer_finetuned      # Fine-tuned LLaMA models
python main.py --infer_slg            # Small Language Graph, using routing.method from config.yaml
python main.py --infer_slg --router cosine
python main.py --infer_slg --router finetuned
python main.py --interactive_slg True --router finetuned # Interactive Small Language Graph REPL
python -m interactive_slg             # Same REPL as a package entry point
python -m interactive_slg --router finetuned --question "How should I classify wing damage?"

# 6. Evaluate results
python main.py --evaluate
python main.py --evaluate --training_metrics  # Include training metrics
```

## CQADupStack Clustered QA Dataset

The clustered CQADupStack QA dataset is stored at:

```text
question_answer/cqadupstack_clustered/qa.json
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

Current `qa.json` dataset size:

```text
3,435 QA rows
72 answer/question clusters
at least 20 questions per answer
```

The broader version with at least 4 questions per answer is preserved at:

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
