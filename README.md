# eng_llm

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
python main.py --inflate_overshadowing
python main.py --split_qa

# 3. Optional: Check data overlap
python main.py --data_overlap_check

# 4. Fine-tune models
python main.py --finetune

# 5. Run inference
python main.py --infer_baseline       # OpenAI GPT-4.1
python main.py --infer_rag            # RAG with GPT-4.1-nano
python main.py --infer_finetuned      # Fine-tuned LLaMA models
python main.py --infer_slg            # Small Language Graph

# 6. Evaluate results
python main.py --evaluate
python main.py --evaluate --training_metrics  # Include training metrics
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
- **Evaluation**: ROUGE, METEOR, Exact Match, semantic similarity, AI Expert

## Output Structure

```
experiments/
└── {experiment_name}/
    ├── finetuned_3_2_1b/       # Fine-tuned adapter
    ├── finetuned_3_1_8b/       # Fine-tuned adapter
    ├── orchestrator_3_2_1b/    # Orchestrator adapter
    ├── slg/                    # SLG expert adapters
    └── metrics.json            # Evaluation results

answers/
└── {experiment_name}/
    ├── gpt-4.1-2025-04-14.json
    ├── rag.json
    ├── finetuned_3_2_1b.json
    ├── finetuned_3_1_8b.json
    └── slg.json
```
