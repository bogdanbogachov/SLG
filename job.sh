#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=4
# ^ Multi-GPU: the pipeline auto-scales to however many GPUs this job is given.
#   Independent jobs (each LoRA expert, each ablation, each scalability size) are
#   dispatched one-per-GPU, so raising --gpus speeds the run up with NO code
#   change. Keep --ntasks=1 (parallelism is in-process, not srun tasks); scale
#   --cpus-per-task (~4/GPU) and --mem (~32G/GPU) with the GPU count.
#   NOTE: request *full* GPUs (--gpus=N or --gpus=h100:N), not a MIG slice like
#   h100_3g.40gb — a single MIG slice is one device and cannot be parallelised.

module load python/3.11.5
module load rust
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

# Activate venv
source ENV/bin/activate

# Export a dummy variable for Open AI API
export OPENAI_API_KEY="dummy"

# Keep CPU thread pools from oversubscribing when several GPU workers run at once.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# To force the old single-GPU sequential behaviour (debugging), set:
#   export SLG_DISABLE_PARALLEL=1

# Run the full pipeline:
#   --slg_descriptions  expert descriptions (prereq)
#   --finetune          LoRA experts (+ 3_2_1b / 3_1_8b baselines per config)
#   --infer_finetuned   single fine-tuned LLaMA baseline (#1)
#   --slg_all           SLG suite: ablations (#2, incl. the full run) -> scalability (#5)
#                       -> metrics (#3,#4)
# Cloud baselines (--infer_baseline/--infer_rag), --evaluate and --paper_assets are
# omitted: --evaluate needs a real OPENAI_API_KEY (the export below is a dummy).
# On the login node with a real key, run `python main.py --evaluate` then
# `python main.py --paper_assets` to produce the LaTeX tables + figures.
python main.py --slg_descriptions=True --finetune=True --infer_finetuned=True --slg_all=True
