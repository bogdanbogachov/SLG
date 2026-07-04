#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gpus-per-node=h100:5
# ^ Multi-GPU: the pipeline auto-scales to however many GPUs this job is given.
#   Independent jobs (each LoRA expert, each ablation, each scalability size) are
#   dispatched one-per-GPU, and each job also batches its own generation to fill
#   that GPU (~75% of 80GB). Keep --ntasks=1 and --nodes=1 (parallelism is
#   in-process, not srun tasks; all 5 GPUs must be on one node). If you change
#   the GPU count, scale with it: ~4-5 CPUs/GPU and ~40G/GPU.
#   NOTE: request *full* GPUs (--gpus-per-node=h100:N), not a MIG slice like
#   h100_3g.40gb — a single MIG slice is one device and cannot be parallelised.
#
#   TIME: 5 days is sized for the full DS with the fine-tuned *baseline*
#   inference (ask_finetuned) still running one-question-at-a-time — the 8B
#   baseline over 12k test Qs is ~2 days on its own and is the current pole.
#   Estimated total ~3.8 days; 5 days leaves margin for the batching-speedup
#   uncertainty and any restart. If ask_finetuned is batched too, this drops to
#   ~2.3 days and --time=3-00:00:00 is enough.

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
