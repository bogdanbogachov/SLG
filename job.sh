#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-adml2021
#SBATCH --time=0-00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
# ^ Multi-GPU: the pipeline auto-scales to however many GPUs this job is given.
#   Independent jobs (each LoRA expert, each ablation, each scalability size) are
#   dispatched one-per-GPU, and each job also batches its own generation to fill
#   that GPU (~75% of 80GB). Keep --ntasks=1 and --nodes=1 (parallelism is
#   in-process, not srun tasks; all GPUs must be on one node). If you change the
#   GPU count, scale with it: ~4-5 CPUs/GPU and ~40G/GPU.
#   NOTE: request *full* GPUs (--gpus-per-node=h100:N), not a MIG slice like
#   h100_3g.40gb — a single MIG slice is one device and cannot be parallelised.
#   RORQUAL: GPU nodes are Dell XE8640 = 4x H100 SXM5 80GB (324 GPUs / 81 nodes),
#   so 4 is the per-node max — h100:5 would never schedule. Node also has 64
#   cores / ~512G RAM, so cpus-per-task=24 and mem=200G are within one node.
#
#   TIME: on 4 GPUs the 4 *coupled* ablations run one-per-GPU (~20h), then the
#   `base` ablation is sharded data-parallel across all 4 GPUs (~5h) instead of
#   running solo -> ablation phase ~25h. Baseline inference (ask_finetuned) is
#   batched (~14h). Estimated total ~2.5 days, so 4 days has comfortable margin.

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
python main.py --finetune=True --infer_finetuned=True --train_limit=1200 --limit=100
