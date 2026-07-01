#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=h100_3g.40gb:1

module load python/3.11.5
module load rust
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

# Activate venv
source ENV/bin/activate

# Export a dummy variable for Open AI API
export OPENAI_API_KEY="dummy"

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
