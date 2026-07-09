#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:4

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

python main.py --slg_descriptions=True --finetune=True --finetune_router=True --infer_slg=True --limit=100
