#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G           # Total memory for the job, not per CPU
#SBATCH --gpus-per-node=1

module load python/3.11.5
module load rust
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

# Activate venv
source venv/bin/activate

# Export a dummy variable for Open AI API
export OPENAI_API_KEY="dummy"

# Run the Python script
python main.py --finetune=True --infer_finetuned=True --infer_slg=True
