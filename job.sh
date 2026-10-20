#!/bin/bash
#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time 48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --signal=B:USR1@600
#SBATCH --export=ALL

set -euo pipefail
if [[ $# -eq 1 && "$1" == "post_analysis" ]]; then
  job_mode=post_analysis
elif [[ $# -eq 3 && "$1" == "smoke_test" ]]; then
  job_mode=smoke_test
elif [[ $# -eq 2 ]]; then
  job_mode=bundle
else
  echo "Usage: sbatch job.sh DATASET SEED | sbatch job.sh smoke_test DATASET SEED | sbatch job.sh post_analysis" >&2
  exit 2
fi
module load python/3.11.5
module load rust
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow
source ENV/bin/activate

trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
if [[ "$job_mode" == post_analysis ]]; then
  python main.py --post_analysis
elif [[ "$job_mode" == smoke_test ]]; then
  python -m commands.smoke_test "$2" "$3"
else
  python main.py --bundle --dataset "$1" --seed "$2"
fi
