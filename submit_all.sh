#!/usr/bin/env bash
set -euo pipefail
datasets=(l_avia s_avia s_avia_25 s_avia_50 s_avia_75 s_avia_100 cqad20)
seeds=(42 53 64)
while (($#)); do
  case "$1" in
    --datasets) IFS=',' read -r -a datasets <<< "$2"; shift 2 ;;
    --seeds) IFS=',' read -r -a seeds <<< "$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    sbatch --parsable job.sh "$dataset" "$seed"
  done
done
echo "Submitted bundle jobs. After they finish, run 'bash evaluate_all.sh' on a login node; then submit post-analysis with 'sbatch job.sh post_analysis'."
