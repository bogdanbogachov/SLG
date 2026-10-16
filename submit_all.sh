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
declare -A job_ids=()
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    job_id=$(sbatch --parsable job.sh "$dataset" "$seed")
    job_ids["$dataset:$seed"]="${job_id%%;*}"
  done
done

full_matrix=1
for dataset in l_avia s_avia s_avia_25 s_avia_50 s_avia_75 s_avia_100 cqad20; do
  [[ " ${datasets[*]} " == *" ${dataset} "* ]] || full_matrix=0
done
for seed in 42 53 64; do
  [[ " ${seeds[*]} " == *" ${seed} "* ]] || full_matrix=0
done
if ((full_matrix)); then
    dependency_ids=()
    for dataset in l_avia s_avia s_avia_25 s_avia_50 s_avia_75 s_avia_100 cqad20; do
      for seed in 42 53 64; do
        key="$dataset:$seed"
        dependency_ids+=("${job_ids[$key]}")
      done
    done
    dependency=$(IFS=:; echo "${dependency_ids[*]}")
  sbatch --parsable --dependency="afterok:${dependency}" job.sh post_analysis
else
  echo "Skipping post-analysis: paired statistics and full-matrix paper tables require all 7 datasets × 3 seeds." >&2
fi
