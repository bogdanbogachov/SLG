#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-s_avia_100}"
seed="${2:-42}"
case "$dataset" in
  l_avia|s_avia|s_avia_25|s_avia_50|s_avia_75|s_avia_100|cqad20) ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
case "$seed" in
  42|53|64) ;;
  *) echo "Seed must be one of: 42, 53, 64" >&2; exit 2 ;;
esac
split_dir="data/$dataset/splits/seed_$seed"
[[ -f "$split_dir/train.json" && -f "$split_dir/test.json" ]] || {
  echo "Prepared split missing at $split_dir; run python main.py --build_datasets first." >&2
  exit 2
}
for model_dir in downloaded_models/downloaded_3_2_1b downloaded_models/downloaded_3_1_8b downloaded_models/jina-v2-base-en; do
  [[ -d "$model_dir" ]] || {
    echo "Downloaded model missing at $model_dir; run python main.py --download_models first." >&2
    exit 2
  }
done
sbatch --parsable --job-name="slg-smoke-${dataset}-s${seed}" job.sh smoke_test "$dataset" "$seed"
