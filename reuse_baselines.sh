#!/usr/bin/env bash
# Copy the 1B/8B adapters and the 1B/8B/RAG answers from one cell into another with the same test split,
# so that a bundle job for DST only trains and runs SLG. Usage: ./reuse_baselines.sh [SRC] [DST] [SEEDS]
# Default: ./reuse_baselines.sh cqad20 cqad20_fine 42,53,64
set -euo pipefail
src=${1:-cqad20}; dst=${2:-cqad20_fine}; IFS=',' read -r -a seeds <<< "${3:-42,53,64}"
for seed in "${seeds[@]}"; do
  s="results/$src/seed_$seed"; d="results/$dst/seed_$seed"
  cmp -s "data/$src/splits/seed_$seed/test.json" "data/$dst/splits/seed_$seed/test.json" \
    || python - "$src" "$dst" "$seed" <<'PY'
import json, sys
src, dst, seed = sys.argv[1:]
q = lambda d: [r["qid"] for r in json.load(open(f"data/{d}/splits/seed_{seed}/test.json"))]
sys.exit(0 if q(src) == q(dst) else f"Test qids differ between {src} and {dst} for seed {seed}")
PY
  mkdir -p "$d/answers"
  for adapter in finetuned_3_2_1b finetuned_3_1_8b; do
    [[ -f "$s/$adapter/.done" && -f "$s/$adapter/adapter_config.json" ]] || { echo "Missing finished adapter $s/$adapter" >&2; exit 1; }
    rm -rf "${d:?}/$adapter"; cp -r "$s/$adapter" "$d/$adapter"
  done
  for answers in finetuned_3_2_1b finetuned_3_1_8b rag; do
    cp "$s/answers/$answers.jsonl" "$d/answers/$answers.jsonl"
  done
  echo "seed $seed: copied 1B/8B adapters and 1B/8B/RAG answers from $s to $d"
done
