#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" || "$OPENAI_API_KEY" == "dummy" ]]; then
  echo "Export OPENAI_API_KEY before running login-node evaluation." >&2
  exit 2
fi

module load python/3.11.5
module load rust
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow
source ENV/bin/activate

python main.py --evaluate_all "$@"
