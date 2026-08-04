#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 MODEL_LABEL GENERATED_FASTA OUTPUT_JSON [basic|full]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
MODEL="$1"
FASTA="$2"
OUT="$3"
PROFILE="${4:-basic}"

METRICS=(basic)
if [ "$PROFILE" = "full" ]; then
  METRICS=(basic esm2 prot_t5 esmfold)
fi

.venv/bin/python -m evaluation.proteins.benchmark \
  --model "$MODEL" --generated-fasta "$FASTA" --out "$OUT" \
  --metrics "${METRICS[@]}"
