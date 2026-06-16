#!/usr/bin/env bash
# Evaluate Sudoku exact-match. CKPT required. Single GPU is fine.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SUDOKU_DIFFICULTY="${SUDOKU_DIFFICULTY:-easy}"
export TOKENIZERS_PARALLELISM=false
: "${CKPT:?set CKPT to a checkpoint .pt path}"
SAMPLER="${SAMPLER:-stochastic}"      # stochastic (headline) | deterministic
GAMMA="${GAMMA:-0.0}"
STEPS="${STEPS:-180}"
LIMIT="${LIMIT:-2000}"

python -m evaluation.tasks.sudoku_eval \
  --config configs/tasks/sudoku_bits.py \
  --checkpoint "${CKPT}" \
  --difficulty "${SUDOKU_DIFFICULTY}" \
  --sampler "${SAMPLER}" --gamma "${GAMMA}" --steps "${STEPS}" --limit "${LIMIT}"
