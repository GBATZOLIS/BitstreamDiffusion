#!/usr/bin/env bash
# Evaluate Sudoku exact-match. CKPT required. Single GPU is fine.
# Headline = DDIM 'ddim_entropic' path with EDM-style churn (SAMPLER=stochastic, GAMMA>0).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SUDOKU_DIFFICULTY="${SUDOKU_DIFFICULTY:-easy}"
export TOKENIZERS_PARALLELISM=false
: "${CKPT:?set CKPT to a checkpoint .pt path}"
SAMPLER="${SAMPLER:-stochastic}"        # stochastic (EDM churn) | deterministic
SAMPLER_KIND="${SAMPLER_KIND:-ddim}"    # ddim (headline) | heun (ablation)
SCHEDULE="${SCHEDULE:-entropic}"        # entropic (trained grid) | karras
GAMMA="${GAMMA:-0.1}"                   # churn strength; 0 => no churn
STEPS="${STEPS:-180}"
LIMIT="${LIMIT:-2000}"

python -m evaluation.tasks.sudoku_eval \
  --config configs/tasks/sudoku_bits.py \
  --checkpoint "${CKPT}" \
  --difficulty "${SUDOKU_DIFFICULTY}" \
  --sampler "${SAMPLER}" --sampler_kind "${SAMPLER_KIND}" --schedule "${SCHEDULE}" \
  --gamma "${GAMMA}" --steps "${STEPS}" --limit "${LIMIT}"
