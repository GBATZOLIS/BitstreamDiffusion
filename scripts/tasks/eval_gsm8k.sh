#!/usr/bin/env bash
# Evaluate GSM8K executable-code accuracy. CKPT required.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
: "${CKPT:?set CKPT to a checkpoint .pt path}"
SAMPLER="${SAMPLER:-stochastic}"
GAMMA="${GAMMA:-0.0}"
STEPS="${STEPS:-1024}"
LIMIT="${LIMIT:-1319}"

python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint "${CKPT}" \
  --sampler "${SAMPLER}" --gamma "${GAMMA}" --steps "${STEPS}" --limit "${LIMIT}"
