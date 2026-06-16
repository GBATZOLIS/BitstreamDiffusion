#!/usr/bin/env bash
# Evaluate GSM8K executable-code accuracy. CKPT required.
# Headline = DDIM 'ddim_entropic' path with EDM-style churn (SAMPLER=stochastic, GAMMA>0).
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
: "${CKPT:?set CKPT to a checkpoint .pt path}"
SAMPLER="${SAMPLER:-stochastic}"
SAMPLER_KIND="${SAMPLER_KIND:-ddim}"
SCHEDULE="${SCHEDULE:-entropic}"
GAMMA="${GAMMA:-0.1}"
STEPS="${STEPS:-1024}"
LIMIT="${LIMIT:-1319}"

python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint "${CKPT}" \
  --sampler "${SAMPLER}" --sampler_kind "${SAMPLER_KIND}" --schedule "${SCHEDULE}" \
  --gamma "${GAMMA}" --steps "${STEPS}" --limit "${LIMIT}"
