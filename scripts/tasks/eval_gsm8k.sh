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
# Optional EDM preconditioning override. Unset => config default (0.5, OWT-parity).
# Set SIGMA_DATA=0.3998 to match the value the model was actually trained with.
SIGMA_DATA="${SIGMA_DATA:-}"

# Posterior-temperature decoding (continuous analogue of MDLM/Duo low-T).
# PT=1.0 => no-op (untempered headline). PT<1 sharpens the per-bit posterior.
#   PT_TARGET   : learned (recommended) | full
#   PT_SCHEDULE : const | sigma_ramp
#   PT_SIGMA_LO / PT_SIGMA_HI : sigma_ramp band edges
PT="${PT:-1.0}"
PT_TARGET="${PT_TARGET:-learned}"
PT_SCHEDULE="${PT_SCHEDULE:-const}"
PT_SIGMA_LO="${PT_SIGMA_LO:-0.1}"
PT_SIGMA_HI="${PT_SIGMA_HI:-4.0}"
# PT_SPACE: bit (per-bit, factorized) | token (joint valid-codeword, MDLM/Duo analogue).
PT_SPACE="${PT_SPACE:-bit}"
CODEWORD_TOPK="${CODEWORD_TOPK:-}"

python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint "${CKPT}" \
  --sampler "${SAMPLER}" --sampler_kind "${SAMPLER_KIND}" --schedule "${SCHEDULE}" \
  --gamma "${GAMMA}" --steps "${STEPS}" --limit "${LIMIT}" \
  --posterior_temp "${PT}" --posterior_temp_target "${PT_TARGET}" \
  --posterior_temp_schedule "${PT_SCHEDULE}" \
  --posterior_temp_sigma_lo "${PT_SIGMA_LO}" --posterior_temp_sigma_hi "${PT_SIGMA_HI}" \
  --posterior_temp_space "${PT_SPACE}" \
  ${CODEWORD_TOPK:+--codeword_topk "${CODEWORD_TOPK}"} \
  ${SIGMA_DATA:+--sigma_data "${SIGMA_DATA}"}
