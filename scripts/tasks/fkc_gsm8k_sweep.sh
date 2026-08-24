#!/usr/bin/env bash
# GSM8K temperature sweeps on the non-CFG 425k checkpoint (reproduces 29.42%).
#
# The question: does temperature scaling improve SINGLE-SAMPLE quality (as low-T
# decoding does for MDLM/Duo, 18% -> ~33-36%), independently of majority voting?
#
# Two distinct mechanisms, two different budgets. Both are read off a metric with
# NO voting in it:
#
#   MODE=beta  FKC tempering, target p^beta, K particles.
#              Metric: particle_mean_accuracy = expected accuracy of ONE draw from
#              the tempered population. beta bites ONLY via resampling: below the
#              ESS cliff (beta<~1.01 here) there are zero resample events and the
#              run is bit-equivalent to beta=1. Costs K x NFE.
#
#   MODE=tau   Track A1 local score-temperature, particle-free, 1x NFE.
#              Metric: accuracy (one sample per problem) -- the budget-matched
#              like-for-like against MDLM/Duo's low-T knob and against our own
#              29.42% baseline.
#
# Paired design: every cell uses the same problems and the same seed, so cells
# differ only in the temperature parameter.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-$HOME/miniconda3/envs/pytorch/bin/python}
CKPT=${CKPT:-runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000425000.pt}
MODE=${MODE:-beta}
LIMIT=${LIMIT:-64}
STEPS=${STEPS:-1024}
SEED=${SEED:-42}
SD=0.399844765663147

if [[ "$MODE" == "beta" ]]; then
  OUT=${OUT:-runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_beta_n${LIMIT}}
  K=${K:-16}
  BATCH=${BATCH:-4}
  GAMMA=${GAMMA:-0.41}
  BETAS=${BETAS:-"1.0 1.01 1.02 1.05"}
  # POLICY/FINALR isolate the FKC CORRECTION from the tempered drift:
  #   ess  + FINALR=1 -> corrected: weights select the population (targets p^beta)
  #   never+ FINALR=0 -> naive: beta-scaled drift only, weights computed but never applied
  # Run both arms on the same betas/seed/problems for a paired A/B.
  POLICY=${POLICY:-ess}
  FINALR=${FINALR:-1}
  mkdir -p "$OUT"
  for B in $BETAS; do
    echo "############ MODE=beta beta=$B K=$K n=$LIMIT policy=$POLICY finalr=$FINALR ############"
    "$PY" -m evaluation.tasks.gsm8k_eval \
      --config configs/tasks/tinygsm_bits.py --checkpoint "$CKPT" \
      --sampler_kind fkc_em --proposal edm_churn --churn_gamma "$GAMMA" \
      --beta "$B" --num_particles "$K" \
      --ess_threshold 0.5 --resampling_policy "$POLICY" --sc_policy inherit \
      --final_resample "$FINALR" \
      --steps "$STEPS" --limit "$LIMIT" --batch_size "$BATCH" \
      --ema 1 --seed "$SEED" --sigma_data "$SD" --out_dir "$OUT" || echo "FAILED beta=$B"
  done
else
  OUT=${OUT:-runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_tau_n${LIMIT}}
  BATCH=${BATCH:-64}
  GAMMA=${GAMMA:-0.41}
  TAUS=${TAUS:-"1.0 0.7 0.5 0.3"}
  mkdir -p "$OUT"
  for T in $TAUS; do
    echo "############ MODE=tau tau=$T n=$LIMIT ############"
    "$PY" -m evaluation.tasks.gsm8k_eval \
      --config configs/tasks/tinygsm_bits.py --checkpoint "$CKPT" \
      --sampler stochastic --gamma "$GAMMA" \
      --score_temp_tau "$T" --score_temp_clean_var 0.25 \
      --steps "$STEPS" --limit "$LIMIT" --batch_size "$BATCH" \
      --ema 1 --seed "$SEED" --sigma_data "$SD" --out_dir "$OUT" || echo "FAILED tau=$T"
  done
fi
echo "############ sweep done (MODE=$MODE) ############"
