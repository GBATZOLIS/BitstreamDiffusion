#!/usr/bin/env bash
# FKC beta-ceiling probe on GSM8K (non-CFG 425k checkpoint).
#
# Purpose: find the largest beta whose ESS does NOT collapse, BEFORE spending a
# full sweep. The FKC log-weight is extensive in the number of free bits, and
# GSM8K has ~7168 (20x Sudoku's 356), so Sudoku's usable ceiling of beta~1.05
# should scale down to roughly beta~1.0025 here. This probe brackets that.
#
# Read min_ess (target: close to K=16; collapse => 1) and total_resample_events.
# beta=1.0 is the control: K independent churn samples + voting, uniform weights.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-$HOME/miniconda3/envs/pytorch/bin/python}
CKPT=${CKPT:-runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000425000.pt}
OUT=${OUT:-runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/probe}
K=${K:-16}
LIMIT=${LIMIT:-8}
STEPS=${STEPS:-1024}
BATCH=${BATCH:-4}          # effective forward batch = K*BATCH = 64, matches baseline memory
GAMMA=${GAMMA:-0.41}       # best plain-sampling churn for this checkpoint
BETAS=${BETAS:-"1.0 1.0005 1.001 1.002 1.005"}

mkdir -p "$OUT"
for B in $BETAS; do
  echo "############ beta=$B  K=$K  limit=$LIMIT  steps=$STEPS ############"
  "$PY" -m evaluation.tasks.gsm8k_eval \
    --config configs/tasks/tinygsm_bits.py \
    --checkpoint "$CKPT" \
    --sampler_kind fkc_em --proposal edm_churn --churn_gamma "$GAMMA" \
    --beta "$B" --num_particles "$K" \
    --ess_threshold 0.5 --resampling_policy ess --sc_policy inherit --final_resample 1 \
    --steps "$STEPS" --limit "$LIMIT" --batch_size "$BATCH" \
    --ema 1 --seed 42 --sigma_data 0.399844765663147 \
    --out_dir "$OUT" || echo "FAILED beta=$B"
done
echo "############ probe done ############"
