#!/usr/bin/env bash
# CoBit Sudoku pass@k / maj@k on the full 2k validation set, all three difficulties.
# beta=1.0 + policy=never + no final resample  =>  K INDEPENDENT churn samples per puzzle
# (no tempering, no selection), which is exactly the multi-sample protocol we want to report.
# Raw (non-EMA) weights per the paper: at 20k steps the EMA has not converged.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/pytorch/bin/python}
K=${K:-16}
LIMIT=${LIMIT:-2000}
STEPS=${STEPS:-180}
BATCH=${BATCH:-16}
for D in ${DIFFS:-easy medium hard}; do
  # paper's best churn: easy/hard gamma=0.4, medium gamma=0.5
  G=0.4; [ "$D" = "medium" ] && G=0.5
  CKPT=runs/tasks/sudoku/$D/cobit_raw_binary_bits/checkpoints/step=000020000.pt
  OUT=runs/tasks/sudoku/$D/cobit_raw_binary_bits/sudoku_passk
  mkdir -p "$OUT"
  echo "######## sudoku $D  K=$K gamma=$G n=$LIMIT ########"
  "$PY" -m evaluation.tasks.sudoku_eval \
    --config configs/tasks/sudoku_bits.py --checkpoint "$CKPT" --difficulty "$D" \
    --sampler_kind fkc_em --proposal edm_churn --churn_gamma "$G" \
    --beta 1.0 --num_particles "$K" --resampling_policy never --final_resample 0 \
    --steps "$STEPS" --limit "$LIMIT" --batch_size "$BATCH" \
    --ema 0 --seed 42 --out_dir "$OUT" || echo "FAILED $D"
done
echo "######## sudoku pass@k done ########"
