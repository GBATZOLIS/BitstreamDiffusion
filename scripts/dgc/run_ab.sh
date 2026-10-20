#!/usr/bin/env bash
# Paired A/B: CoBit-S LM1B, entropy-rate grid vs DGC grid,
# same checkpoint, same seeds, same NFE and gamma. Run from the repo root.
#
#   bash scripts/dgc/run_ab.sh                 # 3 arms x seeds 42 43 44, NFE 256, gamma 0.200
#   SEEDS="42" ARMS="cobit dgc" bash scripts/dgc/run_ab.sh     # quick single-seed check
#   NFES="256,128" bash scripts/dgc/run_ab.sh  # add a lower budget
set -euo pipefail

CONFIG=configs/lm1b/continuous/eval/dgc_schedule_ab.py
ARMS=${ARMS:-"cobit dgc"}
SEEDS=${SEEDS:-"42 43 44"}
export NFES=${NFES:-256}
export GAMMA=${GAMMA:-0.200}
export DET=${DET:-1}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-"$PWD/hf_cache"}

for seed in $SEEDS; do
  for arm in $ARMS; do
    echo "=== SCHEDULE=$arm EVAL_SEED=$seed NFES=$NFES GAMMA=$GAMMA ==="
    export SCHEDULE=$arm EVAL_SEED=$seed
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC:-1} \
      -m evaluation.run_eval --config "$CONFIG" --metrics external_ppl
    python -m evaluation.compute_entropy_from_caches --config "$CONFIG" --include_real
  done
done

python scripts/dgc/summarize_ab.py
