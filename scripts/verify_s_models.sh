#!/bin/bash
# Local verification of the slim EMA-baked CoBit-S checkpoints (OWT + LM1B).
# Runs the two evals CONCURRENTLY, one per GPU (single-process each), then the
# post-hoc entropy estimator. Use after the CoBit-M run frees both GPUs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "${CONDA_ENV:-pytorch}" 2>/dev/null || true
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}"
export EVAL_SEED="${EVAL_SEED:-42}"
mkdir -p logs

run_one () {  # $1=gpu  $2=config  $3=tag
  local gpu="$1" cfg="$2" tag="$3"
  local log="logs/verify_${tag}.log"
  echo "[$tag] GPU$gpu config=$cfg -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    --master_port=$((29200 + gpu)) \
    -m evaluation.run_eval --config "$cfg" --metrics external_ppl > "$log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" python -m evaluation.compute_entropy_from_caches \
    --config "$cfg" --include_real >> "$log" 2>&1
  echo "[$tag] done"
}

run_one 0 configs/owt/eval_750K_seed.py                  s_owt  &
PID_OWT=$!
run_one 1 configs/lm1b/continuous/eval/rate_eval_seeds.py s_lm1b &
PID_LM=$!
wait $PID_OWT; wait $PID_LM
echo "=== both CoBit-S verifications complete ==="
