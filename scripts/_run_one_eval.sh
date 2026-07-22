#!/bin/bash
# Flat, single-GPU eval runner (no subshells/functions). Args: <gpu> <config>
# Optional env: EVAL_CELLS, EVAL_OUT_SUFFIX (for the CoBit-M config).
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "${CONDA_ENV:-pytorch}" 2>/dev/null || true
fi
: "${USER:=$(id -un)}"; export USER
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}_g$1"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}_g$1"
export EVAL_SEED="${EVAL_SEED:-42}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
GPU="$1"; CFG="$2"
echo "[run] GPU=$GPU CFG=$CFG CELLS=${EVAL_CELLS:-<config-default>} SUFFIX=${EVAL_OUT_SUFFIX:-<none>}"
CUDA_VISIBLE_DEVICES="$GPU" torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  --master_port=$((29400 + GPU)) \
  -m evaluation.run_eval --config "$CFG" --metrics external_ppl
echo "[run] generation+score done; computing entropy"
CUDA_VISIBLE_DEVICES="$GPU" python -m evaluation.compute_entropy_from_caches --config "$CFG" --include_real || true
echo "[run] DONE GPU=$GPU CFG=$CFG"
