#!/bin/bash
# Evaluate the released CoBit-M (462M) OWT checkpoint and reproduce the
# Table-2 operating points. Runs generation + GPT-2-Large GenPPL, then the
# post-hoc token-unigram entropy estimator over the cached generations.
#
# Usage (single GPU is the default — most robust):
#   bash scripts/owt/eval_cobit_m.sh                 # 1 GPU, all Table-2 cells
#   NPROC=2 bash scripts/owt/eval_cobit_m.sh         # multi-GPU DDP (see note)
#   EVAL_CELLS=all EVAL_NUM_SAMPLES=16 \             # quick smoke
#       bash scripts/owt/eval_cobit_m.sh
#
# NOTE on multi-GPU: DDP eval needs a working NCCL all-gather to collect the
# generated samples on rank 0. On workstation boxes where the GPUs are linked
# only over PCIe (no NVLink), that collective can busy-wait/hang even with
# NCCL_P2P_DISABLE=1. If a 2-GPU run sticks at 100% GPU with no progress after
# generation, fall back to NPROC=1 (or run two single-GPU processes over
# disjoint cells, e.g. EVAL_CELLS="256:0.21,512:0.26" on one GPU and
# EVAL_CELLS="384:0.24,256:0.13" on the other).
#
# Env knobs (all optional):
#   NPROC               GPUs / processes        (default 1)
#   EVAL_CELLS          table2 | low_ppl | all | "nfe:gamma,nfe:gamma"  (default table2)
#   EVAL_NUM_SAMPLES    samples per cell        (default 1024 — required for paper numbers)
#   EVAL_SEED           sampling seed           (default 42)
#   EVAL_CKPT_STEP      checkpoint step         (default 000750000)
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
PROJECT_DIR="$(pwd)"
mkdir -p logs

CONFIG_PATH="configs/owt/eval_cobit_m_750K.py"
NPROC="${NPROC:-1}"

# Use whatever conda env has torch; override CONDA_ENV if yours differs.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "${CONDA_ENV:-pytorch}" 2>/dev/null || true
fi

export EVAL_SEED="${EVAL_SEED:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# Multi-GPU on consumer/workstation boxes: P2P over PCIe can hang NCCL.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
MASTER_PORT=$((29000 + RANDOM % 1000))

echo "=== CoBit-M eval ==="
echo "config=$CONFIG_PATH nproc=$NPROC cells=${EVAL_CELLS:-table2} N=${EVAL_NUM_SAMPLES:-1024} seed=$EVAL_SEED ckpt=${EVAL_CKPT_STEP:-000750000}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  -m evaluation.run_eval --config "$CONFIG_PATH" --metrics external_ppl

echo "=== post-hoc token-unigram entropy from caches ==="
python -m evaluation.compute_entropy_from_caches --config "$CONFIG_PATH"

echo "=== done; results under runs/.../evaluation_cobit_m_table2_step${EVAL_CKPT_STEP:-000750000}/ ==="
