#!/usr/bin/env bash
# Train CoBit on TinyGSM (250k steps, global batch 512).
#   On Isambard: 2 nodes x 4 GH200 (NPROC=4, NNODES=2) gives 8 GPUs -> 64/GPU.
#   Set TINYGSM_MAX_TRAIN for a capped local smoke run.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NPROC="${NPROC:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29502}"

# Pre-build the tokenized cache once (single process) so no rank tokenizes under DDP.
if [ "${NODE_RANK}" = "0" ]; then
  python scripts/tasks/prebuild_task.py --config configs/tasks/tinygsm_bits.py
fi

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" --rdzv_backend=c10d --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train.py --config configs/tasks/tinygsm_bits.py
