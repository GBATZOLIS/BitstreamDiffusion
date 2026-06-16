#!/usr/bin/env bash
# Train CoBit on the Sudoku task for one difficulty.
#   SUDOKU_DIFFICULTY in {easy,medium,hard} (default easy)
#   NPROC = GPUs per node (default 2). For Isambard multi-node, set
#   NNODES/NODE_RANK/RDZV_ENDPOINT and launch one process group per node.
set -euo pipefail
cd "$(dirname "$0")/../.."

export SUDOKU_DIFFICULTY="${SUDOKU_DIFFICULTY:-easy}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NPROC="${NPROC:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29501}"

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC}" --rdzv_backend=c10d --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train.py --config configs/tasks/sudoku_bits.py
