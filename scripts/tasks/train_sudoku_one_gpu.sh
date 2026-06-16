#!/usr/bin/env bash
# Single-GPU Sudoku training for one difficulty (no DDP / no NCCL).
# Auto-resumes from runs/.../checkpoints/last.pt. Designed to run inside tmux
# so it survives SSH / VSCode / Claude-Code session termination.
#
#   DIFF=easy|medium|hard   GPU=<cuda index>   bash scripts/tasks/train_sudoku_one_gpu.sh
set -eo pipefail   # NOTE: no -u; conda activate scripts reference unbound vars
cd "$(dirname "$0")/../.."

DIFF="${DIFF:-easy}"
GPU="${GPU:-0}"

# Pre-seed vars that conda's cuda activate hook appends to.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Activate the project conda env (edit CONDA_SH if your install differs).
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-pytorch}"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export USE_COMPILE="${USE_COMPILE:-0}"
export SUDOKU_DIFFICULTY="$DIFF"
export SUDOKU_GEN_WORKERS="${SUDOKU_GEN_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[run] difficulty=$DIFF gpu=$GPU env=$CONDA_ENV pid=$$ $(date)"
# Pre-build the dataset cache (single process) if missing, then train.
python scripts/tasks/prebuild_task.py --config configs/tasks/sudoku_bits.py
python train.py --config configs/tasks/sudoku_bits.py
echo "[run] FINISHED difficulty=$DIFF $(date)"
