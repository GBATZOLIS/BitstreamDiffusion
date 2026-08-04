#!/usr/bin/env bash
#SBATCH --job-name=cobit_multimodal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
#
# train_slurm.sbatch - multi-node training launcher.
#
# Usage:
#   sbatch scripts/launch/train_slurm.sbatch [CONFIG]
#   sbatch --nodes=4 --gres=gpu:8 scripts/launch/train_slurm.sbatch CONFIG
#
# CONFIG is any config accepted by train.py and defaults to
# configs/proteins/multimodal_lfq18_35m.py. This script consumes the SAME
# configs as scripts/launch/train_4gpu.sh and differs only in the launcher: srun
# starts one torchrun per node and the rendezvous is derived from SLURM_*
# environment variables (MASTER_ADDR from scontrol, node rank from SLURM_NODEID).
# Node and GPU counts are parameterized: override the SBATCH --nodes / --gres
# directives on the sbatch command line to scale the allocation.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

CONFIG="${1:-configs/proteins/multimodal_lfq18_35m.py}"
if [[ ! -f "$CONFIG" ]]; then
  echo "train_slurm.sbatch: config not found: $CONFIG" >&2
  exit 1
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# World size is derived from the Slurm allocation: one worker per GPU per node.
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"

# Rendezvous host: the first node of the allocation, resolved via scontrol.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((29500 + (SLURM_JOB_ID % 1000)))}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "train_slurm.sbatch: config=$CONFIG nodes=$NNODES gpus_per_node=$GPUS_PER_NODE rdzv=$MASTER_ADDR:$MASTER_PORT"

# Export what the per-task shell needs; SLURM_NODEID and SLURM_JOB_ID are set by
# srun for each task and must be expanded inside the task, not at submit time,
# so the torchrun command lives in a single-quoted bash -c body.
export PYTHON CONFIG NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT

# One srun task per node; each task launches torchrun for that node. torchrun
# exports RANK, LOCAL_RANK and WORLD_SIZE for its workers, which train.py reads
# to initialize the process group and then runs python train.py --config CONFIG.
# With one task per node, SLURM_NODEID is the node rank.
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  "$PYTHON" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$SLURM_NODEID" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py --config "$CONFIG"
'
