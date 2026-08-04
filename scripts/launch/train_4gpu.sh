#!/usr/bin/env bash
#
# train_4gpu.sh - single-node, 4-GPU training launcher.
#
# Usage:
#   scripts/launch/train_4gpu.sh [CONFIG]
#
# CONFIG is any config accepted by train.py and defaults to
# configs/proteins/multimodal_lfq18_35m.py. This launcher and its multi-node
# sibling scripts/launch/train_slurm.sbatch consume the SAME configs and differ
# only in how the distributed workers are spawned: here a single local torchrun,
# there srun plus torchrun with a Slurm-derived rendezvous.
#
# Optional environment overrides:
#   NPROC                 workers spawned on this node          (default 4)
#   CUDA_VISIBLE_DEVICES  local GPUs bound to the workers       (default 0,1,2,3)
#   NNODES                node count for the rendezvous         (default 1)
#   NODE_RANK             this node's rank in the rendezvous    (default 0)
#   MASTER_ADDR           rendezvous host                       (default 127.0.0.1)
#   MASTER_PORT           rendezvous port                       (default derived)
#   PYTHON                interpreter that spawns torchrun      (default repo .venv)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load private credentials (e.g. WANDB_API_KEY) if present, so detached and Slurm
# launches authenticate without a prior interactive `wandb login`. The file lives
# under the git-ignored .trentinium/ and is optional.
if [[ -z "${WANDB_API_KEY:-}" && -f "$REPO_ROOT/.trentinium/keys/wandb.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.trentinium/keys/wandb.env"
fi

CONFIG="${1:-configs/proteins/multimodal_lfq18_35m.py}"
if [[ ! -f "$CONFIG" ]]; then
  echo "train_4gpu.sh: config not found: $CONFIG" >&2
  exit 1
fi

# Bind ranks to local GPUs. torchrun assigns one LOCAL_RANK per worker and
# train.py calls torch.cuda.set_device(LOCAL_RANK), so worker i drives the i-th
# device listed in CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"

# Rendezvous parameters. The defaults describe one local node; an outer launcher
# may override any of them through the environment (this is the "reads world
# size / rank / rendezvous from the launcher env" contract).
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (RANDOM % 1000)))}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "train_4gpu.sh: config=$CONFIG nproc=$NPROC gpus=$CUDA_VISIBLE_DEVICES rdzv=$MASTER_ADDR:$MASTER_PORT"

# torchrun (python -m torch.distributed.run) is the launcher; it exports RANK,
# LOCAL_RANK and WORLD_SIZE for each worker, which train.py reads to initialize
# the process group and then runs python train.py --config "$CONFIG".
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NPROC" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py --config "$CONFIG"
