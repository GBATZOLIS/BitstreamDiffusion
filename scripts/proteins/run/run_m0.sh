#!/usr/bin/env bash
#
# run_m0.sh - launch the optimum M0 run (configs/proteins/m0_v1.py) on the 4
# local GPUs. Thin wrapper over scripts/launch/train_4gpu.sh that pins the M0
# config, tees to a log file, and can detach so the multi-day run survives logout.
#
# Usage:
#   scripts/proteins/run/run_m0.sh              # foreground (console + log file)
#   scripts/proteins/run/run_m0.sh --detach     # background, survives the shell
#   CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/proteins/run/run_m0.sh   # pick GPUs
#
# W&B auth is handled by train_4gpu.sh (it sources .trentinium/keys/wandb.env or
# uses the ~/.netrc cache). Relaunching auto-resumes from the last checkpoint and
# continues the same W&B run (run_id is pinned to "m0_v1" in the config).
#
# Monitor:  tail -f logs/proteins/m0_v1.log
# Stop:     pkill -f 'train.py --config configs/proteins/m0_v1.py'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/proteins/m0_v1.py"
LOG="logs/proteins/m0_v1.log"
mkdir -p "$(dirname "$LOG")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC="${NPROC:-4}"

LAUNCH=(scripts/launch/train_4gpu.sh "$CONFIG")

if [[ "${1:-}" == "--detach" || "${1:-}" == "-d" ]]; then
  nohup "${LAUNCH[@]}" >>"$LOG" 2>&1 &
  PID=$!
  echo "M0 (m0_v1) launched detached on GPUs $CUDA_VISIBLE_DEVICES; PID $PID"
  echo "  log:     tail -f $LOG"
  echo "  wandb:   https://wandb.ai/trentini/cobit-proteins/runs/m0_v1"
  echo "  stop:    pkill -f 'train.py --config configs/proteins/m0_v1.py'"
else
  echo "M0 (m0_v1) on GPUs $CUDA_VISIBLE_DEVICES; logging to $LOG (Ctrl-C to stop)"
  "${LAUNCH[@]}" 2>&1 | tee "$LOG"
fi
