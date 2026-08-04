#!/usr/bin/env bash
# Launch the four M0 v4 base-fraction variants concurrently, one GPU each.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

sessions=(m0_v4_base000 m0_v4_base025 m0_v4_base050 m0_v4_base100)
configs=(
  configs/proteins/m0_v4_base000.py
  configs/proteins/m0_v4_base025.py
  configs/proteins/m0_v4_base050.py
  configs/proteins/m0_v4_base100.py
)
ports=(29740 29741 29742 29743)

if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
  echo "launch_m0_v4_sweep.sh: four visible GPUs are required" >&2
  exit 1
fi

for config in "${configs[@]}"; do
  [[ -f "$config" ]] || {
    echo "launch_m0_v4_sweep.sh: missing config: $config" >&2
    exit 1
  }
done

for session in "${sessions[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "launch_m0_v4_sweep.sh: tmux session already exists: $session" >&2
    exit 1
  fi
done

for gpu in 0 1 2 3; do
  session="${sessions[$gpu]}"
  config="${configs[$gpu]}"
  port="${ports[$gpu]}"
  run_dir="runs/proteins/$session"
  mkdir -p "$run_dir"
  command="exec env CUDA_VISIBLE_DEVICES=$gpu NPROC=1 MASTER_PORT=$port scripts/launch/train_4gpu.sh $config >> $run_dir/train.out 2>&1"
  tmux new-session -d -s "$session" -c "$REPO_ROOT" "$command"
  echo "started $session on GPU $gpu (port $port)"
done

echo
tmux list-sessions
