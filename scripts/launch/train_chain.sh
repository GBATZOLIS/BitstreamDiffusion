#!/usr/bin/env bash
#
# train_chain.sh - submit a chain of dependent Slurm jobs for a multi-day run.
#
# A single training run that exceeds one job's wall-time is split into N jobs,
# each depending on the previous with `--dependency=afterany` so the next starts
# whether the previous finished, hit the wall-time, or was requeued/preempted.
# Every job runs the SAME config; train.py auto-resumes from
# runs/<experiment>/checkpoints/last.pt, so the chain makes continuous progress.
#
# Usage:
#   scripts/launch/train_chain.sh <NUM_JOBS> [CONFIG] [sbatch args...]
#
# Examples:
#   scripts/launch/train_chain.sh 6 configs/proteins/multimodal_lfq18_400m.py
#   scripts/launch/train_chain.sh 10 configs/proteins/multimodal_lfq18_650m.py \
#       --nodes=4 --gres=gpu:8
#
# The first argument is the number of links in the chain; the second is the
# config (defaults to the 400M foundation config); any further arguments are
# passed through to every sbatch call (e.g. --nodes / --gres to size the
# allocation). Requires that train_slurm.sbatch already resumes from last.pt
# (it does: train.py detects the checkpoint and saved config on each launch).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_JOBS="${1:-4}"
CONFIG="${2:-configs/proteins/multimodal_lfq18_400m.py}"
shift $(( $# >= 2 ? 2 : $# )) || true
SBATCH_EXTRA=("$@")

if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || (( NUM_JOBS < 1 )); then
  echo "train_chain.sh: NUM_JOBS must be a positive integer, got '$NUM_JOBS'" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "train_chain.sh: config not found: $CONFIG" >&2
  exit 1
fi

SBATCH="scripts/launch/train_slurm.sbatch"
PREV=""
echo "train_chain.sh: chaining $NUM_JOBS jobs on $CONFIG ${SBATCH_EXTRA[*]:-}"
for (( i = 1; i <= NUM_JOBS; i++ )); do
  if [[ -z "$PREV" ]]; then
    JID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" "$SBATCH" "$CONFIG")
  else
    JID=$(sbatch --parsable --dependency=afterany:"$PREV" "${SBATCH_EXTRA[@]}" "$SBATCH" "$CONFIG")
  fi
  echo "  link $i/$NUM_JOBS: job $JID${PREV:+ (after $PREV)}"
  PREV="$JID"
done
echo "train_chain.sh: submitted. Cancel the whole chain with: scancel $PREV (and its predecessors), or scancel --name=cobit_multimodal"
