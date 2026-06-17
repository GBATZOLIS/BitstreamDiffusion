#!/bin/bash
#SBATCH --job-name=tinygsm_train
#SBATCH --partition=workq
#SBATCH --account=brics.u6ms
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/tasks/%x_%j.log
#
# CoBit TinyGSM training, 250k steps, single 4xGH200 node (batch 512 -> 128/GPU).
#
# AUTONOMOUS CHAIN (disconnect-safe): each job, before training, submits its own
# singleton successor (so a 12h walltime-kill is automatically resumed from
# last.pt), then trains. The chain STOPS when the final checkpoint
# (step=000250000.pt) exists, and is hard-bounded by CHAIN_REMAINING to prevent
# a crash-loop from resubmitting forever.
#
# SAFEGUARD: every job refuses to run (exit 1, no successor) unless the prebuilt
# TinyGSM cache (datasets/tinygsm/*.meta.json) is present. Launch the head of the
# chain gated on the prebuild job so the WHOLE chain dies if prep failed:
#
#   sbatch --dependency=afterok:<PREBUILD_JOBID> scripts/tasks/train_tinygsm.slurm
#
# (successors are submitted internally with --dependency=singleton only.)
set -euo pipefail
echo "Job ID: ${SLURM_JOB_ID:-N/A}; Node: ${SLURMD_NODENAME:-$(hostname)}; Date: $(date)"

PROJECT_DIR="/projects/u6ms/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs/tasks

CONFIG_PATH="configs/tasks/tinygsm_bits.py"
CACHE_META_GLOB="datasets/tinygsm/"*.meta.json
FINAL_CKPT="runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000250000.pt"
CHAIN_REMAINING="${CHAIN_REMAINING:-10}"   # max jobs in the chain (10 * ~11.5h >> ~250k steps)

# --- SAFEGUARD: prebuild cache must exist (prep succeeded) ---------------------
if ! ls $CACHE_META_GLOB >/dev/null 2>&1; then
    echo "ERROR: no TinyGSM cache (datasets/tinygsm/*.meta.json). Prebuild not done/failed."
    echo "Refusing to train and NOT submitting a successor -> chain stops here."
    exit 1
fi

# --- STOP CONDITION: training already complete --------------------------------
if [[ -f "$FINAL_CKPT" ]]; then
    echo "Final checkpoint already exists ($FINAL_CKPT) -> training complete. Stopping chain."
    exit 0
fi

# --- Submit the singleton successor BEFORE training (covers walltime kills) ----
if [[ "$CHAIN_REMAINING" -gt 1 ]]; then
    NEXT=$((CHAIN_REMAINING - 1))
    sbatch --dependency=singleton \
        --export=ALL,CHAIN_REMAINING=$NEXT \
        scripts/tasks/train_tinygsm.slurm \
        && echo "Submitted successor (CHAIN_REMAINING=$NEXT)."
else
    echo "CHAIN_REMAINING=$CHAIN_REMAINING -> not submitting a successor (chain ceiling reached)."
fi

# --- Environment (mirrors the working multimodal GH200 launcher) ---------------
module load gcc-native/12.3 2>/dev/null || true
set +u; source ~/envs/pytorch/bin/activate; set -u
echo "python: $(which python)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TOKENIZERS_PARALLELISM=false
# Fully offline: the SmolLM tokenizer + (no-op) dataset load resolve from cache.
export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# train_tinygsm.sh re-runs prebuild on rank0 (no-op load since the cache exists),
# then launches torchrun for the given NPROC/NNODES. Single node here.
NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" =~ ^[0-9]+$ && "$NPROC" -ge 1 ]] || NPROC=4
echo "Launching training on $NPROC GPUs ..."
NNODES=1 NPROC="$NPROC" RDZV_ENDPOINT="localhost:$((29500 + (${SLURM_JOB_ID:-0} % 400)))" \
    bash scripts/tasks/train_tinygsm.sh
echo "Training job finished (exit $?)."
