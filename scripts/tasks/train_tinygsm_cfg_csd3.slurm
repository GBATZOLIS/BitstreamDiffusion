#!/bin/bash
#SBATCH --job-name=tinygsm_cfg_train
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/tasks/%x_%j.log
#
# CoBit TinyGSM training WITH classifier-free-guidance dropout (p_uncond=0.1),
# 250k steps, single 4xA100 node. Parallel run to the no-guidance experiment so
# we can test whether guidance improves GSM8K. Separate run dir
# (cobit_raw_binary_bits_cfg) so the two never collide. Tokenized cache is shared.
#
# Submitted as an explicit afterany-chained set of 12h links (CHAIN_REMAINING=1
# disables in-script self-submission). The cache must already exist (the base run
# built it); each link's safeguard re-checks it.
set -euo pipefail
echo "Job ID: ${SLURM_JOB_ID:-N/A}; Node: ${SLURMD_NODENAME:-$(hostname)}; Date: $(date)"

PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs/tasks

CONFIG_PATH="configs/tasks/tinygsm_bits_cfg.py"
CACHE_META_GLOB="datasets/tinygsm/"*.meta.json
FINAL_CKPT="runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/checkpoints/step=000250000.pt"
SELF="scripts/tasks/train_tinygsm_cfg_csd3.slurm"
CHAIN_REMAINING="${CHAIN_REMAINING:-1}"

# --- SAFEGUARD: prebuilt cache must exist ---
if ! ls $CACHE_META_GLOB >/dev/null 2>&1; then
    echo "ERROR: no TinyGSM cache (datasets/tinygsm/*.meta.json). Refusing to train."
    exit 1
fi

# --- STOP CONDITION: training already complete ---
if [[ -f "$FINAL_CKPT" ]]; then
    echo "Final checkpoint exists ($FINAL_CKPT) -> complete. Stopping."
    exit 0
fi

# --- optional in-script self-submit (disabled when CHAIN_REMAINING=1) ---
if [[ "$CHAIN_REMAINING" -gt 1 ]]; then
    NEXT=$((CHAIN_REMAINING - 1))
    sbatch --dependency=singleton --export=ALL,CHAIN_REMAINING=$NEXT "$SELF" \
        && echo "Submitted successor (CHAIN_REMAINING=$NEXT)."
fi

# --- Environment (CSD3 Ampere/A100 + conda env) ---
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh
conda activate cobit
echo "python: $(which python)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" =~ ^[0-9]+$ && "$NPROC" -ge 1 ]] || NPROC=4
echo "Launching CFG training (p_uncond=0.1) on $NPROC GPUs ..."
NNODES=1 NPROC="$NPROC" CONFIG="$CONFIG_PATH" \
    RDZV_ENDPOINT="localhost:$((29500 + (${SLURM_JOB_ID:-0} % 400)))" \
    bash scripts/tasks/train_tinygsm.sh
echo "CFG training job finished (exit $?)."
