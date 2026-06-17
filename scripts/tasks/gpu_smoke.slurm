#!/bin/bash
#SBATCH --job-name=gpu_smoke
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/tasks/%x_%j.log
set -euo pipefail
echo "Job ID: ${SLURM_JOB_ID:-N/A}; Node: ${SLURMD_NODENAME:-$(hostname)}; Date: $(date)"

PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"
mkdir -p logs/tasks

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh
conda activate cobit
echo "python: $(which python)"

export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

nvidia-smi -L
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:$((29500 + (${SLURM_JOB_ID:-0} % 400)))" \
    scripts/tasks/gpu_smoke.py
echo "smoke exit=$?"
