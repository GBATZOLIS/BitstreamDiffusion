#!/bin/bash
#SBATCH --job-name=smoke_lm1b_seed42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

module load gcc-native/12.3
source ~/miniconda3/bin/activate pytorch
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

PROJECT_DIR="$HOME/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

CONFIG_PATH="configs/lm1b/continuous/eval/rate_eval_seeds.py"

export EVAL_SEED=42
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29000 + ($SLURM_JOB_ID % 1000)))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_DIR/hf_cache"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "Project Dir: $(pwd)"
echo "Config: $CONFIG_PATH"
echo "EVAL_SEED=$EVAL_SEED"
echo "MASTER_PORT=$MASTER_PORT"
echo "HF_HOME=$HF_HOME"

echo "--- Single-Seed (seed=42) Single-GPU Generative PPL Smoke Test (LM1B) ---"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  -m evaluation.run_eval \
  --config "$CONFIG_PATH" \
  --metrics external_ppl

echo "====================================================="
echo "--- Posthoc Token-ID Entropy Estimation ---"
echo "====================================================="

python -m evaluation.compute_entropy_from_caches \
  --config "$CONFIG_PATH" \
  --include_real

echo "Job Finished Successfully."
