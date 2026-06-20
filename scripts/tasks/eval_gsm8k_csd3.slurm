#!/bin/bash
#SBATCH --job-name=gsm8k_eval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/tasks/%x_%j.log
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-N/A} on ${SLURMD_NODENAME:-$(hostname)}: $(date)"
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"
# env passthrough: CKPT, SAMPLER, GAMMA, STEPS, LIMIT, SIGMA_DATA
bash scripts/tasks/eval_gsm8k.sh
echo "eval exit=$?"
