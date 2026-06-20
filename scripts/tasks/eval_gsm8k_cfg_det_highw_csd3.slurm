#!/bin/bash
#SBATCH --job-name=gsm8k_cfg_detw
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tasks/%x_%A_%a.log
# 250K CFG checkpoint: DETERMINISTIC (gamma=0, no churn) + HIGH guidance sweep.
# The gamma=0 row was still climbing at w=4 (9.3 -> 14.0); this pins the
# pure-guidance ceiling with w in {6,8,12}. NFE from $NFE. One w per array task.
set -euo pipefail
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"

CKPT="runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/checkpoints/step=000250000.pt"
CONFIG="configs/tasks/tinygsm_bits_cfg.py"
OUT_DIR="runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/gsm8k_eval_grid250k"
LIMIT="${LIMIT:-1319}"
NFE="${NFE:?set NFE}"
mkdir -p "$OUT_DIR"

WS=( "6.0" "8.0" "12.0" )
idx="${SLURM_ARRAY_TASK_ID:?must run as a slurm array}"
W="${WS[$idx]}"
echo "Job ${SLURM_JOB_ID} task ${idx} on ${SLURMD_NODENAME:-$(hostname)}: NFE=$NFE gamma=0.0 w=$W (deterministic)  $(date)"

if compgen -G "${OUT_DIR}/gsm8k_results_deterministic_g0.0_w${W}_s${NFE}_*.json" > /dev/null; then
  echo "SKIP: deterministic NFE=$NFE w=$W already exists."; exit 0
fi

python -m evaluation.tasks.gsm8k_eval \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --sampler deterministic --sampler_kind ddim --schedule entropic \
  --gamma 0.0 --guidance_scale "$W" --steps "$NFE" --limit "$LIMIT" \
  --out_dir "$OUT_DIR"
echo "task ${idx} done $(date)"
