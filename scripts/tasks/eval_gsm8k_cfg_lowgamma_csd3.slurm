#!/bin/bash
#SBATCH --job-name=gsm8k_cfg_lowg
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tasks/%x_%A_%a.log
# 250K CFG checkpoint: LOW-gamma + guidance interplay at a single NFE (from $NFE).
# Completes the gamma x guidance surface downward: gamma {0.0,0.1,0.2} x w {0,2,3,4}.
# gamma=0.0 => deterministic sampler (pure guidance, no churn) -- the key anchor.
# One config per array task (12 tasks). Dedicated out_dir + unique tags + skip.
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

# 12 configs for this NFE: "GAMMA W".
CONFIGS=(
  "0.0 0.0"  "0.0 2.0"  "0.0 3.0"  "0.0 4.0"
  "0.1 0.0"  "0.1 2.0"  "0.1 3.0"  "0.1 4.0"
  "0.2 0.0"  "0.2 2.0"  "0.2 3.0"  "0.2 4.0"
)
idx="${SLURM_ARRAY_TASK_ID:?must run as a slurm array}"
read -r GAMMA W <<< "${CONFIGS[$idx]}"

# gamma=0 -> deterministic (no churn); gamma>0 -> stochastic EDM churn.
if [ "$GAMMA" = "0.0" ]; then SAMPLER=deterministic; else SAMPLER=stochastic; fi
echo "Job ${SLURM_JOB_ID} task ${idx} on ${SLURMD_NODENAME:-$(hostname)}: NFE=$NFE gamma=$GAMMA w=$W sampler=$SAMPLER  $(date)"

if compgen -G "${OUT_DIR}/gsm8k_results_${SAMPLER}_g${GAMMA}_w${W}_s${NFE}_*.json" > /dev/null; then
  echo "SKIP: result for $SAMPLER NFE=$NFE gamma=$GAMMA w=$W already exists."; exit 0
fi

python -m evaluation.tasks.gsm8k_eval \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --sampler "$SAMPLER" --sampler_kind ddim --schedule entropic \
  --gamma "$GAMMA" --guidance_scale "$W" --steps "$NFE" --limit "$LIMIT" \
  --out_dir "$OUT_DIR"
echo "task ${idx} done $(date)"
