#!/bin/bash
#SBATCH --job-name=gsm8k_base_1024
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:10:00
#SBATCH --output=logs/tasks/%x_%A_%a.log
# BASE (unguided) 250K checkpoint -- the config that gave the 23.43% headline.
# Push the budget to NFE=1024 with a churn sweep around the best 512 (g0.34) and
# 384 (g0.41) operating points. s_churn = gamma*(NFE-1), so total injected noise
# grows with NFE; gamma=0.20 keeps total churn near the 512 op-point, gamma=0.41
# matches the best 384 gamma. One gamma per array task.
set -euo pipefail
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"

CKPT="runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000250000.pt"
CONFIG="configs/tasks/tinygsm_bits.py"
OUT_DIR="runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval"
LIMIT="${LIMIT:-1319}"
NFE=1024
mkdir -p "$OUT_DIR"

GAMMAS=( "0.2" "0.27" "0.34" "0.41" )
idx="${SLURM_ARRAY_TASK_ID:?must run as a slurm array}"
GAMMA="${GAMMAS[$idx]}"
echo "Job ${SLURM_JOB_ID} task ${idx} on ${SLURMD_NODENAME:-$(hostname)}: BASE NFE=$NFE gamma=$GAMMA  $(date)"

if compgen -G "${OUT_DIR}/gsm8k_results_stochastic_g${GAMMA}_*s${NFE}_*.json" > /dev/null; then
  echo "SKIP: base NFE=$NFE gamma=$GAMMA already exists."; exit 0
fi

python -m evaluation.tasks.gsm8k_eval \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --sampler stochastic --sampler_kind ddim --schedule entropic \
  --gamma "$GAMMA" --steps "$NFE" --limit "$LIMIT" \
  --out_dir "$OUT_DIR"
echo "task ${idx} done $(date)"
