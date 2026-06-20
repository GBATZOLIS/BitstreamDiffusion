#!/bin/bash
#SBATCH --job-name=gsm8k_cfg_w
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tasks/%x_%A_%a.log
# 250K CFG checkpoint: fixed guidance W (from env) at a single NFE (from env),
# one gamma per array task (3 gammas). Same dedicated out_dir + unique tags +
# skip-if-exists => no overlap with the running {0,2,3} grid.
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
NFE="${NFE:?set NFE}"; W="${W:?set W}"
mkdir -p "$OUT_DIR"

GAMMAS=( "0.3" "0.34" "0.41" )
idx="${SLURM_ARRAY_TASK_ID:?must run as a slurm array}"
GAMMA="${GAMMAS[$idx]}"
echo "Job ${SLURM_JOB_ID} task ${idx} on ${SLURMD_NODENAME:-$(hostname)}: NFE=$NFE gamma=$GAMMA w=$W  $(date)"

if compgen -G "${OUT_DIR}/gsm8k_results_stochastic_g${GAMMA}_w${W}_s${NFE}_*.json" > /dev/null; then
  echo "SKIP: result for NFE=$NFE gamma=$GAMMA w=$W already exists."; exit 0
fi

python -m evaluation.tasks.gsm8k_eval \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --sampler stochastic --sampler_kind ddim --schedule entropic \
  --gamma "$GAMMA" --guidance_scale "$W" --steps "$NFE" --limit "$LIMIT" \
  --out_dir "$OUT_DIR"
echo "task ${idx} done $(date)"
