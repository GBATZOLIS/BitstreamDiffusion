#!/bin/bash
#SBATCH --job-name=gsm8k_cfg_g250k
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tasks/%x_%A_%a.log
# 250K CFG checkpoint: gamma x guidance interplay at a single NFE (from $NFE).
# ONE config per array task = independent single-GPU jobs; results stream in.
# Submit twice (NFE=384, NFE=512) with a tight, NFE-matched --time on the sbatch
# command line so short jobs BACKFILL into idle slots fast.
# Grid per NFE: gamma {0.3,0.34,0.41} x w {0.0,2.0,3.0} = 9 tasks (w=1 dropped:
# w=0~=w=1 in the 150K signal). No overlap: unique tags + dedicated out_dir + skip.
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
NFE="${NFE:?set NFE (e.g. 384 or 512) in the environment}"
mkdir -p "$OUT_DIR"

# 9 configs for this NFE: "GAMMA W".
CONFIGS=(
  "0.3 0.0"  "0.3 2.0"  "0.3 3.0"
  "0.34 0.0" "0.34 2.0" "0.34 3.0"
  "0.41 0.0" "0.41 2.0" "0.41 3.0"
)

idx="${SLURM_ARRAY_TASK_ID:?must run as a slurm array}"
read -r GAMMA W <<< "${CONFIGS[$idx]}"
echo "Job ${SLURM_JOB_ID} task ${idx} on ${SLURMD_NODENAME:-$(hostname)}: NFE=$NFE gamma=$GAMMA w=$W  $(date)"

if compgen -G "${OUT_DIR}/gsm8k_results_stochastic_g${GAMMA}_w${W}_s${NFE}_*.json" > /dev/null; then
  echo "SKIP: result for NFE=$NFE gamma=$GAMMA w=$W already exists."
  exit 0
fi

python -m evaluation.tasks.gsm8k_eval \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --sampler stochastic --sampler_kind ddim --schedule entropic \
  --gamma "$GAMMA" --guidance_scale "$W" --steps "$NFE" --limit "$LIMIT" \
  --out_dir "$OUT_DIR"
echo "task ${idx} done $(date)"
