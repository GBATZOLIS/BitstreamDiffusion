#!/bin/bash
#SBATCH --job-name=gsm8k_cfg_early
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/tasks/%x_%j.log
# EARLY directional signal: does classifier-free guidance trend positive on the
# CFG run? Evaluates the 150k CFG checkpoint at a FIXED operating point
# (NFE=256, gamma=0.31 -- the unguided 256 op-point) while sweeping the guidance
# weight w. w=0.0 is the control (= cfg-150k unguided at this op-point); w>0
# shows the marginal effect of guidance. NOT step/NFE-matched to the 23.4%
# (250k base, NFE512) headline -- directional only.
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-N/A} on ${SLURMD_NODENAME:-$(hostname)}: $(date)"
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"

CKPT="${CKPT:-runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/checkpoints/step=000150000.pt}"
STEPS="${STEPS:-256}"
GAMMA="${GAMMA:-0.31}"
WS="${WS:-0.0 1.0 2.0 3.0}"
LIMIT="${LIMIT:-1319}"

for w in $WS; do
  echo "================ NFE=$STEPS  gamma=$GAMMA  guidance_w=$w ================ $(date +%T)"
  python -m evaluation.tasks.gsm8k_eval \
    --config configs/tasks/tinygsm_bits_cfg.py \
    --checkpoint "$CKPT" \
    --sampler stochastic --sampler_kind ddim --schedule entropic \
    --gamma "$GAMMA" --guidance_scale "$w" --steps "$STEPS" --limit "$LIMIT"
done
echo "cfg early-signal sweep done $(date)"
