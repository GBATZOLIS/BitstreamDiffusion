#!/bin/bash
#SBATCH --job-name=gsm8k_250k_best
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/tasks/%x_%j.log
# One job, several (NFE:gamma) configs run sequentially on a single GPU.
# Required env: CKPT, CONFIGS="STEPS:GAMMA STEPS:GAMMA ...". gamma=0 -> deterministic.
# sigma_data auto-resolves from the checkpoint run-dir sidecar (trained 0.3998).
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-N/A} on ${SLURMD_NODENAME:-$(hostname)}: $(date)"
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"
: "${CKPT:?set CKPT}"; : "${CONFIGS:?set CONFIGS=\"STEPS:GAMMA ...\"}"
LIMIT="${LIMIT:-1319}"
for pair in $CONFIGS; do
  STEPS="${pair%%:*}"; g="${pair##*:}"
  if [ "$g" = "0.0" ] || [ "$g" = "0" ]; then S=deterministic; else S=stochastic; fi
  echo "================ NFE=$STEPS  sampler=$S  gamma=$g ================ $(date +%T)"
  python -m evaluation.tasks.gsm8k_eval \
    --config configs/tasks/tinygsm_bits.py \
    --checkpoint "$CKPT" \
    --sampler "$S" --sampler_kind ddim --schedule entropic \
    --gamma "$g" --steps "$STEPS" --limit "$LIMIT"
done
echo "best-config eval done $(date)"
