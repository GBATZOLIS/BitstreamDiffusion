#!/bin/bash
#SBATCH --job-name=tinygsm_cfg_train
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/tasks/%x_%j.log
#
# RESUME the CFG (p_uncond=0.1) TinyGSM run past 250k. Seamless true-resume from
# last.pt; constant LR 3e-4, warmup done. Target step set by TINYGSM_TOTAL_STEPS.
# Separate run dir (cobit_raw_binary_bits_cfg) so it never collides with base.
# Submit head with e.g.:
#   TINYGSM_TOTAL_STEPS=500000 CHAIN_REMAINING=6 sbatch scripts/tasks/resume_tinygsm_cfg_csd3.slurm
set -euo pipefail
echo "Job ID: ${SLURM_JOB_ID:-N/A}; Node: ${SLURMD_NODENAME:-$(hostname)}; Date: $(date)"
PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs/tasks

CONFIG_PATH="configs/tasks/tinygsm_bits_cfg.py"
CACHE_META_GLOB="datasets/tinygsm/"*.meta.json
TARGET_STEPS="${TINYGSM_TOTAL_STEPS:?set TINYGSM_TOTAL_STEPS (e.g. 500000)}"
export TINYGSM_TOTAL_STEPS="$TARGET_STEPS"
CKPT_DIR="runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/checkpoints"
FINAL_CKPT="$CKPT_DIR/step=$(printf '%09d' "$TARGET_STEPS").pt"
SELF="scripts/tasks/resume_tinygsm_cfg_csd3.slurm"
CHAIN_REMAINING="${CHAIN_REMAINING:-6}"

if ! ls $CACHE_META_GLOB >/dev/null 2>&1; then
    echo "ERROR: no TinyGSM cache. Refusing to train; chain stops."; exit 1
fi
if [[ -f "$FINAL_CKPT" ]]; then
    echo "Target checkpoint exists ($FINAL_CKPT) -> resume complete. Stopping chain."; exit 0
fi
# No-op guard: if the latest checkpoint is already at/past TARGET_STEPS this
# resume would train zero steps. Refuse and do NOT submit a successor, otherwise
# the singleton chain spins, respawning identical no-op jobs until CHAIN_REMAINING
# is exhausted (the original failure). CUR_STEP = highest step=*.pt (lower bound
# on the true resume point in last.pt, so this is conservative).
CUR_STEP=0
if compgen -G "$CKPT_DIR/step=*.pt" >/dev/null 2>&1; then
    CUR_STEP=$(ls -1 "$CKPT_DIR"/step=*.pt | sed -E 's#.*step=0*([0-9]+)\.pt$#\1#' | sort -n | tail -1)
fi
if [[ "$CUR_STEP" -ge "$TARGET_STEPS" ]]; then
    echo "ERROR: latest checkpoint step ($CUR_STEP) >= TARGET_STEPS ($TARGET_STEPS)."
    echo "       This resume would train zero steps. NOT submitting a successor; chain stops."
    echo "       Raise TINYGSM_TOTAL_STEPS above $CUR_STEP to actually train further."
    exit 1
fi
echo "Resume guard OK: latest ckpt step=$CUR_STEP < target=$TARGET_STEPS -> work remains."
if [[ "$CHAIN_REMAINING" -gt 1 ]]; then
    NEXT=$((CHAIN_REMAINING - 1))
    sbatch --dependency=singleton \
        --export=ALL,CHAIN_REMAINING=$NEXT,TINYGSM_TOTAL_STEPS=$TARGET_STEPS \
        "$SELF" && echo "Submitted successor (CHAIN_REMAINING=$NEXT, target=$TARGET_STEPS)."
else
    echo "CHAIN_REMAINING=$CHAIN_REMAINING -> no successor (ceiling reached)."
fi

. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
echo "python: $(which python) | resuming CFG to TINYGSM_TOTAL_STEPS=$TARGET_STEPS"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" =~ ^[0-9]+$ && "$NPROC" -ge 1 ]] || NPROC=4
echo "Launching CFG resume training on $NPROC GPUs ..."
NNODES=1 NPROC="$NPROC" CONFIG="$CONFIG_PATH" \
    RDZV_ENDPOINT="localhost:$((29500 + (${SLURM_JOB_ID:-0} % 400)))" \
    bash scripts/tasks/train_tinygsm.sh
echo "CFG resume training job finished (exit $?)."
