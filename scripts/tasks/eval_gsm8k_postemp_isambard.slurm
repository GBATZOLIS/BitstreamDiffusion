#!/bin/bash
#SBATCH --job-name=gsm8k_pt
#SBATCH --partition=workq
#SBATCH --account=brics.u6ms
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/tasks/%x_%A_%a.log
#
# Posterior-temperature GSM8K sweep (continuous analogue of MDLM/Duo low-T /
# S-FLM top-k=1). ONE temperature per array task on the 425k SOTA checkpoint,
# NFE 1024, near-cap churn. T=1.0 reproduces the untempered headline (29.19%) and
# is the zero-regression sanity check. sigma_data auto-resolves from the run-dir
# sidecar (trained ~0.3998) -- do NOT pass it.
#
# Phase 1 (1D T sweep, learned target, const schedule) -- fast pre-screen then full:
#   sbatch --time=00:20:00 --array=0-4 \
#     --export=ALL,GAMMA=0.41,LIMIT=256,TEMPS="1.0 0.9 0.7 0.5 0.3" \
#     scripts/tasks/eval_gsm8k_postemp_isambard.slurm
#   sbatch --time=00:50:00 --array=0-4 \
#     --export=ALL,GAMMA=0.41,LIMIT=1319,TEMPS="1.0 0.9 0.7 0.5 0.3" \
#     scripts/tasks/eval_gsm8k_postemp_isambard.slurm
#
# Phase 2 (schedule + target ablation at the best T, e.g. 0.5):
#   sbatch --time=00:50:00 --array=0-1 \
#     --export=ALL,GAMMA=0.41,LIMIT=1319,TEMPS="0.5 0.5",PT_SCHEDULE=sigma_ramp,PT_SIGMA_LO=0.1,PT_SIGMA_HI=4.0 \
#     scripts/tasks/eval_gsm8k_postemp_isambard.slurm   # (compare to const in Phase 1)
#   ...and re-run a chosen T with PT_TARGET=full to validate the matched-filter reasoning.
#
# Phase 3 (2D T x gamma): submit Phase-1 array once per GAMMA in {0.34, 0.41, 0.4142}.
#
# DDIM entropic-churn headline path; s_churn = gamma*(NFE-1); HARD cap gamma<=sqrt(2)-1~0.4142.
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A} on ${SLURMD_NODENAME:-$(hostname)}: $(date)"
PROJECT_DIR="/home/u6ms/gb511.u6ms/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR"; mkdir -p logs/tasks

: "${TEMPS:?set TEMPS, space-separated, indexed by SLURM_ARRAY_TASK_ID}"
GAMMA="${GAMMA:-0.41}"
STEPS="${STEPS:-1024}"
LIMIT="${LIMIT:-1319}"
CKPT_STEP="${CKPT_STEP:-425000}"                       # 425k = current SOTA
PT_TARGET="${PT_TARGET:-learned}"                      # learned (recommended) | full
PT_SCHEDULE="${PT_SCHEDULE:-const}"                    # const | sigma_ramp
PT_SIGMA_LO="${PT_SIGMA_LO:-0.1}"
PT_SIGMA_HI="${PT_SIGMA_HI:-4.0}"

STEP9="$(printf '%09d' "$CKPT_STEP")"
KTAG="$(( CKPT_STEP / 1000 ))k"
CKPT="runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=${STEP9}.pt"
OUT_DIR="runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_${KTAG}_postemp"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

# pick this task's temperature
read -r -a TARR <<< "$TEMPS"
TID="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$TID" -lt "${#TARR[@]}" ]] || { echo "ERROR: task id $TID >= ${#TARR[@]} temps"; exit 1; }
T="${TARR[$TID]}"

# ---- aarch64 env (same proven stack as the resume/eval runs) ----
module load gcc-native/12.3
module load cuda/12.6
set +u; source "$HOME/envs/pytorch/bin/activate"; set -u

export HF_HOME="$PROJECT_DIR/hf_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR"
export TMPDIR="/tmp/${USER}/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/ti" TRITON_CACHE_DIR="$TMPDIR/tr"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "===== ckpt=$KTAG NFE=$STEPS gamma=$GAMMA T=$T target=$PT_TARGET sched=$PT_SCHEDULE (lo=$PT_SIGMA_LO hi=$PT_SIGMA_HI) limit=$LIMIT (task $TID) ===== $(date +%T)"
echo "python: $(which python)"
python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint "$CKPT" \
  --sampler stochastic --sampler_kind ddim --schedule entropic \
  --gamma "$GAMMA" --steps "$STEPS" --limit "$LIMIT" --ema 1 \
  --posterior_temp "$T" --posterior_temp_target "$PT_TARGET" \
  --posterior_temp_schedule "$PT_SCHEDULE" \
  --posterior_temp_sigma_lo "$PT_SIGMA_LO" --posterior_temp_sigma_hi "$PT_SIGMA_HI" \
  --out_dir "$OUT_DIR"
echo "eval done (T=$T gamma=$GAMMA) exit=$? $(date)"
