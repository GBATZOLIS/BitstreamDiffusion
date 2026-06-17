#!/usr/bin/env bash
#SBATCH --job-name=tinygsm_prebuild
#SBATCH --account=GIROLAMI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/tasks/prebuild_tinygsm.%j.out
#SBATCH --error=logs/tasks/prebuild_tinygsm.%j.out
#
# CSD3 (Wilkes3) CPU-only TinyGSM prebuild: tokenize the (already-downloaded)
# TinyGSM corpus into the uint16 cache under datasets/tinygsm. No GPU and NO
# INTERNET needed — the raw dataset + SmolLM-135M tokenizer are already in
# $HF_HOME (pre-staged on a login node), so we run fully offline.
# Submit from the repo root:
#   sbatch scripts/tasks/prebuild_tinygsm_csd3.slurm
set -euo pipefail

PROJECT_DIR="/rds/project/rds-LlrDsbHU5UM/gb511/projects/BitstreamDiffusion"
cd "${SLURM_SUBMIT_DIR:-$PROJECT_DIR}"
mkdir -p logs/tasks

# --- Environment: CSD3 conda env (replaces Isambard ~/envs/pytorch venv) -------
module purge 2>/dev/null || true
module load rhel8/default-icl 2>/dev/null || true
source /home/gb511/miniconda3/etc/profile.d/conda.sh
conda activate cobit
echo "[slurm] python: $(which python)"

export HF_HOME="$PROJECT_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Compute node (no login RLIMIT_NPROC concern): let the fast tokenizer use all
# allocated cores for batch encoding. Cap BLAS/OMP (numpy work here is trivial).
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[slurm] node=$(hostname) start=$(date -Is)"
python scripts/tasks/prebuild_task.py --config configs/tasks/tinygsm_bits.py
echo "[slurm] done=$(date -Is)"
ls -la datasets/tinygsm/
