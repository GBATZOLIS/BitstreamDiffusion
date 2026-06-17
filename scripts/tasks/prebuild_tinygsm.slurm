#!/usr/bin/env bash
#SBATCH --job-name=tinygsm_prebuild
#SBATCH --account=brics.u6ms
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/tasks/prebuild_tinygsm.%j.out
#SBATCH --error=logs/tasks/prebuild_tinygsm.%j.out
#
# CPU-only TinyGSM prebuild: tokenize the (already-downloaded) TinyGSM corpus
# into the uint16 cache under datasets/tinygsm. No GPU and NO INTERNET needed —
# the raw dataset + SmolLM-135M tokenizer are already in $HF_HOME, so we run
# fully offline. Submit from the repo root:
#   sbatch scripts/tasks/prebuild_tinygsm.slurm
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

source /home/u6ms/gb511.u6ms/envs/pytorch/bin/activate
export HF_HOME=/projects/u6ms/gb511/projects/BitstreamDiffusion/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Single-threaded tokenize loop; cap BLAS/OMP threads (avoids RLIMIT_NPROC blowups).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[slurm] node=$(hostname) start=$(date -Is)"
python scripts/tasks/prebuild_task.py --config configs/tasks/tinygsm_bits.py
echo "[slurm] done=$(date -Is)"
ls -la datasets/tinygsm/
