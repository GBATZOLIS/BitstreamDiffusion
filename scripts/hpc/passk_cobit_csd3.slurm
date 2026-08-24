#!/bin/bash
#SBATCH --job-name=passk_cobit
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/tasks/%x_%a_%j.log
#
# CoBit pass@32 on shard B (1063 problems), split into 4 interleaved chunks so
# each array cell fits inside the 12h limit. beta=1 + no resampling => K
# INDEPENDENT churn samples per problem (no tempering, no selection).
#
# Merge afterwards (chunks are disjoint):
#   python /home/gb511/s-flm/merge_passk.py merged.json \
#       runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_shardB_chunk*/gsm8k_results_*.json
set -euo pipefail
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate cobit
cd "${COBIT_ROOT:-$HOME/rds_work/projects/BitstreamDiffusion}"
mkdir -p logs/tasks

C=$SLURM_ARRAY_TASK_ID
SHARD=${SHARD_DIR:-$HOME/rds_work/projects/s-flm}/data_gsm8k_shardB_chunk${C}.json

python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000425000.pt \
  --gsm8k_test_path "$SHARD" \
  --sampler_kind fkc_em --proposal edm_churn --churn_gamma 0.41 \
  --beta 1.0 --num_particles ${K:-32} --resampling_policy never --final_resample 0 \
  --steps 1024 --limit 400 --batch_size 2 --ema 1 --seed 42 \
  --sigma_data 0.399844765663147 \
  --out_dir runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_shardB_chunk${C}
echo "DONE chunk $C"
