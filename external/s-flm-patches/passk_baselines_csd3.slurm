#!/bin/bash
#SBATCH --job-name=passk_base
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/passk_%x_%a_%j.log
#
# pass@32 for the GSM8K baselines on shard B (1063 problems), 1024 steps.
# Their eval shards by rank internally, so 4 GPUs ~= 4x speedup per cell.
# Array cell -> (model, temperature):
#     0 duo  T=1.0     1 duo  T=0.1
#     2 mdlm T=1.0     3 mdlm T=0.1
#     4 sfm  (top-1 velocity, their best GSM8K variant)
set -euo pipefail
. /etc/profile.d/modules.sh; module purge; module load rhel8/default-amp
source /home/gb511/miniconda3/etc/profile.d/conda.sh; conda activate sfm2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "${SFM_ROOT:-$HOME/rds_work/projects/s-flm}"
mkdir -p logs

K=${K:-32}; STEPS=${STEPS:-1024}; BS=${BS:-128}; FP64=${FP64:-false}
DATA=$PWD/data_gsm8k_shardB_1063.json
CACHE=$PWD/data_cache_shardB

case $SLURM_ARRAY_TASK_ID in
  0) NAME=duo_T1.0;  ALGO=duo-base; CKPT=duo.ckpt;  T=1.0 ;;
  1) NAME=duo_T0.1;  ALGO=duo-base; CKPT=duo.ckpt;  T=0.1 ;;
  2) NAME=mdlm_T1.0; ALGO=mdlm;     CKPT=mdlm.ckpt; T=1.0 ;;
  3) NAME=mdlm_T0.1; ALGO=mdlm;     CKPT=mdlm.ckpt; T=0.1 ;;
  4) NAME=sfm_top1 ;;
esac

if [ "$SLURM_ARRAY_TASK_ID" = "4" ]; then
  K=$K STEPS=$STEPS BS=$BS DATA=$DATA CACHE=$CACHE DEVICES=4 \
    OUT=$PWD/eval_runs/shardB/$NAME bash scripts/sample/tinygsm/passk_sfm.sh
else
  python -u -m main \
    mode=gsm8k_eval eval.checkpoint_path=$PWD/checkpoints/tinygsm/$CKPT \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir=$CACHE data.data_path=$DATA \
    model=small model.length=512 algo=$ALGO sampler=ancestral \
    sampler.steps=$STEPS sampler.temperature=$T sampler.use_float64=$FP64 \
    loader.eval_batch_size=$BS loader.num_workers=8 \
    trainer.num_nodes=1 trainer.devices=4 \
    gsm8k.num_samples=$K gsm8k.output_dir=$PWD/eval_runs/shardB/$NAME \
    +wandb.offline=True hydra.run.dir=$PWD/eval_runs/shardB/$NAME
fi
echo "DONE $NAME"
