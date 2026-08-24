#!/bin/bash
# CoBit cell for shard A -- the SAME random 256 problems the baselines run on.
# beta=1, no resampling => K independent churn samples per problem.
set -u
cd /home/gb511/BitstreamDiffusion-fkc
~/miniconda3/envs/pytorch/bin/python -m evaluation.tasks.gsm8k_eval \
  --config configs/tasks/tinygsm_bits.py \
  --checkpoint runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000425000.pt \
  --gsm8k_test_path /home/gb511/s-flm/data_gsm8k_shardA_256.json \
  --sampler_kind fkc_em --proposal edm_churn --churn_gamma 0.41 \
  --beta 1.0 --num_particles ${K:-32} --resampling_policy never --final_resample 0 \
  --steps 1024 --limit 256 --batch_size 2 --ema 1 --seed 42 \
  --sigma_data 0.399844765663147 \
  --out_dir runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_shardA
