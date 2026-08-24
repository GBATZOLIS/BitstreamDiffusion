# pass@32 on CSD3 — what to run

Two array jobs cover shard B (1063 problems); shard A (256) runs on csic47.
Together they tile the full 1319-problem GSM8K test set exactly once.

## 0. One-time setup on CSD3
```bash
cd ~/rds_work/projects
git clone https://github.com/jdeschena/s-flm.git      # baselines
cd s-flm && hf download jdeschena/s-flm --local-dir ./checkpoints --include 'tinygsm/**'
# copy the LOCAL patches (pass@k wiring + shared grader + shard files):
rsync -av csic47:/home/gb511/s-flm/{main.py,sandbox_gsm8k.py,merge_passk.py} .
rsync -av csic47:/home/gb511/s-flm/data_gsm8k_shard*.json .
rsync -av csic47:/home/gb511/s-flm/scripts/sample/tinygsm/passk*.sh scripts/sample/tinygsm/
rsync -av csic47:/home/gb511/s-flm/scripts/hpc/ scripts/hpc/
# CoBit side:
cd ~/rds_work/projects/BitstreamDiffusion
git fetch origin && git checkout tasks/fkc-temperature && git pull --ff-only
```
The s-flm README says they develop inside the NGC PyTorch container, which ships
flash_attn. Building flash-attn by hand cost an hour on csic47 (pip's torch wheel
was CUDA 13 vs local nvcc 12.1) — prefer the container or an env with a matching
CUDA/torch/flash_attn stack.

## 1. Submit
```bash
cd ~/rds_work/projects/s-flm            && sbatch scripts/hpc/passk_baselines_csd3.slurm
cd ~/rds_work/projects/BitstreamDiffusion && sbatch scripts/hpc/passk_cobit_csd3.slurm
```

## 2. Merge shards (A from csic47 + B from CSD3)
```bash
python merge_passk.py merged_duo_T1.json  <shardA duo results.json> <shardB duo results.json>
```
Shards are disjoint by construction, so pass@k over the union is just the
per-problem concatenation. The script warns if the merged count != 1319.

## Non-negotiables
* K=32 for EVERY method — matched K is what makes the comparison valid.
* Same precision everywhere. `sampler.use_float64` must match between shards.
* sigma_data=0.399844765663147 for CoBit (the config default 0.5 is wrong
  and fails silently).
* Never use a test-set PREFIX as a subset: the first 256 problems run ~6 points
  hot. The shard files are a seed-0 random split.
