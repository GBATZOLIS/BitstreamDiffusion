# s-flm patches — GSM8K pass@k for the baselines

The baselines (MDLM, Duo, FLM, CANDI, S-FLM, AR) live in the S-FLM authors' repo, which also
ships their TinyGSM checkpoints. We modified that repo to add multi-sample evaluation; the
changes are vendored here so they can be reapplied to a fresh clone.

## Apply

```bash
cd ~/  && git clone https://github.com/jdeschena/s-flm.git && cd s-flm
git apply /path/to/BitstreamDiffusion/external/s-flm-patches/passk_eval.patch
cp /path/to/BitstreamDiffusion/external/s-flm-patches/{merge_passk.py,passk.sh,passk_sfm.sh} .
mv passk.sh passk_sfm.sh scripts/sample/tinygsm/
hf download jdeschena/s-flm --local-dir ./checkpoints --include 'tinygsm/**'   # ~25 GB
```

## What the patch does

* `configs/config.yaml` — adds `gsm8k.num_samples` (K samples per problem; 1 = their original
  single-sample behaviour, so their published numbers stay reproducible).
* `main.py` — repeats the sampling pass K times, tags each record with `problem_idx` / `rep` /
  `correct`, and computes **pass@k for all k ≤ K** with the unbiased Codex estimator
  `1 − C(n−c,k)/C(n,k)` plus **maj@K** by plurality over *executed answers*. Results land in
  `results.json` under `multi_sample`.
* `sandbox_gsm8k.py` — replaced with the CoBit copy. The two files were **byte-identical**
  except that ours factors out `predict_answer()` (returns the executed numeric answer rather
  than a bool), which multi-sample voting needs. Grading semantics are unchanged, and using one
  file for both sides means every method in the comparison is graded by the same code.

## Run

```bash
# Duo / MDLM (ancestral sampler). TEMP=0.1 is their optimized low-temperature setting.
PY=~/miniconda3/envs/sfm2/bin/python ALGO=duo-base CKPT=$PWD/checkpoints/tinygsm/duo.ckpt \
  TEMP=0.1 K=32 STEPS=1024 BS=128 DATA=$PWD/data_gsm8k_test_full.json \
  bash scripts/sample/tinygsm/passk.sh

# S-FLM (needs the sphere-arch config family; defaults to top-1 velocity = their best GSM8K)
K=32 STEPS=1024 BS=128 bash scripts/sample/tinygsm/passk_sfm.sh
```

## Environment

`flash_attn` is load-bearing (rotary + attention). Building it against a mismatched CUDA fails:
pip's `torch` wheel was CUDA 13 while csic47's `nvcc` is 12.1. What worked:

```bash
conda create -y --clone pytorch -n sfm2        # torch 2.5.1 / cu121 / flash_attn 2.5.9
~/miniconda3/envs/sfm2/bin/pip install hydra-core==1.3.2 omegaconf==2.3.0 \
  transformers==4.45.0 tokenizers==0.20.3 datasets==3.5.0 huggingface-hub==0.30.2 \
  einops fancy-einsum timm rich termcolor torchmetrics matplotlib seaborn blobfile
```

On CSD3 prefer the NGC PyTorch container, which the s-flm README says they develop in and which
ships flash_attn.

## Sharding

`data_gsm8k_shard_manifest.json` records the seed-0 random split of the 1319 test problems into
shard A (256) and shard B (1063). Shards are disjoint and tile the full set, so runs on
different machines merge by concatenation:

```bash
python merge_passk.py merged.json shardA/results.json shardB/results.json
```

Every shard must share checkpoint, K, steps, temperature **and precision** — only the problem
set may differ. `merge_passk.py` warns if the merged count is not 1319.
