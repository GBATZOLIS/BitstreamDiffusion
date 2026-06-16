# Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion

Official implementation of the paper:

> **Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion**
> Georgios Batzolis, Mark Girolami, Luca Ambrogioni
> *arXiv preprint, 2026.* — [arXiv:2605.07013](https://arxiv.org/abs/2605.07013)

We model text as a continuous diffusion process over fixed-width binary bitstreams. Semantic tokens are encoded as analog bit sequences; a matched-filter residual parameterization isolates contextual learning from analytic independent-bit posteriors; and an entropy-rate-gated stochastic sampler concentrates Langevin-type corrections in information-active noise regions. The resulting 130M-parameter model (**CoBit-S**) reaches GenPPL = **59.76** at matched real-data entropy on LM1B and GenPPL = **27.06** at entropy 5.26 on OpenWebText, both at 256 NFEs. Scaling to 462M (**CoBit-M**) reaches GenPPL = **19.48** at entropy 5.40 (256 NFE) and **9.87** at entropy 5.25 (512 NFE) on OpenWebText.

---

## Headline results

### CoBit-S (130M) — Table 1

All rows below come from a single 130M-parameter SDT trunk per dataset; only the sampler differs. Both models use the matched-filter residual parameterization, binary score matching with EDM weighting, and an entropy-rate noise schedule. See the paper for details.

| Dataset | Sampler | NFE | GenPPL ↓ | Entropy | Reproduces |
|---|---|---|---|---|---|
| LM1B real data | — | — | 53.06 | 4.31 | reference |
| LM1B | Deterministic (probability flow) | 256 | 82.90 ± 1.11 | 4.30 | Table 1 |
| LM1B | **Stochastic (γ=0.200, full band)** | 256 | **59.76 ± 0.57** | **4.31** | Table 1 |
| OWT real data | — | — | 15.06 | 5.44 | reference |
| OWT | Deterministic (probability flow) | 256 | 46.32 ± 0.93 | 5.13 | Table 1 |
| OWT | **Stochastic (γ=0.130, full band)** | 256 | **27.06 ± 0.57** | **5.26** | Table 1 (headline) |
| OWT | Stochastic (γ=0.180, full band) | 256 | 34.35 | 5.32 | §4.2 / Table 14 (high-entropy) |

### CoBit-M (462M) — Table 2 (OpenWebText scaling)

Scaling the same recipe to a 462M SDT trunk (`embed_dim 1024`, `n_blocks 24`,
`n_heads 16`) moves the GenPPL–entropy frontier substantially. All rows are the
EMA `step=000750000` checkpoint under the entropy-rate stochastic sampler;
only the NFE budget / churn differs. Real OWT reference: GenPPL 15.07, entropy 5.44.

| Sampler | NFE | γ | GenPPL ↓ | Entropy | Reproduces |
|---|---|---|---|---|---|
| **Stochastic (full band)** | 256 | 0.21 | **19.48** | 5.40 | Table 2 (plotted near-real-entropy point) |
| Stochastic (full band) | 256 | 0.13 | 18.47 | 5.378 | Table 2 caption (low-perplexity 256-NFE point) |
| Stochastic (full band) | 384 | 0.24 | 13.06 | 5.33 | Table 2 |
| **Stochastic (full band)** | 512 | 0.26 | **9.87** | 5.25 | Table 2 |

Reproduce every row in one run with [`configs/owt/eval_cobit_m_750K.py`](configs/owt/eval_cobit_m_750K.py) (see [Step 3](#step-3--evaluate)).

---

## Repository layout

```
BitstreamDiffusion/
├── train.py                       # entry point for training (DDP via torchrun)
├── trainers/                      # training loop + checkpointing
├── models/
│   ├── sdt.py                     # Sequence Diffusion Transformer (the model)
│   ├── sedd_wrapper.py            # SEDD baseline wrapper
│   ├── autoregressive/            # AR Transformer baseline
│   └── backbones/                 # SEDD backbones
├── diffusion/
│   ├── continuous/                # VE forward, score loss, DDIM/Heun samplers
│   └── discrete/                  # SEDD-style absorbing diffusion (baselines)
├── data/
│   ├── lm1b.py                    # LM1B dataset + bit packing
│   └── openwebtext.py             # OWT dataset + 16-bit codec
├── evaluation/                    # GenPPL, MAUVE, VLB, entropy estimators
├── utils/                         # EMA, ECC, optim, callbacks, text decoders
├── configs/
│   ├── lm1b/continuous/
│   │   ├── rate_bits_1M_edm_weight.py            # training
│   │   └── eval/rate_eval_seeds.py               # evaluation (Table 1)
│   └── owt/
│       ├── rate_bits_edm_weight.py                      # CoBit-S training (130M)
│       ├── eval_750K_seed.py                            # CoBit-S eval (Table 1 + high-entropy)
│       ├── rate_bits_edm_weight_medium_24x1024.py       # CoBit-M training (462M)
│       └── eval_cobit_m_750K.py                         # CoBit-M eval (Table 2)
└── scripts/
    ├── lm1b/                      # LM1B download + cache + semantic map
    ├── owt/                       # OWT codec training + cache prebuild
    ├── profile_*.py               # efficiency benchmarks (Table 2)
    ├── smoketest_lm1b.sh          # one-seed reproducibility check (SLURM)
    └── smoketest_owt.sh
```

---

## Installation

Tested with Python 3.10 and CUDA 12.x. We recommend a fresh conda environment.

```bash
conda create -n bitstream python=3.10 -y
conda activate bitstream

# IMPORTANT: use `python -m pip ...` rather than bare `pip ...`. On many
# clusters the `pip` shim resolves to the system Python instead of the
# conda one, silently installing everything into the wrong interpreter.
# Confirm with: `which python && python -m pip --version`.

# PyTorch — pick the build matching your CUDA toolkit
# (replace cu121 with your CUDA version; see https://pytorch.org/get-started)
python -m pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
python -m pip install -r requirements.txt

# Optional: FlashAttention 2 for ~30% faster training/sampling.
# Skip this if your hardware / toolchain makes the build painful — the
# code transparently falls back to SDPA (see note below).
python -m pip install flash-attn --no-build-isolation
```

`flash_attn` is genuinely optional: the import is guarded and the attention
layer transparently falls back to `torch.nn.functional.scaled_dot_product_attention`
([models/backbones/official_sedd.py](models/backbones/official_sedd.py)) if
the package is not installed.

The image-FID dependency `pytorch-fid` is **not** installed by default — its
pinned `torchvision` requirement conflicts with the `torch==2.4.*` build we
recommend, and it is only used for an unrelated image-FID evaluator that the
text experiments in this paper never invoke. If you need it, install it
separately:

```bash
python -m pip install pytorch-fid
```

---

## Quick start: reproducing the paper

The fastest path is to download the released checkpoints and pre-built dataset caches, then run the evaluation configs. Total reviewer time: ~1.5h on a single H100/GH200 for LM1B and ~2.5h for OWT.

### Step 1 — Download released artifacts

We release the trained checkpoints and the OWT second-stage code tokenizer on
the [Hugging Face Hub](https://huggingface.co/gbatzolis/CoBit). All checkpoints
are **EMA weights** — evaluate them with the default settings (the eval configs
load them with `apply_ema=True`, which is a no-op on these baked weights).

| Artifact | Model | Size | Destination |
|---|---|---|---|
| `checkpoints/cobit_s_lm1b_1M_ema.pt`   | CoBit-S LM1B (1M)   | 0.53 GB | `runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/checkpoints/step=001000000.pt` |
| `checkpoints/cobit_s_owt_750k_ema.pt`  | CoBit-S OWT (750K)  | 0.54 GB | `runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M/checkpoints/step=000750000.pt` |
| `checkpoints/cobit_m_owt_750k_ema.pt`  | **CoBit-M OWT (750K)** | 1.85 GB | `runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024/checkpoints/step=000750000.pt` |
| `tokenizer/…_base1024.json` + `.meta.json` | OWT 16-bit code tokenizer | 2.2 MB | `datasets/openwebtext_gpt2_trainm100k/` |
| `entropy_tables/{lm1b,owt,owt_medium}/…` | entropy-rate schedules | ~7 KB ea. | `assets/entropy_tables/` (also ships in this repo) |

#### One-shot download (places files in the exact expected paths)

```bash
python -m pip install "huggingface_hub>=0.23"

# All three checkpoints + the OWT code tokenizer:
python scripts/download_from_hf.py --repo-id gbatzolis/CoBit

# …or just the CoBit-M (462M) model:
python scripts/download_from_hf.py --repo-id gbatzolis/CoBit --models cobit_m
```

`scripts/download_from_hf.py` resolves each file from the Hub and copies it into
the directory the training / evaluation configs read — no manual path edits. The
entropy-rate tables already ship in this repository under
[`assets/entropy_tables/`](assets/entropy_tables/), so they are not fetched by
default (pass `--entropy-tables` to restore them if you deleted the in-repo copy).

#### A note on the entropy schedule artefacts

The entropy-rate noise schedule used in the paper (Eq. 13, Figure 4) is
**dataset-specific** — the LM1B and OWT models were trained with different
entropy profiles, and the sampler needs the matching profile at evaluation
time. Each profile is four small `.pt` files (~7 KB total per dataset):

```
entropy_pdf.pt      # bin probabilities of the entropy-rate distribution
entropy_cdf.pt      # cumulative integral, used by the entropic σ-schedule
entropy_sigmas.pt   # bin midpoints in σ-space
entropy_edges.pt    # bin edges in σ-space
```

These files **ship with the repository** under
[`assets/entropy_tables/lm1b/`](assets/entropy_tables/lm1b/),
[`assets/entropy_tables/owt/`](assets/entropy_tables/owt/) (CoBit-S), and
[`assets/entropy_tables/owt_medium/`](assets/entropy_tables/owt_medium/)
(CoBit-M), and every eval config references the matching path via
`cfg.evaluation.entropy_run_dir`. The CoBit-M profile is **distinct** from the
CoBit-S OWT profile — the 462M model was trained with its own entropy-rate
schedule, so `eval_cobit_m_750K.py` points at `owt_medium`, not `owt`. There's
normally nothing to download; the Hub copy is a redundant mirror.

> ⚠️  If the entropy tables for the requested dataset are missing, the
> sampler **hard-errors** rather than silently falling back to a Karras
> schedule. The Karras fallback typically degrades GenPPL by 30–40 points
> on LM1B (see Figure 4 in the paper appendix).

If you train your own model with `train.py`, the trainer writes its own
`entropy_{pdf,cdf,sigmas,edges}.pt` into the training `run_dir`. To
evaluate that model rather than the released checkpoint, edit your eval
config and either remove the `cfg.evaluation.entropy_run_dir` line
(default: derived from `cfg.evaluation.checkpoint_path`) or set it to
your training run directory.

### Step 2 — Prepare datasets

We do not host the raw dataset caches (they are large and trivially regenerable). Build them from scratch with the scripts below — the resulting layout is exactly what the configs expect, including the OWT tokenizer files you downloaded in Step 1.

Expected on-disk layout once Step 2 completes:

```
datasets/lm1b/
  cache_train_tokens.uint16
  cache_train_tokens.meta.json
  cache_test_tokens.uint16
  cache_test_tokens.meta.json
  train.txt
  test.txt

datasets/openwebtext_gpt2_trainm100k/
  cache_train_gpt2_flat_eos1.uint16
  cache_train_gpt2_flat_eos1.meta.json
  cache_val_gpt2_flat_eos1.uint16
  cache_val_gpt2_flat_eos1.meta.json
  cache_train_gpt2_gpt2id_bpe16_v65536_base1024_len1024.uint16
  cache_train_gpt2_gpt2id_bpe16_v65536_base1024_len1024.meta.json
  cache_val_gpt2_gpt2id_bpe16_v65536_base1024_len1024.uint16
  cache_val_gpt2_gpt2id_bpe16_v65536_base1024_len1024.meta.json
  tokenizer_gpt2id_bpe16_65536_base1024.json         # from Step 1
  tokenizer_gpt2id_bpe16_65536_base1024.meta.json    # from Step 1
```

#### Build the caches

**LM1B** — three steps (≈1h on a single CPU):

```bash
# (1) Download LM1B and write packed line files
python -m scripts.lm1b.download_lm1b --out_dir datasets/lm1b

# (2) Build BERT-tokenized fixed-block caches (128 tokens / block)
python -m scripts.lm1b.build_lm1b_bert_caches \
  --root datasets/lm1b \
  --tokenizer_name bert-base-uncased \
  --seq_len_tokens 128 \
  --boundary_mode sep
```

A semantic-map step (`scripts.lm1b.prepare_lm1b_semantic_map`) is also provided for the semantic-bit ablations in the paper, but it is **not required** for the headline `raw_binary` results.

**OpenWebText** — two stages: build the GPT-2 flat cache, then download (recommended) or re-train the 16-bit code tokenizer, then materialize the second-stage cache.

```bash
# (1) Build the GPT-2 flat token caches (train/val).
#     This downloads OpenWebText via 🤗 datasets and packs GPT-2 ids contiguously.
python -m scripts.owt.prebuild_owt_caches \
  --config configs/owt/rate_bits_edm_weight.py

# (2) Train the second-stage 16-bit code tokenizer (gpt2id_bpe16, |V|=65,536).
#     SKIP THIS STEP if you downloaded the released tokenizer in Step 1 — it's deterministic
#     but takes a few hours of CPU time.
python -m scripts.owt.train_owt_gpt2id_bpe16 \
  --root datasets/openwebtext_gpt2_trainm100k \
  --tokenizer_name gpt2 \
  --base_seq_len_tokens 1024 \
  --code_vocab_size 65536 \
  --split_name train

# (3) Re-run prebuild to materialize the gpt2id_bpe16 second-stage cache,
#     now that the codec is in place.
python -m scripts.owt.prebuild_owt_caches \
  --config configs/owt/rate_bits_edm_weight.py
```

### Step 3 — Evaluate

The evaluation configs already encode the exact sampler settings (NFE = 256, full-band entropy window, S_noise = 1.003, `ati_eta = 0`) used to produce the headline numbers. Each config sweeps the operating points reported in the paper.

```bash
# Common environment
export EVAL_SEED=42
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/hf_cache"
```

**LM1B (reproduces Table 1, LM1B Stochastic row, γ = 0.200):**

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m evaluation.run_eval \
  --config configs/lm1b/continuous/eval/rate_eval_seeds.py \
  --metrics external_ppl

python -m evaluation.compute_entropy_from_caches \
  --config configs/lm1b/continuous/eval/rate_eval_seeds.py \
  --include_real
```

The config also runs a γ = 0.185 spec, which produces a nearby Pareto point.
Outputs land at `runs/.../lm1b/.../evaluation_cleanup_smoketest/results.{csv,jsonl}`.
Expected on a single seed: GenPPL ≈ 59–60 at entropy ≈ 4.31 (paper: 59.76 ± 0.57 across 10 seeds).

**OWT (reproduces Table 1 OWT Stochastic + the §4.2 high-entropy point):**

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m evaluation.run_eval \
  --config configs/owt/eval_750K_seed.py \
  --metrics external_ppl

python -m evaluation.compute_entropy_from_caches \
  --config configs/owt/eval_750K_seed.py \
  --include_real
```

This sweeps three operating points in a single run:

| γ | NFE | Reproduces | Expected (single seed) |
|---|---|---|---|
| 0.130 | 256 | Table 1 OWT Stochastic | GenPPL ≈ 27, entropy ≈ 5.26 |
| 0.175 | 256 | Frontier point | GenPPL ≈ 34, entropy ≈ 5.31 |
| 0.180 | 256 | §4.2 high-entropy / Table 14 sample | GenPPL ≈ 34, entropy ≈ 5.32 |

**CoBit-M / 462M (reproduces Table 2, OpenWebText scaling):**

```bash
# All three Table-2 rows (256/384/512 NFE) in one run. Add the low-PPL
# 256-NFE caption point with EVAL_CELLS=all.
bash scripts/owt/eval_cobit_m.sh                 # 2 GPUs (default)
NPROC=1 bash scripts/owt/eval_cobit_m.sh         # single GPU
EVAL_CELLS=all bash scripts/owt/eval_cobit_m.sh  # + the γ=0.13 caption point
```

The launcher runs generation + GenPPL and then the post-hoc token-unigram
entropy estimator. Equivalent explicit form:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m evaluation.run_eval --config configs/owt/eval_cobit_m_750K.py --metrics external_ppl
python -m evaluation.compute_entropy_from_caches --config configs/owt/eval_cobit_m_750K.py
```

This sweeps the Table-2 operating points in a single run (`EVAL_CELLS=table2`):

| γ | NFE | Reproduces | Expected (single seed) |
|---|---|---|---|
| 0.21 | 256 | Table 2 (plotted point) | GenPPL ≈ 19.5, entropy ≈ 5.40 |
| 0.24 | 384 | Table 2 | GenPPL ≈ 13.1, entropy ≈ 5.33 |
| 0.26 | 512 | Table 2 | GenPPL ≈ 9.9, entropy ≈ 5.25 |
| 0.13 | 256 | Table 2 caption (low-PPL) | GenPPL ≈ 18.5, entropy ≈ 5.38 |

Outputs land at `runs/.../continuous_rate_raw_binary_bits_medium_24x1024/evaluation_cobit_m_table2_step000750000/results.{csv,jsonl}`. GenPPL agrees with the paper to within seed/hardware noise (Table-1-style ±0.5).

#### Multi-seed evaluation

The paper reports mean ± std across 10 seeds. To replicate, loop the seed:

```bash
for s in {42..51}; do
  EVAL_SEED=$s torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m evaluation.run_eval \
    --config configs/owt/eval_750K_seed.py \
    --metrics external_ppl
done
python -m evaluation.compute_entropy_from_caches \
  --config configs/owt/eval_750K_seed.py --include_real
```

Provided SLURM templates [`scripts/smoketest_lm1b.sh`](scripts/smoketest_lm1b.sh) and [`scripts/smoketest_owt.sh`](scripts/smoketest_owt.sh) demonstrate the single-seed run on a 1-GPU partition.

---

## Training from scratch

The CoBit-S training configs implement the exact protocol described in Appendix B of the paper: 12-block / 768-d SDT trunk, AdamW (lr = 3 × 10⁻⁴, cosine decay, 2.5K warmup), global batch size 512, BF16, EDM loss weighting, entropy-rate noise schedule with 40K-step warmup and 10K-step transition, self-conditioning probability 0.5 with carry-mode at sampling. CoBit-M keeps this protocol unchanged except for the trunk (1024-d / 24 blocks / 16 heads), a more conservative lr = 2 × 10⁻⁴ and 5K warmup at the larger scale.

The hardware below matches what we actually used to produce the released checkpoints. Other configurations (more or fewer GPUs) work identically as long as the per-step global batch size stays at 512 — the trainer divides the global batch evenly across the world size.

```bash
# LM1B — 1,000,000 optimizer steps. ~2 days on 2 × NVIDIA GH200, global batch 512.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train.py --config configs/lm1b/continuous/rate_bits_1M_edm_weight.py

# OpenWebText — CoBit-S (130M). ~6 days on 4 × NVIDIA GH200, global batch 512.
# Paper-reported numbers use the step=000750000.pt checkpoint; training continues to 1M.
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py --config configs/owt/rate_bits_edm_weight.py

# OpenWebText — CoBit-M (462M, Table 2). 2 × GH200 nodes (8 GPUs), global batch 512.
# Same recipe; only the trunk grows (1024-d / 24 blocks / 16 heads) and lr 2e-4 / warmup 5K.
# Table-2 numbers use the step=000750000.pt checkpoint.
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config configs/owt/rate_bits_edm_weight_medium_24x1024.py
```

Checkpoints are written every 50,000 steps (and a rolling `last.pt` every 5,000 steps for resume) to `runs/<cfg.experiment>/checkpoints/`. Training auto-resumes from `last.pt` if the run directory already exists.

Multi-node DDP is supported via standard `torchrun` flags (`--nnodes`, `--nproc_per_node`, `--rdzv_endpoint`, etc.). The trainer initializes `nccl` with a 20-minute timeout.

---

## Configuration files at a glance

| File | Purpose |
|---|---|
| [`configs/lm1b/continuous/rate_bits_1M_edm_weight.py`](configs/lm1b/continuous/rate_bits_1M_edm_weight.py) | LM1B training (1M steps) |
| [`configs/lm1b/continuous/eval/rate_eval_seeds.py`](configs/lm1b/continuous/eval/rate_eval_seeds.py) | LM1B evaluation: γ ∈ {0.185, 0.200} stochastic specs |
| [`configs/owt/rate_bits_edm_weight.py`](configs/owt/rate_bits_edm_weight.py) | CoBit-S OWT training (1M-step schedule; 750K used in paper) |
| [`configs/owt/eval_750K_seed.py`](configs/owt/eval_750K_seed.py) | CoBit-S OWT evaluation: γ ∈ {0.13, 0.175, 0.18} stochastic specs (Table 1) |
| [`configs/owt/rate_bits_edm_weight_medium_24x1024.py`](configs/owt/rate_bits_edm_weight_medium_24x1024.py) | CoBit-M OWT training (462M; 750K used in paper) |
| [`configs/owt/eval_cobit_m_750K.py`](configs/owt/eval_cobit_m_750K.py) | CoBit-M OWT evaluation: 256/384/512-NFE Table-2 operating points |

The discrete-diffusion bitstream baseline (Appendix C) is exposed via `configs/lm1b/discrete/`.

---

## Sampling specifics

Generated samples are produced by:

1. integrating the probability-flow ODE in Eq. (8) of the paper on the entropy-rate sigma grid (DDIM-style, 256 steps);
2. interleaving EDM-style stochastic churn (`gamma_target` and `s_noise = 1.003`) over the full entropy band `q ∈ [0, 1]`;
3. self-conditioning in carry mode;
4. terminal sigma `σ_term = 0.0794`, `ati_eta = 0`.

Each generation produces 1024 samples per spec. Generative perplexity is computed under `openai-community/gpt2-large`. Token-frequency entropy is computed on `bert-base-uncased` IDs for LM1B and on inverse-decoded GPT-2 IDs for OWT (so entropies are directly comparable to GPT-2-tokenized baselines, not to our intermediate 16-bit code vocabulary).

---

## Citation

```bibtex
@misc{batzolis2026bitstream,
  title         = {Towards Closing the Autoregressive Gap in Language Modeling
                   via Entropy-Gated Continuous Bitstream Diffusion},
  author        = {Batzolis, Georgios and Girolami, Mark and Ambrogioni, Luca},
  year          = {2026},
  eprint        = {2605.07013},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## Acknowledgements

This codebase builds on ideas and reference implementations from EDM (Karras et al., 2022), Analog Bits (Chen et al., 2022), SEDD (Lou et al., 2024), CDCD (Dieleman et al., 2022), and the FLM / LangFlow continuous-DLM family. We thank the authors of these works for making their code and analyses public.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
