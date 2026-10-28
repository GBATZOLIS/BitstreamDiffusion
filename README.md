# CoBit - Continuous Bitstream Diffusion

Every modality is written as **bits**. The same input projection, transformer trunk, bitwise prediction head and binary score-matching objective serves all of them The denoiser is modality agnostic, both at training and inference time. Modality-specific design is only needed for tokenization and the input sequence structure.

**Check out our [demo](https://georgekontos14.github.io/MultimodalCobitDemo/)**

This repository contains official implementations of the following papers:

| Paper | Scope | arXiv
|---|---|---|
| **Multimodal Co-Generation with Continuous Bitstram Diffusion** | image+text<br>audio+text<br>protein sequence+structure | [arXiv:2610.14349](https://arxiv.org/abs/2610.14349) | 
| **CoBit: Language Modeling with Bitstream Diffusion** | LM1B, OpenWebText | [arXiv:2605.07013](https://arxiv.org/abs/2605.07013)

Additional language modeling documentation: [docs/README_language_modeling.md](docs/README_language_modeling.md)

A single checkpoint per modality pair performs **joint generation and every
conditional direction it was trained on** — you change which bits are observed,
not which model you load.

---

## Released Checkpoints

**Multimodal checkpoints**

| Experiment | Model | Params | Directions from one checkpoint | Status |
|---|---|---|---|---|
| **image+text** (MNIST-Sum) | `mnist_sum_p28_12x512` | 63.0M | image→text, text→image, joint | **released** |
| **speech+text** | CoBit-LibriTTS | 134M | TTS, ASR, continuation, joint | **released** |
| **speech+text** | CoBit-MLS | 633M | TTS, ASR, continuation, joint | **released** |
| protein sequence+structure | — | 396M | forward folding, inverse folding, joint | *to be added* |

All artifacts live in one Google Drive folder, organised as
`<modality-pair>/<run-name>/`:

> **[CoBit released checkpoints](https://drive.google.com/drive/folders/1-6RPouCc-xJnE_6h6PT-l2GO_Eputx-o)**

Each run directory has the same shape, and the layout is not cosmetic — the
sampler resolves the entropy tables **relative to the checkpoint**, so the files
must stay together:

```
<run-name>/
├── checkpoints/            # the evaluated checkpoint
├── config.json             # resolved config as trained
├── original_config.py      # the config file as written
├── entropy_{pdf,cdf,sigmas,edges}.pt   # entropy-rate schedule  <-- required
├── training_logs.zip       # TensorBoard event files
└── evaluation/             # the results.json behind the paper table
```

**Language checkpoints**

We release the trained checkpoints and the OWT second-stage code tokenizer on
the [Hugging Face Hub](https://huggingface.co/gbatzolis/CoBit). All checkpoints
are **EMA weights** — evaluate them with the default settings (the eval configs
load them with `apply_ema=True`, which is a no-op on these baked weights).

|Experiment | Model | Artifact | Params/Size | Destination
|---|---|---|---|---|
| LM1B | CoBit-S (1M) | `checkpoints/cobit_s_lm1b_1M_ema.pt` | 134M / 0.53 GB | `runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/checkpoints/step=001000000.pt` |
| OWT | CoBit-S (750K) | `checkpoints/cobit_s_owt_750k_ema.pt` | 134M / 0.54 GB |  `runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M/checkpoints/step=000750000.pt` |
| OWT | CoBit-M (750K) | `checkpoints/cobit_m_owt_750k_ema.pt` | 462M / 1.85 GB | `runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024/checkpoints/step=000750000.pt` |
| OWT | 16-bit code tokenizer | `tokenizer/…_base1024.json` + `.meta.json` | 2.2 MB | `datasets/openwebtext_gpt2_trainm100k/` |
| --- | entropy-rate schedules | `entropy_tables/{lm1b,owt,owt_medium}/…` | ~7 KB ea. | `assets/entropy_tables/` (also ships in this repo) |

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

> ⚠️ **The entropy tables are not optional.** The entropy-rate noise schedule is
> estimated online *per run*, and the sampler needs that run's profile. If they
> are missing, this codebase **raises** rather than silently falling back to a
> Karras grid. Do not work around it: on the text models the fallback costs
> 30–40 GenPPL, and on MNIST-Sum it makes every number incomparable.

---

## Installation

```bash
conda create -n bitstream python=3.11 -y
conda activate bitstream

# Confirm that pip belongs to the activated interpreter.
which python
python -m pip --version

# Core training environment.
./setup.sh train
```

| Profile | Command | Additional dependencies |
|---|---|---|
| Core training | `./setup.sh train` | Base training stack |
| Text-only | `./setup.sh text` | Gensim and text evaluation |
| MNIST-Sum | `./setup.sh mnist-sum` | Released-artifact downloader |
| Text-audio | `./setup.sh text-audio` | Torchaudio and audio evaluation stack |
| Protein training | `./setup.sh protein` | Base training stack |
| Protein evaluation | `./setup.sh protein-eval` | Folding and structural metrics; downloads evaluator models |
| Development | `./setup.sh dev` | All profiles, pytest and Ruff |

To target a different official PyTorch wheel index, set
`PYTORCH_INDEX_URL` before running the script. For example:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup.sh train
```

FlashAttention remains optional for the main model and can be installed after
the selected profile when the CUDA compiler and GPU support it:

```bash
python -m pip install flash-attn --no-build-isolation
```

The additional speech-tokenizer and metric packages are installed separately
when needed:

```bash
bash scripts/text_audio/textaudio_install.sh
```

---

## Multimodal Experiment 1 — image+text (MNIST-Sum)

A controlled test of the multimodal interface with **no learned tokenizer on
either side**, so there is no reconstruction floor and every direction has a
unique correct answer. Four binarised MNIST digits in a 2×2 grid are paired with
their equation; the model reads and adds a supplied image, draws the four digits
a supplied equation names, or generates an agreeing pair from noise.

One flat sequence of **3,388 bits** — 3,136 raw pixel bits + 84 text bits
(7-bit codes, 40 symbols) + 168 marker bits — in 121 positions of 28 bits. The
denoiser sees that sequence and its 1-D position. No pixel grid, no 2-D
attention, no modality embedding.

### Reproduce the paper table (~1 GB download, ~50 min on one A6000)

```bash
# 1. fetch checkpoint + entropy tables + classifier + evaluation corpus
bash scripts/mnist_sum/fetch_mnist_sum_release.sh

# 2. evaluate — all three directions, n=4,096 per direction per split
python evaluation/run_eval.py \
    --config configs/mnist_sum/mnist_sum_p28_small.py \
    --metrics mnist_sum \
    --mnist_sum_checkpoint runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt \
    --mnist_sum_out results/mnist_sum_p28_500k \
    --mnist_sum_n 4096 --mnist_sum_n_joint 4096

# 3. check every cell against the published intervals
python scripts/mnist_sum/check_mnist_sum_results.py \
    results/mnist_sum_p28_500k/results.json
```

Step 3 prints a pass/fail line per cell and exits non-zero if anything lands
outside its published 95% interval. It also checks the *protocol* first —
`karras_fallback`, `n`, step counts, EMA, `torch.compile` and the checkpoint
step — because a number produced under different settings is not a failed
reproduction, it is a different measurement.

### Expected result

n=4,096 per direction per split; image→text and text→image at guidance 2.0,
joint unguided; 95% Wilson intervals.

| Direction | val | held-out |
|---|---|---|
| image→text, exact string match | **97.56** [97.04, 97.99] | **97.85** [97.36, 98.25] |
| text→image, all four digits | **99.95** [99.82, 99.99] | **99.93** [99.78, 99.98] |
| joint from noise, mutually consistent | **99.46** [99.19, 99.65] | — |

Joint decomposes into 100.00% well-formed equations, 99.71% whose named addends
match the generated image, and 99.76% whose arithmetic is valid.

### Three things that will silently corrupt this number

1. **Use `step=000500000.pt`, never `best.pt`.** `best.pt` is selected on
   validation loss, and the entropy-schedule warmup at step 40k raises the loss
   floor from ~0.086 to ~0.222. No later epoch can beat a pre-warmup loss, so
   `best.pt` is frozen at ~step 31k — roughly 10 accuracy points below the
   final model. The fetch script only downloads the right one.
2. **Do not retrain the quadrant classifier.** `runs/mnist_sum_quadrant_cnn.pt`
   is the instrument that reads text→image and the image half of joint. Two arms
   read by two instruments is not a comparison. The offline harness will train
   one if the file is absent — the fetch script places the released one first.
3. **Read text→image against the train-pool ceiling of 99.78%, not 96.09%.**
   The corpus builds val / held-out composites from MNIST's *test* digits while
   the classifier is fitted on the *train* pool, and generated digits imitate the
   pool the classifier knows. Both ceilings are in `results.json`; against 96.09
   the model reads as "104% of ceiling", which is meaningless.

### Train from scratch (~40 h on 2× A6000)

```bash
# Build the corpus (deterministic, seed 42; downloads MNIST) -- or pass --full
# to the fetch script to download the exact 1M-example corpus we trained on.
python scripts/mnist_sum/phase_50_mnist_sum_dataset.py \
    --out datasets/mnist_sum_p28 --patch_size 28

# 500,000 steps, global batch 512, ~282 ms/step
torchrun --nproc_per_node=2 train.py \
    --config configs/mnist_sum/mnist_sum_p28_small.py
```

The run writes its own `entropy_{pdf,cdf,sigmas,edges}.pt` into the run
directory from step 2,000 onward; evaluate against *those*, not the released
ones. Image→text accuracy is logged to TensorBoard under `mnist_sum/` at every
50k steps and saturates by ~150k.

The config is derived from the CC3M recipe by import-and-override
([`configs/mnist_sum/_mnist_sum_base.py`](configs/mnist_sum/_mnist_sum_base.py)),
so everything not listed in its `OVERRIDES` docstring is byte-identical to CC3M
by construction. The only overrides are data geometry and trunk scale: **nothing
in `cfg.model.*` is overridden**, so the backbone is the stock one, identical to
the speech models.

---

## Multimodal Experiment 2 — speech+text

Two checkpoints, each performing text→speech, speech→text, speech continuation
and joint co-generation from one set of weights. Text (o200k), a 32-token
BiCodec-global speaker representation and StableCodec speech tokens share one
18-bit word.

| Model | Config | Trunk | Speech budget | Training |
|---|---|---|---|---|
| CoBit-LibriTTS (134M) | [`configs/text_audio/libritts/libritts_train.py`](configs/text_audio/libritts/libritts_train.py) | 12×768 | 800 tokens | LibriTTS, 140K steps |
| CoBit-MLS (633M) | [`configs/text_audio/mls/mls_train.py`](configs/text_audio/mls/mls_train.py) | 26×1152 | 500 tokens | LibriTTS + MLS-English, 1.2M steps |

### Setup

```bash
bash setup.sh text-audio
```

Download `audio+text/cobit-libritts/` or `audio+text/cobit-mls/` from the Drive
folder and place each run directory under `runs/` at the name its config uses —
`runs/cobit_libritts/` for CoBit-LibriTTS and `runs/cobit_mls/` for
CoBit-MLS — keeping `checkpoints/` and the four `entropy_*.pt` files together.
The evaluation corpora (LibriSpeech-PC, SALMON) are under `audio+text/datasets/`.

### Evaluate

```bash
python evaluation/run_eval.py \
    --config configs/text_audio/mls/eval_sweeps/512.py \
    --metrics textaudio_generate --compile
```

Per-direction operating points are in Appendix C.1 of the paper: TTS, ASR and
continuation all use 512 steps (1024 NFE) with guidance 5, at γ = 0.27 / 0.27 /
0.4 for CoBit-MLS; co-generation uses γ = 0.175 and no guidance. The released
`evaluation.zip` and `validation.zip` in each run directory contain the
generations and sweeps behind Tables 2 and 6–23.

> The speech evaluation depends on external judges (Whisper-large, UTMOS,
> ECAPA-TDNN, WavLM, Emotion2Vec, GPT-4o) — see Appendix C.1. Exact invocation
> for each metric is being folded into this README; until then, the sweep
> configurations recorded in the released `validation.zip` are the ground truth
> for what was run.
>
> The per-model evaluation sweeps live in
> `configs/text_audio/{libritts,mls}/eval_sweeps/`. Select the sweep config whose
> sampling budget and task set match the evaluation you want to run.

---

## Multimodal Experiment 3 — protein sequence+structure

One checkpoint performs forward folding, inverse folding and joint sequence–
structure co-generation. Each residue is represented by one 18-bit position:
5 bits encode the amino-acid sequence token and 13 bits encode the frozen DPLM
structure token.

| Model | Config | Trunk | Maximum length | Training |
|---|---|---|---|---|
| CoBit-Protein (396M) | [`configs/proteins/proteins_400m.py`](configs/proteins/proteins_400m.py) | 20×1024 | 512 residues | Paired PDB/Swiss-Prot + UniRef50 replay, 300K steps |

### Setup and data

Install the training environment, download the frozen structure-tokenizer
configuration and prepare the training datasets:

```bash
bash setup.sh protein
python scripts/proteins/download_config.py
python scripts/proteins/download_data.py train
```

The paired sequence–structure examples come from the
[DPLM PDB/Swiss-Prot dataset](https://huggingface.co/datasets/airkingbd/pdb_swissprot).
The preparation step also selects 400,000 replay sequences from
[EvoDiff UniRef50](https://zenodo.org/records/6564798). Prepared data is written
under `datasets/`.

### Train

The paper configuration uses FP32, an effective batch size of 512 and 300,000
optimizer steps. The paper run used eight H100 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/proteins/proteins_400m.py
```

Checkpoints, entropy tables and logs are stored under `runs/proteins/paper/`.
Rerunning the same command resumes from `checkpoints/last.pt` when it exists.

### Evaluate

Install the protein evaluation dependencies and frozen evaluator models, then
prepare the CATH 4.3 and CAMEO 2022 evaluation data:

```bash
bash setup.sh protein-eval
python scripts/proteins/download_data.py eval
```

Run entropy profiling, inverse folding, forward folding and co-generation from
the unified evaluation entry point:

```bash
python evaluation/run_eval.py \
    --config configs/proteins/proteins_400m.py \
    --metrics proteins \
    --protein_checkpoint runs/proteins/paper/checkpoints/best.pt
```

Entropy profiling runs first on the CATH validation split and writes the
checkpoint-specific entropy tables beside the run. Inverse folding uses CATH
test backbones, forward folding uses the CAMEO 2022 targets, and co-generation
samples lengths 100, 200, 300, 400 and 500. The saved protocol uses 50 sampling
grid points, guidance 5 for conditional tasks and entropy-windowed churn.

---

## How one model serves every direction

The parts below are shared by all three experiments — this is the whole of the
multimodal extension.

**Layout.** Each modality is written as bits, directly or through a frozen
tokenizer, and placed in a fixed sequence template that sets the order, position
and budget of each segment plus the marker positions. Nothing is learned about
where a code goes.

**Partial denoising.** Training samples a regime per example and builds a mask
`M`: observed bits stay clean, `x_σ = x_0 + (1 − M) ⊙ σε`, and the loss is
applied only to `(1 − M)`. Leaving all content bits unobserved trains joint
generation; observing one modality trains generation of the other.
The shared mask contract is in [`utils/regimes.py`](utils/regimes.py). The
layout-specific implementations are
[`utils/mnist_sum/utils.py`](utils/mnist_sum/utils.py),
[`utils/text_audio/utils.py`](utils/text_audio/utils.py), and the protein state
and dataset logic in [`data/protein_tasks.py`](data/protein_tasks.py) and
[`data/protein_multimodal.py`](data/protein_multimodal.py).

**Markers are always observed, and never dropped.** Structural marker positions
stay clean in every regime, and — this is the subtle part — they must keep their
**true codes in the CFG unconditional branch** rather than being nulled to 1/2,
which is a value the model never sees in training. That carve-out is
`protect_mask` in
[`diffusion/continuous/samplers.py`](diffusion/continuous/samplers.py); without
it, guided conditional generation degrades with no error raised.
[`tests/test_protect_mask.py`](tests/test_protect_mask.py) pins the behaviour.

**Unimplemented config flags raise.** `cfg.model.*` keys that nothing reads are
rejected at model construction
([`models/__init__.py`](models/__init__.py)). A config that silently asks for a
component the code does not have is how an architecture claim and a trained
model come apart.

---

## Repository layout

```
BitstreamDiffusion/
├── setup.sh                         # Conda environment profiles
├── train.py                         # shared training entry point (DDP via torchrun)
├── trainers/
│   ├── trainer.py                  # shared training loop
│   └── multimodal_step.py          # protein sequence–structure training step
├── models/
│   ├── sdt.py                      # Sequence Diffusion Transformer
│   └── protein_multimodal.py       # protein-specific embeddings and conditioning
├── diffusion/continuous/           # VE process, binary score loss and samplers
├── data/
│   ├── mnist_sum/                  # MNIST-Sum dataset and image+text codecs
│   ├── text_audio/                 # speech+text dataset
│   ├── protein_multimodal.py       # paired protein dataset
│   └── protein_tasks.py            # protein regimes and modality masks
├── configs/
│   ├── mnist_sum/                  # MNIST-Sum and CC3M
│   ├── text_audio/                 # CoBit-LibriTTS and CoBit-MLS
│   ├── proteins/                   # protein sequence+structure
│   └── text/{lm1b,owt}/            # language modeling
├── evaluation/
│   ├── run_eval.py                 # unified evaluation entry point
│   └── evaluation_drivers/         # MNIST-Sum, text-audio and protein drivers
├── scripts/{mnist_sum,text_audio,proteins,text}/
│                                      # preparation and experiment utilities
├── utils/
│   ├── regimes.py                  # shared conditioning-plan interface
│   ├── callbacks/                  # training-time evaluation and generation
│   └── {mnist_sum,text_audio,proteins}/
│                                      # experiment-specific utilities
├── assets/                         # fixed codecs and evaluation assets
└── docs/                           # experiment reports and language-modeling guide
```

---

## Citation

```bibtex
@article{multimodalcobit2026,
  title   = {Multimodal Co-Generation With Continuous Bitstream Diffusion},
  author  = {Batzolis, Georgios and Kontos, Georgios and Trentini, Bruno and Chatterjee, Pranam and Livne, Micha and Girolami, Mark and Ambrogioni, Luca},
  journal = {arXiv preprint arXiv:2610.14349},
  year    = {2026}
}
```

```bibtex
@article{cobit2026,
  title  = {CoBit: Language Modeling with Bitstream Diffusion},
  author = {Batzolis, Georgios and Girolami, Mark and Ambrogioni, Luca},
  journal = {arXiv preprint arXiv:2605.07013},
  year   = {2026}
}
```

