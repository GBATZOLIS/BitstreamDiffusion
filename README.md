# CoBit — Continuous Bitstream Diffusion

Every modality is written as **bits**. One input projection, one transformer
trunk, one bitwise prediction head and one binary score-matching objective serve
all of them. Nothing in the denoiser is modality-specific — not an embedding,
not a loss term, not a sampler.

This repository is the official implementation of two papers:

| Paper | Scope | Guide |
|---|---|---|
| **Multimodal Co-Generation with Continuous Bitstream Diffusion** | image+text, speech+text, protein sequence+structure | **this README** |
| CoBit: Language Modeling with Bitstream Diffusion — [arXiv:2605.07013](https://arxiv.org/abs/2605.07013) | LM1B, OpenWebText | [docs/README_language_modeling.md](docs/README_language_modeling.md) |

A single checkpoint per modality pair performs **joint generation and every
conditional direction it was trained on** — you change which bits are observed,
not which model you load.

---

## Released checkpoints

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

> ⚠️ **The entropy tables are not optional.** The entropy-rate noise schedule is
> estimated online *per run*, and the sampler needs that run's profile. If they
> are missing, this codebase **raises** rather than silently falling back to a
> Karras grid. Do not work around it: on the text models the fallback costs
> 30–40 GenPPL, and on MNIST-Sum it makes every number incomparable.

---

## Installation

```bash
conda create -n cobit python=3.10 && conda activate cobit

# IMPORTANT: use `python -m pip ...`, not bare `pip ...`. On many clusters the
# `pip` shim resolves to the system Python and installs into the wrong
# interpreter. Confirm with: which python && python -m pip --version

# PyTorch — pick the build matching your CUDA toolkit
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
```

**Speech+text only.** The speech stack (StableCodec, Spark-TTS/BiCodec, jiwer,
NeMo, fairseq/s3prl) is large and is *not* in `requirements.txt`:

```bash
bash textaudio_install.sh
```

Everything else — image+text, protein, LM1B, OpenWebText — runs without it.
Those imports are lazy, so a missing speech stack is only an error if you
actually ask for a speech config.

**Optional.** FlashAttention 2 gives ~30% faster training and sampling; the code
falls back to SDPA transparently if it is absent.

---

## Experiment 1 — image+text (MNIST-Sum)

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
bash scripts/multimodal/fetch_mnist_sum_release.sh

# 2. evaluate — all three directions, n=4,096 per direction per split
python scripts/multimodal/eval_mnist_sum_offline.py \
    --config configs/multimodal/mnist_sum_p28_small.py \
    --ckpt   runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt \
    --out    results/mnist_sum_p28_500k --n 4096 --n_joint 4096

# 3. check every cell against the published intervals
python scripts/multimodal/check_mnist_sum_results.py \
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
python scripts/multimodal/phase_50_mnist_sum_dataset.py --patch-size 28

# 500,000 steps, global batch 512, ~282 ms/step
torchrun --nproc_per_node=2 train.py \
    --config configs/multimodal/mnist_sum_p28_small.py
```

The run writes its own `entropy_{pdf,cdf,sigmas,edges}.pt` into the run
directory from step 2,000 onward; evaluate against *those*, not the released
ones. Image→text accuracy is logged to TensorBoard under `mnist_sum/` at every
50k steps and saturates by ~150k.

The config is derived from the CC3M recipe by import-and-override
([`configs/multimodal/_mnist_sum_base.py`](configs/multimodal/_mnist_sum_base.py)),
so everything not listed in its `OVERRIDES` docstring is byte-identical to CC3M
by construction. The only overrides are data geometry and trunk scale: **nothing
in `cfg.model.*` is overridden**, so the backbone is the stock one, identical to
the speech models.

---

## Experiment 2 — speech+text

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
bash textaudio_install.sh        # required: the speech stack is not in requirements.txt
```

Download `audio+text/cobit-libritts/` or `audio+text/cobit-mls/` from the Drive
folder and place each run directory under `runs/` at the name its config uses —
`runs/cobit_libritts/` for CoBit-LibriTTS and `runs/cobit_mls/` for
CoBit-MLS — keeping `checkpoints/` and the four `entropy_*.pt` files together.
The evaluation corpora (LibriSpeech-PC, SALMON) are under `audio+text/datasets/`.

### Evaluate

```bash
python evaluation/run_eval.py \
    --config configs/text_audio/mls/mls_train.py \
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
> The per-model evaluation sweeps now live in `configs/text_audio/{libritts,mls}/eval_sweeps/`
> (added in `f2b24d9`); the command above predates them and has not been checked against them.

---

## Experiment 3 — protein sequence+structure

*To be added.* The 396M denoiser, its DPLM-2 LFQ tokenisation (5 amino-acid bits
+ 13 structure bits per residue), the CATH / CAMEO evaluation sets and the
ProteinMPNN/ESMFold self-consistency pipeline are described in Appendix D. The
checkpoint will appear under `protein-structure+sequence/` in the same Drive
folder, with the same run-directory layout.

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
Implementations: [`utils/mnist_sum_utils.py`](utils/mnist_sum_utils.py) and
[`utils/textaudio_utils.py`](utils/textaudio_utils.py).

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
├── train.py                        # training entry point (DDP via torchrun)
├── trainers/trainer.py             # training loop, partial-denoising regimes
├── models/sdt.py                   # Sequence Diffusion Transformer
├── diffusion/continuous/           # VE forward process, binary score loss, samplers
├── data/
│   ├── mnist_sum_{codec,bits}.py   # MNIST-Sum corpus and 7-bit text codec
│   ├── mm_codec.py                 # shared 19-bit image+text codec (CC3M/CC12M)
│   └── textaudio.py                # speech+text dataset
├── configs/
│   ├── multimodal/                 # MNIST-Sum, CC3M
│   ├── textaudio/                  # CoBit-LibriTTS, CoBit-MLS
│   ├── lm1b/  owt/                 # the language-modeling paper
├── evaluation/
│   ├── mnist_sum{,_classifier}.py  # exact scorers + quadrant CNN
│   └── run_eval.py                 # unified evaluation entry point
├── scripts/multimodal/             # MNIST-Sum corpus builder, harness, fetch/check
├── utils/schedule_controller.py    # online entropy-rate estimation
└── docs/                           # per-experiment reports and the LM guide
```

---

## Citation

```bibtex
@article{cobit2026,
  title  = {CoBit: Language Modeling with Bitstream Diffusion},
  author = {Batzolis, Georgios and Girolami, Mark and Ambrogioni, Luca},
  journal = {arXiv preprint arXiv:2605.07013},
  year   = {2026}
}
```

The multimodal paper is under review; the citation will be added on acceptance.
