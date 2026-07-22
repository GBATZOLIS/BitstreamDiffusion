---
license: mit
tags:
  - text-generation
  - diffusion
  - language-model
  - bitstream-diffusion
library_name: pytorch
---

# CoBit — Continuous Bitstream Diffusion language models

Released checkpoints for **"CoBit: Language Modeling with Bitstream Diffusion"**
(Batzolis, Girolami, Ambrogioni, 2026). Code, configs and full reproduction instructions:
**https://github.com/GBATZOLIS/BitstreamDiffusion** · paper: [arXiv:2605.07013](https://arxiv.org/abs/2605.07013)

Text is modelled as a continuous diffusion process over fixed-width binary
bitstreams, with a matched-filter residual parameterization and an
entropy-rate-gated stochastic sampler. All checkpoints are **EMA weights**;
evaluate them with the repo's eval configs (default `apply_ema=True`).

## Checkpoints

| File | Model | Dataset | Steps | GenPPL (best reported) |
|---|---|---|---|---|
| `checkpoints/cobit_s_lm1b_1M_ema.pt`  | CoBit-S (130M) | LM1B | 1.0M  | 59.76 @ H 4.31 (256 NFE) |
| `checkpoints/cobit_s_owt_750k_ema.pt` | CoBit-S (130M) | OpenWebText | 750K | 27.06 @ H 5.26 (256 NFE) |
| `checkpoints/cobit_m_owt_750k_ema.pt` | **CoBit-M (462M)** | OpenWebText | 750K | **9.87 @ H 5.25 (512 NFE)** |

### CoBit-M (462M) — OpenWebText, Table 2

| NFE | γ | GenPPL ↓ | Entropy |
|---|---|---|---|
| 256 | 0.21 | 19.48 | 5.40 |
| 256 | 0.13 | 18.47 | 5.378 |
| 384 | 0.24 | 13.06 | 5.33 |
| 512 | 0.26 | 9.87  | 5.25 |

Real OpenWebText reference: GenPPL 15.07, entropy 5.44. GenPPL is GPT-2-Large
perplexity; entropy is GPT-2-token unigram entropy.

## Usage

```bash
git clone https://github.com/GBATZOLIS/BitstreamDiffusion && cd BitstreamDiffusion
python -m pip install -r requirements.txt "huggingface_hub>=0.23"

# Fetch checkpoints into the paths the configs expect:
python scripts/download_from_hf.py --repo-id gbatzolis/CoBit

# Reproduce the CoBit-M Table-2 numbers:
bash scripts/owt/eval_cobit_m.sh
```

Also bundled: the OWT 16-bit code tokenizer (`tokenizer/`) and the
dataset-specific entropy-rate schedule tables (`entropy_tables/`).

## Citation

```bibtex
@misc{batzolis2026bitstream,
  title         = {CoBit: Language Modeling with Bitstream Diffusion},
  author        = {Batzolis, Georgios and Girolami, Mark and Ambrogioni, Luca},
  year          = {2026},
  eprint        = {2605.07013},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
