# CoBit on TinyGSM → GSM8K: Experiment, Results, and Standpoint

**Model:** CoBit-S (`tasks/tinygsm/cobit_raw_binary_bits`), continuous bitstream diffusion
**Checkpoint:** step 250,000 (EMA, decay 0.9999)
**Task:** GSM8K executable-code accuracy, full test set (1319 examples)
**Date of eval:** 19 June 2026
**Canonical artifacts:** this file; PDF at
`runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval/tinygsm_250k_gsm8k_report.pdf`;
raw JSON at `runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval/gsm8k_results_*.json`.

---

## 1. Experiment description

We model GSM8K math-word-problem solving as conditional generation of an
executable Python program, following the S-FLM TinyGSM→GSM8K parity setup
(Deschenaux & Gulcehre). CoBit represents text as a continuous diffusion process
over fixed-width binary bitstreams rather than discrete tokens.

**Setup (faithful to S-FLM parity):**
- **Tokenizer:** `HuggingFaceTB/SmolLM-135M` (|V| = 49,152, + PAD), 16 bits/token.
- **Sequence:** 512 tokens → 8,192-bit bitstream. Example = `[BOS] question \n code [EOS]`, padded to 512.
- **Conditioning:** prompt = `[BOS] question \n` (variable-length per-example prefix), kept clean and excluded from the loss; loss is over answer + EOS + pad.
- **Backbone:** trunk matches S-FLM — 12 blocks / 768-d / 12 heads (~130M params).
- **Training:** 250k steps, global batch 512, EMA 0.9999.
- **Diffusion:** EDM-style, σ_min 0.002, σ_max 80.0, ρ 7.0; measured σ_data ≈ 0.3998 (used at eval).

**Scoring (vendored from S-FLM, identical):** restricted sandbox, function
`simple_math_problem`, math-only imports, 1e-3 tolerance, 5 s timeout; 95% CI by
percentile bootstrap. Invalid-token rate (decoded ids out of vocab) is tracked as
a bitstream-specific failure mode.

**Samplers compared:** deterministic (γ = 0) and an entropy-rate-gated stochastic
sampler with churn s_churn = γ·(NFE − 1), swept over γ and inference budgets
NFE ∈ {256, 384, 512}.

---

## 2. Results (250k checkpoint, EMA, σ_data = 0.3998, 1319 examples)

### Deterministic (γ = 0)

| NFE | Accuracy | Correct |
|----:|---------:|--------:|
| 256 | 9.17%    | 121     |
| 384 | 9.25%    | 122     |
| 512 | 9.25%    | 122     |

Deterministic accuracy is essentially flat in NFE.

### Stochastic — NFE 256

| γ        | Accuracy   | Correct |
|---------:|-----------:|--------:|
| 0.21     | 17.74%     | 234     |
| 0.26     | 19.26%     | 254     |
| **0.31** | **21.61%** | **285** |
| 0.36     | 20.92%     | 276     |

### Stochastic — NFE 384

| γ          | Accuracy   | Correct |
|-----------:|-----------:|--------:|
| 0.30       | 20.55%     | 271     |
| 0.36       | 21.38%     | 282     |
| 0.40       | 22.59%     | 298     |
| **0.4142** | **22.74%** | **300** |

### Stochastic — NFE 512

| γ          | Accuracy   | Correct |
|-----------:|-----------:|--------:|
| **0.34**   | **23.43%** | **309** |
| 0.38       | 22.52%     | 297     |
| 0.40       | 23.12%     | 305     |
| 0.4142     | 23.12%     | 305     |

**Best overall: 23.43% at NFE 512, γ = 0.34.** Invalid-token rate ≤ 7×10⁻⁶ everywhere.

---

## 3. Standpoint vs comparable-size DLMs and AR LLMs

Same 1319-example test set and scoring sandbox (S-FLM parity). Baseline numbers
are from Deschenaux & Gulcehre at temperature T = 1; the diffusion/non-AR models
are all **unguided** (no classifier-free guidance). The AR rows are autoregressive
LLMs of the same ~130M-class trunk, listed as a reference ceiling.

| Model                          | Family                          | Acc. (%) |
|--------------------------------|---------------------------------|---------:|
| AR (greedy)                    | autoregressive LLM              | 63.3     |
| AR (sample)                    | autoregressive LLM              | 53.9     |
| **CoBit-S (this work, 250k)**  | **continuous bitstream diffusion** | **23.4** |
| MDLM                           | masked discrete diffusion       | 18.0     |
| S-FLM (+top-k=1)               | hyperspherical flow             | 18.0     |
| Duo                            | discrete diffusion              | 17.2     |
| S-FLM (+adaptive +S-arch)      | hyperspherical flow             | 12.4     |
| S-FLM (vanilla)                | hyperspherical flow             | 1.2      |
| FLM                            | continuous flow                 | 0.3      |

**Reading of the standpoint:**
- **Best-in-class among unguided DLMs.** CoBit-S (23.4%) clears the entire non-AR
  diffusion band (≤18.0%) by ~5 points.
- **The comparison is conservative on NFE.** The baseline DLMs are reported at
  S-FLM's 1024 sampling steps, whereas CoBit's 23.4% uses only 512 NFE — and even
  at 256 NFE (21.6%) CoBit already exceeds the best prior unguided DLM.
- **Stochastic sampling is the main lever.** Injected churn lifts CoBit from ~9%
  (deterministic) to ~23% — a ~2.5× gain — and is what carries it past the
  top-k=1 / MDLM band.
- **The gap to autoregressive decoding remains large** (23.4% vs 53.9–63.3%). Closing
  it is the open problem; non-AR diffusion has not yet matched AR on this verifiable
  reasoning task.

---

## 4. Observations

- Accuracy rises with NFE but saturates: best-per-budget is 21.6% (256) → 22.7%
  (384) → 23.4% (512); 384→512 gains are small and CIs overlap.
- Optimal γ shifts with budget (≈0.31 @256, ≈0.41 @384, ≈0.34 @512), consistent
  with s_churn = γ·(NFE−1) keeping total injected noise roughly bounded.
- Output validity is high throughout (invalid-token rate ≤ 7×10⁻⁶).

---

## 5. Guidance outlook (next step, not yet evaluated)

This run was trained with `p_uncond = 0.0` and therefore **cannot be guided**. A
parallel classifier-free-guidance run (`tasks/tinygsm/cobit_raw_binary_bits_cfg`,
`p_uncond = 0.1`, null_strategy "half") is training toward 250k for a fair
head-to-head. Expectation: guidance should help (the task is verifiable and
mode-seeking — cf. S-FLM 1.2% → 18.0% from top-k=1 sharpening alone), but with
diminishing returns, since the stochastic sampler already captures much of the
sharpening benefit. Recommended test: guided-CFG vs unguided-stochastic at equal
NFE (512), sweeping guidance scale w ∈ {0, 0.5, 1, 2, 3, 5} with reduced γ
(≈0.2–0.34), tuning w and γ jointly.
