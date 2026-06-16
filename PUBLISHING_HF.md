# Publishing the CoBit checkpoints to the Hugging Face Hub

This is the maintainer checklist for hosting the released checkpoints on Hugging
Face (more credible + faster downloads than Google Drive). End users never run
these steps — they only run `scripts/download_from_hf.py` (see the README).

The release artefacts are produced **once** from the full training checkpoints
and live in `release/`:

| File | What it is | Size |
|---|---|---|
| `release/cobit_s_lm1b_step001000000_ema.pt` | CoBit-S LM1B, EMA | 0.53 GB |
| `release/cobit_s_owt_step000750000_ema.pt`  | CoBit-S OWT, EMA  | 0.54 GB |
| `release/cobit_m_owt_step000750000_ema.pt`  | CoBit-M OWT, EMA  | 1.85 GB |

These are EMA-baked (the EMA shadow is copied into the weights and the optimizer
state dropped) by `scripts/owt/make_release_checkpoint.py`. Evaluating them with
the default `apply_ema=True` is numerically identical to evaluating the original
full training checkpoint — that is exactly how the paper tables were produced.

## What you need to do (one time)

### 1. Make a Hugging Face account + a model repo

1. Sign up / log in at <https://huggingface.co>.
2. (Optional but recommended) create an **organization** for the project so the
   repo lives at `org/CoBit` rather than under your personal handle.
3. Create a **model** repository: <https://huggingface.co/new>
   - Owner: your user or the org.
   - Name: e.g. `CoBit`.
   - Visibility: **Public**.
   - You do **not** need to add any files in the web UI — the upload script does that.

   The repo id is `«owner»/CoBit` (e.g. `gbatzolis/CoBit`). Note it down.

### 2. Get a write token

1. <https://huggingface.co/settings/tokens> → **Create new token** → type **Write**.
2. Copy the `hf_…` string. Treat it like a password (don't commit it).

### 3. Install the client and log in

```bash
python -m pip install "huggingface_hub>=0.23"
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx        # the write token from step 2
# (equivalently: `hf auth login` and paste the token interactively)
```

### 4. (If not already done) build the release checkpoints

If `release/` is empty — e.g. on a fresh clone — regenerate the EMA checkpoints
from the full training checkpoints:

```bash
python scripts/owt/make_release_checkpoint.py \
  --in  <path>/lm1b/.../checkpoints/step=001000000.pt \
  --out release/cobit_s_lm1b_step001000000_ema.pt
python scripts/owt/make_release_checkpoint.py \
  --in  <path>/owt/.../continuous_rate_raw_binary_bits_1M/checkpoints/step=000750000.pt \
  --out release/cobit_s_owt_step000750000_ema.pt
python scripts/owt/make_release_checkpoint.py \
  --in  <path>/owt/.../continuous_rate_raw_binary_bits_medium_24x1024/checkpoints/step=000750000.pt \
  --out release/cobit_m_owt_step000750000_ema.pt
```

### 5. Dry-run, then upload

```bash
# See exactly what will be pushed (no network writes):
python scripts/upload_to_hf.py --repo-id «owner»/CoBit --dry-run

# Upload checkpoints + tokenizer + entropy tables:
python scripts/upload_to_hf.py --repo-id «owner»/CoBit
```

The script creates the repo if needed and pushes this layout:

```
checkpoints/cobit_s_lm1b_1M_ema.pt
checkpoints/cobit_s_owt_750k_ema.pt
checkpoints/cobit_m_owt_750k_ema.pt
tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.json
tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.meta.json
entropy_tables/{lm1b,owt,owt_medium}/entropy_{pdf,cdf,sigmas,edges}.pt
```

### 6. Point the README at the repo

Replace the literal `COBIT_HF_REPO` placeholder in `README.md` with your repo id
(`«owner»/CoBit`), commit, and push the GitHub repo. That's it — downloaders now
run `python scripts/download_from_hf.py --repo-id «owner»/CoBit`.

### 7. (Recommended) add a model card

On the Hub repo page, paste a short model card (title, the Table-1/Table-2
numbers, a link back to this GitHub repo and the arXiv paper). The Hub renders
the repo's `README.md`; you can copy the headline-results tables from this repo.

## Notes

- **Large files**: `huggingface_hub` uploads `.pt` via the Hub's LFS-backed
  storage automatically; no `git lfs` setup needed.
- **Re-uploads** are idempotent — re-running the script overwrites the files.
- **Keep Google Drive?** Not necessary once HF is live. If you want a mirror,
  leave the old Drive links in a footnote; otherwise delete them.
