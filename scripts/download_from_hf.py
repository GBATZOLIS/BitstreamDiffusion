#!/usr/bin/env python
"""Download the released CoBit checkpoints from the Hugging Face Hub into the
exact paths the training / evaluation configs expect.

    python -m pip install "huggingface_hub>=0.23"
    python scripts/download_from_hf.py --repo-id <user-or-org>/CoBit
    # only the medium model:
    python scripts/download_from_hf.py --repo-id <user-or-org>/CoBit --models cobit_m

The entropy-rate schedule tables already ship in this repo under
assets/entropy_tables/, so they are NOT downloaded by default (pass
--entropy-tables to restore them if you deleted them).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# repo_path_in_hub -> local destination (relative to repo root)
CKPTS = {
    "cobit_s_lm1b": (
        "checkpoints/cobit_s_lm1b_1M_ema.pt",
        "runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/checkpoints/step=001000000.pt",
    ),
    "cobit_s_owt": (
        "checkpoints/cobit_s_owt_750k_ema.pt",
        "runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M/checkpoints/step=000750000.pt",
    ),
    "cobit_m": (
        "checkpoints/cobit_m_owt_750k_ema.pt",
        "runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024/checkpoints/step=000750000.pt",
    ),
}
TOKENIZER = [
    ("tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.json",
     "datasets/openwebtext_gpt2_trainm100k/tokenizer_gpt2id_bpe16_65536_base1024.json"),
    ("tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.meta.json",
     "datasets/openwebtext_gpt2_trainm100k/tokenizer_gpt2id_bpe16_65536_base1024.meta.json"),
]
ENTROPY = [
    (f"entropy_tables/{ds}/{n}", f"assets/entropy_tables/{ds}/{n}")
    for ds in ("lm1b", "owt", "owt_medium")
    for n in ("entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt", "entropy_edges.pt")
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/CoBit")
    ap.add_argument("--models", nargs="+", default=["cobit_s_lm1b", "cobit_s_owt", "cobit_m"],
                    choices=list(CKPTS), help="which checkpoints to fetch")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the OWT code tokenizer")
    ap.add_argument("--entropy-tables", action="store_true",
                    help="also re-download entropy tables (they already ship in the repo)")
    ap.add_argument("--token", default=None, help="HF token (only needed for private repos)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed: python -m pip install 'huggingface_hub>=0.23'", file=sys.stderr)
        sys.exit(1)

    jobs = [CKPTS[m] for m in args.models]
    if not args.no_tokenizer:
        jobs += TOKENIZER
    if args.entropy_tables:
        jobs += ENTROPY

    for repo_path, local_rel in jobs:
        dest = REPO_ROOT / local_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"↓ {repo_path}  ->  {local_rel}")
        cached = hf_hub_download(repo_id=args.repo_id, filename=repo_path, token=args.token)
        # hf_hub_download returns a cache path; copy into the expected location.
        import shutil
        shutil.copyfile(cached, dest)
    print("\n✓ Download complete. Run an eval config to reproduce the paper numbers.")


if __name__ == "__main__":
    main()
