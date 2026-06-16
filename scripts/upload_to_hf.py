#!/usr/bin/env python
"""Upload the released CoBit checkpoints + artefacts to a Hugging Face Hub repo.

This packages everything a downloader needs to reproduce the paper tables:
the three EMA checkpoints (CoBit-S LM1B, CoBit-S OWT, CoBit-M OWT), the OWT
16-bit code tokenizer, and the dataset-specific entropy-rate schedule tables.

Prerequisites
-------------
    python -m pip install "huggingface_hub>=0.23"
    # A *write* access token from https://huggingface.co/settings/tokens
    export HF_TOKEN=hf_xxx              # or pass --token / use `hf auth login`

Layout produced on the Hub (repo_type=model)
---------------------------------------------
    checkpoints/cobit_s_lm1b_1M_ema.pt
    checkpoints/cobit_s_owt_750k_ema.pt
    checkpoints/cobit_m_owt_750k_ema.pt
    tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.json
    tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.meta.json
    entropy_tables/lm1b/{pdf,cdf,sigmas,edges}.pt
    entropy_tables/owt/{pdf,cdf,sigmas,edges}.pt
    entropy_tables/owt_medium/{pdf,cdf,sigmas,edges}.pt

Usage
-----
    python scripts/upload_to_hf.py --repo-id <user-or-org>/CoBit
    # dry-run first to see exactly what would be pushed:
    python scripts/upload_to_hf.py --repo-id <user-or-org>/CoBit --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (REPO_ROOT / p)


def build_manifest(release_dir: Path) -> list[tuple[Path, str]]:
    """Return [(local_path, path_in_repo)] for everything we ship."""
    rd = _resolve(release_dir)
    assets = _resolve(Path("assets/entropy_tables"))
    tok_dir = _resolve(Path("datasets/openwebtext_gpt2_trainm100k"))

    manifest: list[tuple[Path, str]] = [
        (rd / "cobit_s_lm1b_step001000000_ema.pt", "checkpoints/cobit_s_lm1b_1M_ema.pt"),
        (rd / "cobit_s_owt_step000750000_ema.pt",  "checkpoints/cobit_s_owt_750k_ema.pt"),
        (rd / "cobit_m_owt_step000750000_ema.pt",  "checkpoints/cobit_m_owt_750k_ema.pt"),
        (tok_dir / "tokenizer_gpt2id_bpe16_65536_base1024.json",
         "tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.json"),
        (tok_dir / "tokenizer_gpt2id_bpe16_65536_base1024.meta.json",
         "tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.meta.json"),
    ]
    for ds in ("lm1b", "owt", "owt_medium"):
        for name in ("entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt", "entropy_edges.pt"):
            manifest.append((assets / ds / name, f"entropy_tables/{ds}/{name}"))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/CoBit")
    ap.add_argument("--release-dir", default="release", type=Path, help="dir holding the *_ema.pt checkpoints")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--dry-run", action="store_true", help="print the manifest and exit without uploading")
    args = ap.parse_args()

    manifest = build_manifest(args.release_dir)

    missing = [str(src) for src, _ in manifest if not src.exists()]
    print("=== upload manifest ===")
    for src, dst in manifest:
        flag = "  MISSING" if not src.exists() else ""
        size = f"{src.stat().st_size/1e9:.2f} GB" if src.exists() else "—"
        print(f"  {dst:55s} <- {src}  ({size}){flag}")
    if missing:
        print(f"\n{len(missing)} file(s) missing — produce the *_ema.pt checkpoints with "
              f"scripts/owt/make_release_checkpoint.py first.", file=sys.stderr)
        if not args.dry_run:
            sys.exit(1)

    if args.dry_run:
        print("\n(dry-run) nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed: python -m pip install 'huggingface_hub>=0.23'", file=sys.stderr)
        sys.exit(1)

    # token=None falls back to cached credentials from `hf auth login` (preferred:
    # no token in the shell history). If neither is present, create_repo below
    # raises a clear auth error.
    if not args.token:
        print("No --token/HF_TOKEN given; using cached `hf auth login` credentials.")
    api = HfApi(token=args.token or None)
    print(f"\nCreating/ensuring repo {args.repo_id} (private={args.private}) ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for src, dst in manifest:
        print(f"uploading {dst} ...")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=args.repo_id, repo_type="model")
    print(f"\n✓ Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
