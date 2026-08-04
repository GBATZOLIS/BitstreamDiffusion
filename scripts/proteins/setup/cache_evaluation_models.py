#!/usr/bin/env python3
"""Explicitly prefetch evaluator weights into the Hugging Face cache."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

MODELS = {
    "esm2": "facebook/esm2_t33_650M_UR50D",
    "prot_t5_tokenizer": "Rostlab/prot_t5_xl_uniref50",
    "prot_t5": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "esmfold": "facebook/esmfold_v1",
    "dplm": "airkingbd/dplm_150m",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        choices=["all", *MODELS],
        help="Weights to cache; this may require tens of GB",
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    selected = list(MODELS) if "all" in args.models else args.models
    for name in selected:
        print(f"Caching {name}: {MODELS[name]}")
        snapshot_download(MODELS[name], cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
