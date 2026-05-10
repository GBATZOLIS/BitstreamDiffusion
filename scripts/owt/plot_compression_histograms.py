#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tqdm import tqdm

def get_lengths_for_tokenizer(tokenizer_path: Path, strings_path: Path, limit: int, batch_size: int = 2048) -> np.ndarray:
    """Reads PUA strings, encodes them in batches, and returns an array of lengths."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    lengths = []
    batch = []
    seen = 0

    def flush(b):
        if not b:
            return
        encs = tok.encode_batch(b)
        lengths.extend(len(enc.ids) for enc in encs)

    with open(strings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Encoding with {tokenizer_path.stem}", unit="seq"):
            if limit is not None and seen >= limit:
                break
            batch.append(line.rstrip("\n"))
            seen += 1

            if len(batch) >= batch_size:
                flush(batch)
                batch = []
                
    flush(batch)
    return np.array(lengths)

def main():
    parser = argparse.ArgumentParser(description="Plot compression histograms for BPE tokenizers.")
    parser.add_argument("--root_dir", type=str, default="runs/LargeVocabularies", help="Root directory containing the run folders.")
    parser.add_argument("--run_suffix", type=str, default="train200k_stats500k", help="Suffix of the output folders.")
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 17, 18, 19, 20, 21], help="List of bit sizes to process.")
    parser.add_argument("--pua_strings_path", type=str, required=True, help="Path to the shared PUA strings file to use for evaluation.")
    parser.add_argument("--limit_sequences", type=int, default=50000, help="Number of sequences to encode for the histogram.")
    parser.add_argument("--base_seq_len", type=int, default=1024, help="Original GPT-2 sequence length.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    pua_strings_path = Path(args.pua_strings_path)

    if not pua_strings_path.exists():
        raise FileNotFoundError(f"PUA strings file not found: {pua_strings_path}")

    all_lengths = {}
    combined_out_dir = root_dir / "compression_histograms"
    combined_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process each bit size and generate individual histograms
    for bits in args.bits:
        vocab_size = 2 ** bits
        run_dir = root_dir / f"bpe{bits}_{args.run_suffix}"
        
        # Fallback for warm-started tokenizers (e.g., bpe21_from_bpe20_...)
        if not run_dir.exists():
            alt_dirs = list(root_dir.glob(f"bpe{bits}_from_*_{args.run_suffix}"))
            if alt_dirs:
                run_dir = alt_dirs[0]

        tok_path = run_dir / "tokenizers" / f"tokenizer_gpt2id_bpe{bits}_{vocab_size}_base{args.base_seq_len}.json"

        if not tok_path.exists():
            print(f"[Warning] Tokenizer not found for {bits}-bit: {tok_path}. Skipping.")
            continue

        print(f"\nProcessing {bits}-bit tokenizer...")
        lengths = get_lengths_for_tokenizer(tok_path, pua_strings_path, args.limit_sequences)
        all_lengths[bits] = lengths

        # Plot individual histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.75, color='steelblue', edgecolor='black')
        plt.title(f"Encoded Sequence Length Distribution ({bits}-bit BPE)\nOriginal Length: {args.base_seq_len} tokens", fontsize=14)
        plt.xlabel("Encoded Length (tokens)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        mean_len = lengths.mean()
        compression_ratio = args.base_seq_len / mean_len
        plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {mean_len:.1f} (Ratio: {compression_ratio:.2f}x)')
        
        plt.grid(axis='y', alpha=0.5)
        plt.legend(fontsize=12)
        
        # Save to the run's stats directory
        stats_dir = run_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        out_plot = stats_dir / f"histogram_bpe{bits}.png"
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved individual histogram to: {out_plot}")

    # 2. Plot combined overlapping histogram for comparison
    if all_lengths:
        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_lengths)))
        
        for (bits, lengths), color in zip(all_lengths.items(), colors):
            # Calculate integer rounded mean and 99.9th percentile
            mean_val = int(round(lengths.mean()))
            p999_val = int(round(np.percentile(lengths, 99.9)))
            
            # Compact label formatting
            label_str = f'{bits}-bit (μ: {mean_val}, p999: {p999_val})'
            
            plt.hist(lengths, bins=50, alpha=0.5, label=label_str, color=color, density=True)

        plt.title(f"Compression Comparison: Encoded Length Distributions", fontsize=16)
        plt.xlabel("Encoded Length (tokens)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.grid(axis='y', alpha=0.5)
        plt.legend(fontsize=12)
        
        combined_plot = combined_out_dir / "combined_compression_histograms.png"
        plt.savefig(combined_plot, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nSaved combined comparison histogram to: {combined_plot}")

if __name__ == "__main__":
    main()