"""Build reproducible CSV/Markdown paper tables from frozen metric JSON files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "valid_fraction",
    "canonical_sequence_fraction",
    "canonical_residue_fraction",
    "unique_fraction",
    "exact_novelty",
    "esm2_650m_pppl",
    "esm2_650m_pppl_reference",
    "esmfold_plddt",
    "esmfold_fraction_plddt_ge_70",
    "prot_t5_frechet_distance",
    "prot_t5_mmd_rbf",
    "amino_acid_composition_jsd",
    "length_wasserstein",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("MODEL", "METRICS_JSON"),
        required=True,
    )
    parser.add_argument("--reported-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for model, path_text in args.result:
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        values = payload.get("results", payload)
        row = {"model": model, "source": str(path)}
        row.update({metric: values.get(metric) for metric in METRICS})
        rows.append(row)

    if args.reported_csv is not None:
        with args.reported_csv.open("r", encoding="utf-8", newline="") as handle:
            for reported in csv.DictReader(handle):
                rows.append(
                    {"model": reported.get("model"), "source": str(args.reported_csv)}
                    | {metric: reported.get(metric) or None for metric in METRICS}
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", *METRICS, "source"]
    csv_path = args.out_dir / "protein_generation_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.out_dir / "protein_generation_table.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames[:-1]) + " |\n")
        handle.write("| " + " | ".join(["---"] * (len(fieldnames) - 1)) + " |\n")
        for row in rows:
            cells = []
            for field in fieldnames[:-1]:
                value = row.get(field)
                cells.append("—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)))
            handle.write("| " + " | ".join(cells) + " |\n")
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
