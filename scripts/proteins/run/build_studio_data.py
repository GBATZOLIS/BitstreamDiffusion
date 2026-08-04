"""Build the ``eval_studio_data.js`` blob (``window.M0_EVAL``) consumed by
``m0_m1_work.html``.

This is the committed, reproducible generator for the studio/report data blob
(previously an ad-hoc script). It reads the four dashboard artefacts produced by
``evaluate_m0_dashboard.py`` (+ ``decode_m0_dashboard.py`` + ``score_m0_sequences.py``)
and emits a single ``window.M0_EVAL = {...};`` file:

    token_metrics           <- token_metrics.json            (verbatim)
    geometry_metrics        <- geometry_metrics.json         (verbatim)
    sequence_judge_metrics  <- sequence_judge_metrics.json   (verbatim)
    sequences               <- sequence_marginal.fasta + test_references.fasta
    ca                      <- decoded_coordinates.npz  (Cα traces, tenths-of-Å ints)
    exemplars               <- derived worst/median/best rankings (forward & marginal TM)

Usage (repo root):
    .venv/bin/python scripts/proteins/run/build_studio_data.py \
        --dir runs/proteins/m0_v1/protein_eval/dashboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif line.strip():
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def _ca_traces(coords: np.ndarray, lengths: np.ndarray, scale: int) -> list[list[int]]:
    """(N, maxL, 4, 3) backbone coords -> per-target flat [x,y,z,...] * scale, int."""
    out = []
    for i, length in enumerate(lengths):
        ca = coords[i, : int(length), 1, :]  # atom index 1 == Cα (N, CA, C, O)
        out.append([int(round(v * scale)) for v in ca.reshape(-1).tolist()])
    return out


def _rank(values: np.ndarray) -> dict:
    order = np.argsort(values, kind="stable")
    return {
        "worst": int(order[0]),
        "median": int(order[len(order) // 2]),
        "best": int(order[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard"))
    ap.add_argument("--scale", type=int, default=10, help="Cα coordinate quantisation (tenths of Å)")
    args = ap.parse_args()
    d = args.dir

    token = json.loads((d / "token_metrics.json").read_text())
    geometry = json.loads((d / "geometry_metrics.json").read_text())
    judge = json.loads((d / "sequence_judge_metrics.json").read_text())
    generated = _read_fasta(d / "sequence_marginal.fasta")
    reference = _read_fasta(d / "test_references.fasta")

    archive = np.load(d / "decoded_coordinates.npz", allow_pickle=False)
    lengths = archive["lengths"].astype(int)
    scale = int(args.scale)
    ca = {"lengths": lengths.tolist(), "scale": scale}
    for name in ("forward", "reference", "marginal", "shuffled", "unconditional"):
        ca[name] = _ca_traces(archive[name], lengths, scale)

    # Per-target TM to the decoded reference, ordered by the per_condition 'index'.
    n = len(lengths)
    forward_tm = [0.0] * n
    marginal_tm = [0.0] * n
    for rec in geometry["per_condition"]["forward"]:
        forward_tm[rec["index"]] = float(rec["tm_score_to_decoded_reference"])
    for rec in geometry["per_condition"]["marginal"]:
        marginal_tm[rec["index"]] = float(rec["tm_score_to_decoded_reference"])

    seqstr = _rank(np.asarray(forward_tm))    # Seq->3D folding quality ranking
    strstr = _rank(np.asarray(marginal_tm))   # de-novo backbone ranking
    exemplars = {
        "seqstr": seqstr,
        "strstr": strstr,
        "all": sorted(set(list(seqstr.values()) + list(strstr.values()))),
        "forward_tm": forward_tm,
        "marginal_tm": marginal_tm,
    }

    payload = {
        "token_metrics": token,
        "geometry_metrics": geometry,
        "sequence_judge_metrics": judge,
        "sequences": {"generated": generated, "reference": reference,
                      "lengths": [len(s) for s in reference]},
        "ca": ca,
        "exemplars": exemplars,
    }
    out = d / "eval_studio_data.js"
    out.write_text("window.M0_EVAL = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)  targets={n}  "
          f"seqstr={seqstr}  strstr={strstr}")


if __name__ == "__main__":
    main()
