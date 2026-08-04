"""Build design-evidence metrics and standard protein renders for the M0 report.

The script consumes frozen four-seed evaluation bundles plus MMseqs2 outputs. It
does not invent self-consistency metrics: rankings are explicitly labelled as
plausibility queues until ProteinMPNN + ESMFold designability is available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from evaluation.proteins.io import atomic_json, read_fasta
from evaluation.proteins.structure_metrics import tm_score


SEEDS = (20260719, 20260720, 20260721, 20260722)
COLORS = ("#006d77", "#0a9396", "#55a630", "#ee9b00", "#bb3e03")


def _segments(points: np.ndarray) -> np.ndarray:
    return np.stack((points[:-1], points[1:]), axis=1)


def _equal_axes(ax, arrays: list[np.ndarray]) -> None:
    points = np.concatenate(arrays, axis=0)
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    radius = max(float((high - low).max()) / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    left, right = mobile.mean(0), reference.mean(0)
    covariance = (mobile - left).T @ (reference - right)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return (mobile - left) @ rotation + right


def _draw_trace(ax, ca: np.ndarray, *, color: str | None = None, alpha: float = 1.0, width: float = 3.2) -> None:
    if color:
        ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color=color, alpha=alpha, linewidth=width, solid_capstyle="round")
        return
    values = np.linspace(0, 1, max(1, len(ca) - 1))
    collection = Line3DCollection(_segments(ca), cmap="viridis", linewidth=width, alpha=alpha)
    collection.set_array(values)
    ax.add_collection3d(collection)


def _render_single(ca: np.ndarray, out: Path, title: str, subtitle: str) -> None:
    fig = plt.figure(figsize=(5.2, 4.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _draw_trace(ax, ca)
    _equal_axes(ax, [ca])
    fig.text(0.06, 0.94, title, ha="left", va="top", fontsize=12, weight="bold", color="#14211c")
    fig.text(0.06, 0.885, subtitle, ha="left", va="top", fontsize=8.5, color="#52645b")
    fig.tight_layout(pad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _render_overlap(generated: np.ndarray, reference: np.ndarray, out: Path, title: str, subtitle: str) -> None:
    generated = _kabsch(generated, reference)
    fig = plt.figure(figsize=(5.2, 4.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _draw_trace(ax, reference, color="#9aa39f", alpha=0.58, width=4.0)
    _draw_trace(ax, generated, color="#007f78", alpha=0.95, width=3.0)
    _equal_axes(ax, [generated, reference])
    fig.text(0.06, 0.94, title, ha="left", va="top", fontsize=12, weight="bold", color="#14211c")
    fig.text(0.06, 0.885, subtitle, ha="left", va="top", fontsize=8.5, color="#52645b")
    fig.text(0.06, 0.055, "gray: decoded LFQ reference   teal: generated", ha="left", fontsize=8, color="#52645b")
    fig.tight_layout(pad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _read_hits(path: Path) -> dict[str, list[dict]]:
    fields = ("query", "target", "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits")
    numeric_float = {"fident", "qcov", "tcov", "evalue", "bits"}
    numeric_int = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
    hits: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = dict(zip(fields, line.split("\t")))
        for key in numeric_float:
            row[key] = float(row[key])
        for key in numeric_int:
            row[key] = int(row[key])
        hits[row["query"]].append(row)
    return hits


def _mmseqs_summary(manifest: dict, hits: dict[str, list[dict]], clusters_path: Path) -> tuple[dict, dict]:
    representative = {}
    for line in clusters_path.read_text(encoding="utf-8").splitlines():
        rep, member = line.split("\t")[:2]
        representative[member] = rep
    query_rows = {}
    for query in manifest["queries"]:
        query_id = query["query"]
        candidates = hits.get(query_id, [])
        top = max(candidates, key=lambda item: item["bits"]) if candidates else None
        maximum_identity = max((item["fident"] for item in candidates), default=None)
        query_rows[query_id] = {
            **query,
            "significant_hits": len(candidates),
            "top_hit": top,
            "max_identity": maximum_identity,
            "below_30pct_identity": maximum_identity is None or maximum_identity < 0.30,
            "cluster_representative": representative.get(query_id, query_id),
        }
    summary = {}
    for kind in sorted({row["kind"] for row in manifest["queries"]}):
        rows = [row for row in query_rows.values() if row["kind"] == kind]
        reps = {row["cluster_representative"] for row in rows}
        detected = [row for row in rows if row["significant_hits"]]
        summary[kind] = {
            "queries": len(rows),
            "queries_with_significant_hit": len(detected),
            "fraction_without_significant_hit": 1 - len(detected) / len(rows),
            "fraction_below_30pct_identity": float(np.mean([row["below_30pct_identity"] for row in rows])),
            "max_observed_identity": max((row["max_identity"] for row in detected), default=None),
            "clusters_at_30pct_identity_80pct_coverage": len(reps),
            "cluster_fraction": len(reps) / len(rows),
        }
    summary["combined"] = {
        "queries": len(query_rows),
        "clusters_at_30pct_identity_80pct_coverage": len(set(representative.get(query, query) for query in query_rows)),
        "training_sequences": manifest["training_sequences"],
        "search_evalue": 1e-3,
        "search_min_bidirectional_coverage": 0.30,
    }
    return summary, query_rows


def _mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("runs/proteins/m0_v2/protein_eval"))
    parser.add_argument(
        "--bundle-name-template",
        default="m0_v2_b_best_steps250_seed{seed}",
        help="Evaluation bundle directory template below --eval-root.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.out_dir / "renders"

    manifest = json.loads((args.out_dir / "mmseqs_input_manifest.json").read_text())
    hits = _read_hits(args.out_dir / "mmseqs_hits.tsv")
    mmseqs, query_rows = _mmseqs_summary(manifest, hits, args.out_dir / "mmseqs_query_clusters_cluster.tsv")

    bundles = {}
    for seed in SEEDS:
        path = args.eval_root / args.bundle_name_template.format(seed=seed)
        coords = np.load(path / "decoded_coordinates.npz", allow_pickle=False)
        bundles[seed] = {
            "path": path,
            "lengths": coords["lengths"].astype(int),
            "reference": coords["reference"],
            "forward": coords["forward"],
            "marginal": coords["marginal"],
            "geometry": json.loads((path / "geometry_metrics.json").read_text()),
            "judge": json.loads((path / "sequence_judge_metrics.json").read_text()),
            "sequences": read_fasta(path / "sequence_marginal.fasta"),
            "inverse": json.loads((path / "inverse_predictions.json").read_text()),
        }

    structure_rows = []
    for target in range(32):
        length = int(bundles[SEEDS[0]]["lengths"][target])
        candidates = [bundles[seed]["marginal"][target, :length, 1] for seed in SEEDS]
        pairwise = np.full((len(SEEDS), len(SEEDS)), np.nan)
        for left in range(len(SEEDS)):
            for right in range(left + 1, len(SEEDS)):
                value = float(tm_score(candidates[left], candidates[right]))
                pairwise[left, right] = pairwise[right, left] = value
        for index, seed in enumerate(SEEDS):
            nearest = float(np.nanmax(pairwise[index]))
            marginal_metrics = bundles[seed]["geometry"]["per_condition"]["marginal"][target]
            structure_rows.append({
                "seed": seed,
                "target": target,
                "length": length,
                "nearest_generated_tm_same_length": nearest,
                "tm_to_length_matched_decoded_test_reference": marginal_metrics["tm_score_to_decoded_reference"],
            })
    structural_diversity = {
        "candidates": len(structure_rows),
        "mean_nearest_generated_tm": _mean(row["nearest_generated_tm_same_length"] for row in structure_rows),
        "fraction_nearest_tm_below_0_5": float(np.mean([row["nearest_generated_tm_same_length"] < 0.5 for row in structure_rows])),
        "mean_tm_to_length_matched_decoded_test_reference": _mean(row["tm_to_length_matched_decoded_test_reference"] for row in structure_rows),
        "fraction_reference_tm_below_0_5": float(np.mean([row["tm_to_length_matched_decoded_test_reference"] < 0.5 for row in structure_rows])),
        "method": "pairwise Kabsch TM approximation among four same-length seed samples",
        "limitation": "This is decoded-token-space diversity and held-out reference proximity, not Foldseek novelty against native training structures.",
    }

    representative = bundles[SEEDS[0]]
    ranking = []
    for target in range(32):
        length = int(representative["lengths"][target])
        geometry = representative["geometry"]["per_condition"]["marginal"][target]
        pppl = float(representative["judge"]["per_sequence"]["generated_pppl"][target])
        query = query_rows[f"marginal_s{SEEDS[0]}_t{target:02d}"]
        clashes_per_100 = 100 * geometry["ca_clashes"] / length
        geometry_score = math.exp(-clashes_per_100 / 4) * math.exp(-geometry["chain_breaks"])
        plausibility_score = max(0.0, min(1.0, (24.0 - pppl) / 10.0))
        novelty_score = 1.0 if query["max_identity"] is None else 1.0 - query["max_identity"]
        score = 0.50 * plausibility_score + 0.35 * geometry_score + 0.15 * novelty_score
        ranking.append({
            "target": target,
            "seed": SEEDS[0],
            "length": length,
            "evidence_score": score,
            "esm2_8m_pppl": pppl,
            "ca_clashes": geometry["ca_clashes"],
            "ca_clashes_per_100_residues": clashes_per_100,
            "chain_breaks": geometry["chain_breaks"],
            "mmseqs_max_training_identity": query["max_identity"],
            "mmseqs_significant_hits": query["significant_hits"],
            "designability": None,
            "plddt": None,
        })
    ranking.sort(key=lambda row: row["evidence_score"], reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    gallery = []
    for row in ranking[:6]:
        target, length = row["target"], row["length"]
        path = render_dir / f"rank_{row['rank']:02d}_target_{target:02d}.png"
        _render_single(
            representative["marginal"][target, :length, 1],
            path,
            f"Plausibility rank {row['rank']} · target {target:02d}",
            f"L={length}  ESM-2 8M PPPL={row['esm2_8m_pppl']:.2f}  clashes={row['ca_clashes']}  pLDDT=n/a",
        )
        gallery.append({**row, "image": str(path)})

    forward = representative["geometry"]["per_condition"]["forward"]
    ordered = sorted(range(32), key=lambda index: forward[index]["tm_score_to_decoded_reference"])
    chosen = sorted(set(ordered[index] for index in (0, 6, 12, 19, 25, 31)), key=lambda index: forward[index]["tm_score_to_decoded_reference"], reverse=True)
    overlaps = []
    for target in chosen:
        length = int(representative["lengths"][target])
        metric = forward[target]
        path = render_dir / f"overlap_target_{target:02d}.png"
        _render_overlap(
            representative["forward"][target, :length, 1],
            representative["reference"][target, :length, 1],
            path,
            f"Sequence-conditioned overlap · target {target:02d}",
            f"L={length}  TM={metric['tm_score_to_decoded_reference']:.3f}  lDDT={metric['lddt_to_decoded_reference']:.3f}  pLDDT=n/a",
        )
        overlaps.append({
            "target": target,
            "length": length,
            "tm": metric["tm_score_to_decoded_reference"],
            "lddt": metric["lddt_to_decoded_reference"],
            "rmsd": metric["rmsd_to_decoded_reference"],
            "plddt": None,
            "image": str(path),
        })

    inverse_profiles = []
    for seed in SEEDS:
        inverse_profiles.append(bundles[seed]["inverse"]["conditioned_normalized_position_profile"])
    per_residue_profile = []
    for bin_index in range(20):
        values = [profile[bin_index]["recovery"] for profile in inverse_profiles]
        per_residue_profile.append({
            "bin": bin_index,
            "normalized_midpoint": (bin_index + 0.5) / 20,
            "mean_recovery": float(np.mean(values)),
            "min_seed_recovery": float(np.min(values)),
            "max_seed_recovery": float(np.max(values)),
        })

    profile_path = render_dir / "per_residue_recovery.svg"
    profile_x = [row["normalized_midpoint"] for row in per_residue_profile]
    profile_mean = [100 * row["mean_recovery"] for row in per_residue_profile]
    profile_low = [100 * row["min_seed_recovery"] for row in per_residue_profile]
    profile_high = [100 * row["max_seed_recovery"] for row in per_residue_profile]
    fig, ax = plt.subplots(figsize=(8.2, 3.1), facecolor="white")
    ax.fill_between(profile_x, profile_low, profile_high, color="#99d5cc", alpha=0.55, label="four-seed range")
    ax.plot(profile_x, profile_mean, color="#006d77", linewidth=2.6, marker="o", markersize=3.5, label="mean recovery")
    ax.set(xlim=(0, 1), ylim=(0, max(35, max(profile_high) + 3)), xlabel="Normalised residue position (N-terminus → C-terminus)", ylabel="Exact recovery (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dfe8e3", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(profile_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "mmseqs2": mmseqs,
        "mmseqs2_per_query": query_rows,
        "structural_diversity_proxy": structural_diversity,
        "per_residue_inverse_folding": {
            "status": "measured",
            "seeds": list(SEEDS),
            "profile": per_residue_profile,
            "figure": str(profile_path),
            "source": "inverse_predictions.json in each seed bundle",
        },
        "plausibility_ranking": {
            "label": "screening queue, not designability ranking",
            "formula": "0.50 compact-ESM plausibility + 0.35 decoded-backbone geometry + 0.15 MMseqs novelty",
            "limitations": [
                "No ProteinMPNN-to-ESMFold self-consistency, scTM, scRMSD, or refold pLDDT is available.",
                "Ranking weights were not calibrated on an external development set and are for triage only.",
            ],
            "rows": ranking,
        },
        "gallery": gallery,
        "overlaps": overlaps,
        "outstanding": {
            "designability": "blocked: ProteinMPNN and ESMFold weights are not cached",
            "structure_novelty": "blocked: no native training structures and no Foldseek executable",
            "matched_protein_model_baselines": "not run: no local matched DPLM-2/DiMA/ProteinMPNN generation checkpoints and protocols",
        },
    }
    atomic_json(args.out_dir / "design_grade_metrics.json", payload)
    (args.out_dir / "design_grade_data.js").write_text(
        "window.M0_DESIGN_GRADE = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"mmseqs2": mmseqs, "structural_diversity": structural_diversity, "top_ranked": gallery[0]}, indent=2))


if __name__ == "__main__":
    main()
