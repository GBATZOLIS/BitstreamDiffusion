"""Reproducible four-direction M0 dashboard evaluation.

This is an exploratory, control-aware evaluation on a deterministic length-
stratified subset of the local CAMEO-labelled test cache.  It measures sequence
and LFQ-token behaviour without pretending the cache contains native coordinates.
The companion ``decode_m0_dashboard.py`` decodes the saved LFQ tokens and adds
geometry metrics/overlay coordinates with the frozen DPLM tokenizer.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.protein_multimodal import CANONICAL_AA, ProteinMultimodalDataset
from data.protein_structure_codec import DEFAULT_STRUCT_CODEC
from evaluation.proteins.generate_multimodal import generate
from evaluation.proteins.io import atomic_json, load_binary_protein_model, write_fasta
from evaluation.proteins.metrics import basic_metrics


def _sequence(ids: np.ndarray) -> str:
    return "".join(CANONICAL_AA[int(i)] for i in ids)


def _select(lengths: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(lengths, kind="stable")
    if count >= len(order):
        return order
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return order[positions]


def _identity(sequence: str, reference: str) -> float:
    return float(np.mean(np.frombuffer(sequence.encode(), dtype="S1") == np.frombuffer(reference.encode(), dtype="S1")))


def _struct_scores(pred: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    exact = float(np.mean(pred == reference))
    left = DEFAULT_STRUCT_CODEC.index_to_bits_np(pred)
    right = DEFAULT_STRUCT_CODEC.index_to_bits_np(reference)
    return exact, float(np.mean(left == right))


def _raw_invalid_fraction(bits: torch.Tensor, length: int) -> float:
    patches = bits.reshape(bits.shape[0], length, 18)[:, :, :5].cpu().numpy()
    values = np.sum(patches.astype(np.int64) * (2 ** np.arange(4, -1, -1)), axis=-1)
    return float(np.mean(values >= len(CANONICAL_AA)))


def _jsd_counts(left: np.ndarray, right: np.ndarray, bins: int) -> float:
    p = np.bincount(left.reshape(-1), minlength=bins).astype(np.float64)
    q = np.bincount(right.reshape(-1), minlength=bins).astype(np.float64)
    p /= max(1.0, p.sum()); q /= max(1.0, q.sum()); m = 0.5 * (p + q)
    value = 0.0
    for distribution in (p, q):
        mask = distribution > 0
        value += 0.5 * float(np.sum(distribution[mask] * np.log2(distribution[mask] / m[mask])))
    return value


def _summary(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
        "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
    }


def _bootstrap(values, seed: int, draws: int = 4000) -> dict:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def _condition_summary(records: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: _summary([record[key] for record in records]) for key in keys}


def _pad(rows: list[np.ndarray], maximum: int) -> np.ndarray:
    out = np.zeros((len(rows), maximum), dtype=np.uint16)
    for i, row in enumerate(rows): out[i, : len(row)] = row
    return out


def _best_struct(generated: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    scored = []
    for row in generated:
        exact, bit = _struct_scores(row, reference)
        scored.append((exact, bit))
    best = max(range(len(scored)), key=lambda i: (scored[i][1], scored[i][0]))
    return generated[best], {
        "exact_mean": float(np.mean([x[0] for x in scored])),
        "exact_best": float(max(x[0] for x in scored)),
        "bit_mean": float(np.mean([x[1] for x in scored])),
        "bit_best": float(max(x[1] for x in scored)),
    }


def _plot_training(out_dir: Path, experiment: str) -> None:
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    log_dir = Path("runs") / experiment / "training_logs"
    events = sorted(log_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found under {log_dir}")
    event = events[-1]
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0}); acc.Reload()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for tag, label, color in (("loss/iter_train_smooth", "smoothed train", "#d06a2d"),):
        rows = acc.Scalars(tag); axes[0].plot([r.step / 1000 for r in rows], [r.value for r in rows], label=label, color=color)
    val = acc.Scalars("loss/epoch_val")
    axes[0].scatter([r.step / 1000 for r in val], [r.value for r in val], s=12, alpha=.6, label="stochastic validation", color="#1769aa")
    axes[0].set(ylim=(0, 10), xlabel="optimizer steps (k)", ylabel="weighted denoising loss"); axes[0].legend(frameon=False)
    for tag, label, color in (("multimodal/seq_bit_acc", "sequence", "#7b4fc4"), ("multimodal/struct_bit_acc", "structure", "#087f76")):
        rows = acc.Scalars(tag); axes[1].plot([r.step / 1000 for r in rows], [r.value for r in rows], label=label, color=color, alpha=.8)
    axes[1].set(ylim=(.65, 1), xlabel="optimizer steps (k)", ylabel="noisy-bit accuracy"); axes[1].legend(frameon=False)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.svg", transparent=True); plt.close(fig)


def _plot_tasks(out_dir: Path, inverse: dict, folding: dict) -> None:
    import matplotlib.pyplot as plt
    labels = ["conditioned", "shuffled", "unconditional"]
    seq = [inverse[k]["aar_best"]["mean"] for k in labels]
    struct = [folding[k]["bit_best"]["mean"] for k in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(labels, seq, color=["#7b4fc4", "#b9a4db", "#cfd5d1"]); axes[0].set(title="3D → sequence", ylabel="best-of-N amino-acid recovery", ylim=(0, 1))
    axes[1].bar(labels, struct, color=["#087f76", "#79bdb7", "#cfd5d1"]); axes[1].set(title="sequence → 3D", ylabel="best-of-N LFQ bit accuracy", ylim=(0, 1))
    for ax in axes: ax.tick_params(axis="x", rotation=18); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); fig.savefig(out_dir / "conditional_controls.svg", transparent=True); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proteins/m0_v1.py")
    parser.add_argument("--checkpoint", default="runs/proteins/m0_v1/checkpoints/last.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard"))
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dataset = ProteinMultimodalDataset("datasets/dplm_paired_m0", split="test", min_len=40, max_len=512)
    chosen = _select(dataset.lengths, min(args.targets, len(dataset)))
    rows = [dataset[int(i)] for i in chosen]
    model, cfg = load_binary_protein_model(args.config, args.checkpoint, args.device, num_steps=args.steps, context="M0 dashboard")

    references = [_sequence(row["seq_ids"]) for row in rows]
    seq_marginal, struct_marginal = [], []
    marginal_invalid = []
    inverse_records, folding_records = [], []
    forward_top, shuffled_top, unconditional_top = [], [], []
    rng = np.random.default_rng(args.seed)

    for position, row in enumerate(rows):
        length = int(row["length"]); native_seq = references[position]
        native_struct = np.asarray(row["struct_index"], dtype=np.uint16)
        print(f"[{position + 1:02d}/{len(rows)}] length={length}", flush=True)

        torch.manual_seed(args.seed + position * 100)
        seq_out = generate(model, cfg, "sequence_marginal", length, 1, args.device)
        seq_marginal.append(seq_out["seq_strings"][0]); marginal_invalid.append(_raw_invalid_fraction(seq_out["bits"], length))
        torch.manual_seed(args.seed + position * 100 + 1)
        str_out = generate(model, cfg, "structure_marginal", length, 1, args.device)
        struct_marginal.append(np.asarray(str_out["struct_index"][0], dtype=np.uint16))

        conditions = {
            "conditioned": native_struct,
            "shuffled": rng.permutation(native_struct),
            "unconditional": None,
        }
        inv_record = {}
        for offset, (name, observed_struct) in enumerate(conditions.items(), start=2):
            torch.manual_seed(args.seed + position * 100 + offset)
            task = "inverse_folding" if observed_struct is not None else "sequence_marginal"
            observed = {"struct_index": observed_struct} if observed_struct is not None else None
            out = generate(model, cfg, task, length, args.samples, args.device, observed=observed)
            identities = [_identity(sequence, native_seq) for sequence in out["seq_strings"]]
            inv_record[name] = {"aar_mean": float(np.mean(identities)), "aar_best": float(max(identities)), "unique_fraction": len(set(out["seq_strings"])) / args.samples, "raw_invalid_fraction": _raw_invalid_fraction(out["bits"], length)}
        inverse_records.append(inv_record)

        seq_ids = np.asarray(row["seq_ids"], dtype=np.int64)
        fold_conditions = {"conditioned": seq_ids, "shuffled": rng.permutation(seq_ids), "unconditional": None}
        fold_record = {}
        saved = []
        for offset, (name, observed_seq) in enumerate(fold_conditions.items(), start=10):
            torch.manual_seed(args.seed + position * 100 + offset)
            task = "forward_folding" if observed_seq is not None else "structure_marginal"
            observed = {"seq_ids": observed_seq} if observed_seq is not None else None
            out = generate(model, cfg, task, length, args.samples, args.device, observed=observed)
            best, score = _best_struct(np.asarray(out["struct_index"], dtype=np.uint16), native_struct)
            fold_record[name] = score; saved.append(best)
        folding_records.append(fold_record); forward_top.append(saved[0]); shuffled_top.append(saved[1]); unconditional_top.append(saved[2])

    inverse = {name: _condition_summary([record[name] for record in inverse_records], ("aar_mean", "aar_best", "unique_fraction", "raw_invalid_fraction")) for name in ("conditioned", "shuffled", "unconditional")}
    folding = {name: _condition_summary([record[name] for record in folding_records], ("exact_mean", "exact_best", "bit_mean", "bit_best")) for name in ("conditioned", "shuffled", "unconditional")}
    inverse["paired_lifts"] = {
        "conditioned_minus_shuffled_aar_best": _bootstrap([r["conditioned"]["aar_best"] - r["shuffled"]["aar_best"] for r in inverse_records], args.seed),
        "conditioned_minus_unconditional_aar_best": _bootstrap([r["conditioned"]["aar_best"] - r["unconditional"]["aar_best"] for r in inverse_records], args.seed + 1),
    }
    folding["paired_lifts"] = {
        "conditioned_minus_shuffled_bit_best": _bootstrap([r["conditioned"]["bit_best"] - r["shuffled"]["bit_best"] for r in folding_records], args.seed + 2),
        "conditioned_minus_unconditional_bit_best": _bootstrap([r["conditioned"]["bit_best"] - r["unconditional"]["bit_best"] for r in folding_records], args.seed + 3),
    }

    all_ref_struct = np.concatenate([np.asarray(r["struct_index"], dtype=np.uint16) for r in rows])
    all_gen_struct = np.concatenate(struct_marginal)
    seq_basic = basic_metrics(seq_marginal, references)
    seq_basic["raw_invalid_code_fraction"] = float(np.mean(marginal_invalid))
    seq_basic["mean_position_matched_identity"] = float(np.mean([_identity(a, b) for a, b in zip(seq_marginal, references)]))
    struct_basic = {
        "num_samples": len(struct_marginal), "token_jsd_to_test": _jsd_counts(all_gen_struct, all_ref_struct, 8192),
        "unique_token_fraction": float(len(np.unique(all_gen_struct)) / max(1, len(all_gen_struct))),
        "reference_unique_token_fraction": float(len(np.unique(all_ref_struct)) / max(1, len(all_ref_struct))),
        "position_matched_lfq_bit_accuracy": float(np.mean([_struct_scores(a, np.asarray(b["struct_index"]))[1] for a, b in zip(struct_marginal, rows)])),
    }

    maximum = int(max(row["length"] for row in rows)); lengths = np.asarray([row["length"] for row in rows], dtype=np.int64)
    np.savez_compressed(args.out_dir / "structure_tokens.npz", lengths=lengths,
        reference=_pad([np.asarray(r["struct_index"], dtype=np.uint16) for r in rows], maximum),
        forward=_pad(forward_top, maximum), shuffled=_pad(shuffled_top, maximum),
        unconditional=_pad(unconditional_top, maximum), marginal=_pad(struct_marginal, maximum))
    write_fasta(args.out_dir / "sequence_marginal.fasta", seq_marginal, "m0_seq_marginal")
    write_fasta(args.out_dir / "test_references.fasta", references, "cameo_labelled_test")
    payload = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory length-stratified CAMEO-labelled token-cache evaluation",
        "config": args.config, "checkpoint": args.checkpoint,
        "arguments": vars(args) | {"out_dir": str(args.out_dir)},
        "dataset": {"available_test_rows": len(dataset), "evaluated_rows": len(rows), "selected_indices": chosen.tolist(), "lengths": lengths.tolist()},
        "sequence_to_sequence": seq_basic, "structure_to_structure": struct_basic,
        "structure_to_sequence": inverse, "sequence_to_structure": folding,
        "limitations": [
            "LFQ-token cache has no native coordinate arrays",
            f"sampler ran {args.steps} Heun steps",
            "one checkpoint and one evaluation seed",
            "CAMEO-labelled test is not a verified temporal split",
        ],
    }
    atomic_json(args.out_dir / "token_metrics.json", payload)
    _plot_training(args.out_dir, str(cfg.experiment)); _plot_tasks(args.out_dir, inverse, folding)
    print(json.dumps({"sequence_to_sequence": seq_basic, "structure_to_structure": struct_basic, "structure_to_sequence": inverse["paired_lifts"], "sequence_to_structure": folding["paired_lifts"]}, indent=2))


if __name__ == "__main__":
    main()
