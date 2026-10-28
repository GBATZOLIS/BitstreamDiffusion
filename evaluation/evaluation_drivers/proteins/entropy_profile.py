"""
Estimate an entropy schedule from a frozen checkpoint and validation
data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data import protein_tasks as task_states  # noqa: E402
from data.protein_multimodal import CANONICAL_AA  # noqa: E402
from data.protein_structure_codec import (  # noqa: E402
    DEFAULT_STRUCT_CODEC,
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
)
from data.proteins import build_token_to_bits_table  # noqa: E402
from diffusion.continuous.processes import (
    ContinuousForwardProcess,  # noqa: E402
)
from evaluation.evaluation_drivers.proteins import (  # noqa: E402
    inverse_folding,
)
from evaluation.evaluation_drivers.proteins.io import (  # noqa: E402
    atomic_json,
    load_binary_protein_model,
    repository_commit,
    sha256,
)
from evaluation.evaluation_drivers.proteins.sharding import (  # noqa: E402
    ShardWriter,
    add_shard_args,
    finalise,
    partition,
    resolve_shard,
    unit_keys,
    unit_seed,
)
from trainers.multimodal_step import multimodal_training_step  # noqa: E402
from utils.schedule_controller import regularized_entropy_weights  # noqa: E402

_StructureTokenResolver = inverse_folding._StructureTokenResolver
load_backbone_records = inverse_folding.load_backbone_records

PROFILE_TASKS = (
    "joint",
    "forward_folding",
    "inverse_folding",
    "sequence_marginal",
    "structure_marginal",
)
_AA_TO_ID = {aa: index for index, aa in enumerate(CANONICAL_AA)}
_SEQ_BITS = (
    build_token_to_bits_table(len(CANONICAL_AA), SEQ_BITS_PER_RESIDUE)
    .numpy()
    .astype(np.float32)
)


def select_validation_records(records, count: int, seed: int):
    """Choose a deterministic random validation subset without replacement."""
    records = list(records)
    count = int(count)
    if count < 1:
        raise ValueError(
            f"validation subset size must be positive, got {count}"
        )
    if count > len(records):
        raise ValueError(
            f"requested {count} validation backbones but only "
            f"{len(records)} matched"
        )
    rng = np.random.default_rng(int(seed))
    chosen = sorted(
        int(index) for index in rng.choice(len(records), count, replace=False)
    )
    return [records[index] for index in chosen]


def _unit_key(unit: tuple[str, str]) -> str:
    task, stable_id = unit
    return f"{task}/{stable_id}"


def _clean_batch(
    sequence: str,
    struct_index: np.ndarray,
    task: str,
    samples: int,
    *,
    absent_fill: float,
) -> dict:
    length = len(sequence)
    if len(struct_index) != length:
        raise ValueError("sequence and structure-token lengths differ")
    try:
        seq_ids = np.asarray(
            [_AA_TO_ID[aa] for aa in sequence], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(f"non-canonical amino acid {exc.args[0]!r}") from exc
    seq_bits = _SEQ_BITS[seq_ids]
    struct_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
        np.asarray(struct_index, dtype=np.int64)
    ).astype(np.float32)
    patch = np.concatenate([seq_bits, struct_bits], axis=1)

    state = task_states.build_example_states(task, length)
    states = np.repeat(state[None, :, :], int(samples), axis=0)
    residue_mask = np.ones((int(samples), length), dtype=bool)
    x0 = np.repeat(patch[None, :, :], int(samples), axis=0)
    absent = task_states.absent_bit_mask_np(states).reshape(
        int(samples), length, PATCH_BITS_PER_RESIDUE
    )
    x0[absent] = float(absent_fill)
    seq_full, struct_full = task_states.full_modality_target_masks_np(
        states, residue_mask
    )
    return {
        "x0": torch.from_numpy(x0.reshape(int(samples), -1)).float(),
        "states": torch.from_numpy(states.astype(np.int64)),
        "seq_target_full": torch.from_numpy(seq_full),
        "struct_target_full": torch.from_numpy(struct_full),
        "residue_mask": torch.from_numpy(residue_mask),
    }


def _stratified_sigmas(
    *,
    num_bins: int,
    samples: int,
    sigma_min: float,
    sigma_max: float,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    start = int(seed) % int(num_bins)
    bins = (start + np.arange(int(samples)) * 17) % int(num_bins)
    log_edges = np.linspace(
        math.log(float(sigma_min)),
        math.log(float(sigma_max)),
        int(num_bins) + 1,
    )
    u = rng.random(int(samples))
    log_sigma = log_edges[bins] + u * (log_edges[bins + 1] - log_edges[bins])
    return torch.from_numpy(np.exp(log_sigma).astype(np.float32)), bins


class _FixedSigmaDraw:
    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas

    def __call__(self, batch_size: int) -> torch.Tensor:
        if int(batch_size) != int(self.sigmas.numel()):
            raise ValueError("fixed sigma vector does not match profile batch")
        return self.sigmas.clone()


@torch.no_grad()
def profile_unit(
    model,
    cfg,
    proc,
    device,
    *,
    sequence: str,
    struct_index: np.ndarray,
    task: str,
    num_bins: int,
    samples_per_unit: int,
    seed: int,
) -> dict:
    sigma_min = float(cfg.diffusion.continuous.sigma_min)
    sigma_max = float(cfg.diffusion.continuous.sigma_max)
    sigmas, bins = _stratified_sigmas(
        num_bins=num_bins,
        samples=samples_per_unit,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        seed=seed,
    )
    batch = _clean_batch(
        sequence,
        struct_index,
        task,
        samples_per_unit,
        absent_fill=float(cfg.diffusion.continuous.data_center),
    )
    entropy: dict[str, torch.Tensor] = {}
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    multimodal_training_step(
        model,
        batch,
        proc,
        cfg,
        device=device,
        is_train=False,
        sigma_draw_fn=_FixedSigmaDraw(sigmas.to(device)),
        entropy_sink=entropy,
    )

    result: dict[str, object] = {
        "task": task,
        "num_bins": int(num_bins),
        "seq_sum_rate": [0.0] * int(num_bins),
        "seq_count": [0] * int(num_bins),
        "struct_sum_rate": [0.0] * int(num_bins),
        "struct_count": [0] * int(num_bins),
    }
    for modality in ("seq", "struct"):
        sigma = entropy[f"sigma_{modality}"].detach().float().cpu().numpy()
        metric = entropy[f"metric_{modality}"].detach().float().cpu().numpy()
        valid = (
            entropy[f"valid_{modality}"].detach().cpu().numpy().astype(bool)
        )
        rates = metric / np.maximum(sigma**2, 1e-12)
        sums = np.zeros(int(num_bins), dtype=np.float64)
        counts = np.zeros(int(num_bins), dtype=np.int64)
        np.add.at(sums, bins[valid], rates[valid])
        np.add.at(counts, bins[valid], 1)
        result[f"{modality}_sum_rate"] = sums.tolist()
        result[f"{modality}_count"] = counts.tolist()
    return result


def aggregate_entropy_profile(
    records: Sequence[Mapping[str, object]],
    *,
    num_bins: int,
    sigma_min: float,
    sigma_max: float,
    regularizer_c: float,
    regularizer_n: float,
    rate_power: float,
) -> dict:
    sums = {
        modality: np.zeros(int(num_bins), dtype=np.float64)
        for modality in ("seq", "struct")
    }
    counts = {
        modality: np.zeros(int(num_bins), dtype=np.int64)
        for modality in ("seq", "struct")
    }
    for record in records:
        if int(record["num_bins"]) != int(num_bins):
            raise ValueError("entropy-profile records disagree on num_bins")
        for modality in ("seq", "struct"):
            sums[modality] += np.asarray(
                record[f"{modality}_sum_rate"], dtype=np.float64
            )
            counts[modality] += np.asarray(
                record[f"{modality}_count"], dtype=np.int64
            )
    total_sum = sums["seq"] + sums["struct"]
    total_count = counts["seq"] + counts["struct"]
    if np.any(total_count == 0):
        missing = np.flatnonzero(total_count == 0).tolist()
        raise ValueError(
            f"entropy profile has empty sigma bins: {missing[:10]}"
        )
    edges = np.exp(
        np.linspace(
            math.log(float(sigma_min)),
            math.log(float(sigma_max)),
            int(num_bins) + 1,
        )
    )
    mids = np.sqrt(edges[:-1] * edges[1:])
    mean_rate = total_sum / total_count
    unnormalized = (
        regularized_entropy_weights(
            torch.from_numpy(mean_rate),
            torch.from_numpy(mids),
            c=regularizer_c,
            n=regularizer_n,
            power=rate_power,
        )
        .cpu()
        .numpy()
    )
    if not np.isfinite(unnormalized).all() or unnormalized.sum() <= 0:
        raise ValueError("entropy profile produced a non-finite/zero density")
    pdf = unnormalized / unnormalized.sum()
    cdf = np.cumsum(pdf)
    cdf[-1] = 1.0
    return {
        "pdf": pdf.astype(np.float32),
        "cdf": cdf.astype(np.float32),
        "sigmas": mids.astype(np.float32),
        "edges": edges.astype(np.float32),
        "mean_rate": mean_rate.astype(np.float32),
        "counts": total_count,
        "seq_counts": counts["seq"],
        "struct_counts": counts["struct"],
    }


def _atomic_torch_save(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def write_entropy_profile(
    records: Sequence[Mapping[str, object]],
    meta: Mapping[str, object],
    out: Path,
) -> dict:
    result = aggregate_entropy_profile(
        records,
        num_bins=int(meta["num_bins"]),
        sigma_min=float(meta["sigma_min"]),
        sigma_max=float(meta["sigma_max"]),
        regularizer_c=float(meta["regularizer_c"]),
        regularizer_n=float(meta["regularizer_n"]),
        rate_power=float(meta["rate_power"]),
    )
    entropy_out_dir = Path(str(meta["entropy_out_dir"]))
    tensors = {
        "entropy_pdf.pt": result["pdf"],
        "entropy_cdf.pt": result["cdf"],
        "entropy_sigmas.pt": result["sigmas"],
        "entropy_edges.pt": result["edges"],
    }
    for filename, values in tensors.items():
        _atomic_torch_save(
            entropy_out_dir / filename,
            torch.from_numpy(np.asarray(values, dtype=np.float32)),
        )
    table_hashes = {
        filename: sha256(entropy_out_dir / filename) for filename in tensors
    }
    payload = dict(meta["payload"])
    payload["results"] = {
        "num_profile_units": len(records),
        "num_bins": int(meta["num_bins"]),
        "total_samples": int(np.asarray(result["counts"]).sum()),
        "samples_per_bin_min": int(np.asarray(result["counts"]).min()),
        "samples_per_bin_max": int(np.asarray(result["counts"]).max()),
        "seq_samples": int(np.asarray(result["seq_counts"]).sum()),
        "struct_samples": int(np.asarray(result["struct_counts"]).sum()),
        "entropy_tables": table_hashes,
    }
    atomic_json(out, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--struct-tokenizer")
    parser.add_argument("--entropy-out-dir", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", default=list(PROFILE_TASKS))
    parser.add_argument("--num-validation-backbones", type=int, default=128)
    parser.add_argument("--samples-per-unit", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--subset-seed", type=int, default=20260815)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="auto")
    add_shard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = resolve_shard(args)
    if args.validation_data.name.lower().startswith("test"):
        raise SystemExit(
            "entropy profiling refuses a file named as a test split"
        )
    tasks = list(dict.fromkeys(str(task) for task in args.tasks))
    unknown = sorted(set(tasks).difference(PROFILE_TASKS))
    if unknown:
        raise SystemExit(f"unsupported entropy-profile tasks: {unknown}")
    if args.num_bins < 2 or args.samples_per_unit < 1:
        raise SystemExit(
            "--num-bins must be >=2 and --samples-per-unit positive"
        )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from evaluation.utils import load_config

    profile_cfg = load_config(args.config)
    profile = profile_cfg.evaluation.entropy
    configured_power = float(profile.rate_power)
    loaded = load_backbone_records(
        args.validation_data,
        max_len=int(profile_cfg.data.max_len),
        default_split="validation",
    )
    test_records = [
        record.stable_id
        for record in loaded
        if str(record.split).lower() == "test"
    ]
    if test_records:
        raise SystemExit(
            f"entropy profiling refuses records labelled test; first IDs: "
            f"{test_records[:5]}"
        )
    records = select_validation_records(
        loaded, args.num_validation_backbones, args.subset_seed
    )
    by_id = {record.stable_id: record for record in records}
    units = [(task, record.stable_id) for task in tasks for record in records]
    all_keys = unit_keys(units, key=_unit_key)

    sigma_min = float(profile_cfg.diffusion.continuous.sigma_min)
    sigma_max = float(profile_cfg.diffusion.continuous.sigma_max)
    payload = {
        "schema_version": 1,
        "study_type": "frozen_multimodal_entropy_profile",
        "training": False,
        "test_set_used": False,
        "config": {"path": args.config, "sha256": sha256(Path(args.config))},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "weights_modified": False,
        },
        "validation": {
            "path": str(args.validation_data),
            "sha256": sha256(args.validation_data),
            "selected_target_ids": [record.stable_id for record in records],
            "subset_seed": int(args.subset_seed),
        },
        "tasks": tasks,
        "entropy_schedule": {
            "profile_weight_implementation": (
                "utils.schedule_controller.regularized_entropy_weights"
            ),
            "integration_grid_implementation": (
                "diffusion.continuous.samplers.SigmaSchedule"
            ),
            ("rate_proxy"): (
                "per-example denoising metric divided by sigma squared"
            ),
            "target": "sqrt-rate",
            "rate_power": configured_power,
            "regularizer": "sigma^n / (sigma^n + c^n)",
            "regularizer_c": float(profile.regularizer_c),
            "regularizer_n": float(profile.regularizer_n),
            "base_fraction": 0.0,
            "inverse_cdf": True,
            "entropic_blend_alpha": float(
                profile_cfg.evaluation.entropic_blend_alpha
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "git_commit": repository_commit(_REPO_ROOT),
        },
    }
    meta = {
        "payload": payload,
        "num_bins": int(args.num_bins),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "regularizer_c": float(profile.regularizer_c),
        "regularizer_n": float(profile.regularizer_n),
        "rate_power": configured_power,
        "entropy_out_dir": str(args.entropy_out_dir),
    }
    writer = ShardWriter(plan, task="entropy_profile", meta=meta)
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()
    mine = partition(units, plan.shard_id, plan.num_shards, key=_unit_key)
    todo = [unit for unit in mine if _unit_key(unit) not in done]

    if todo:
        model, cfg = load_binary_protein_model(
            args.config,
            args.checkpoint,
            device,
            context="frozen entropy profiling",
        )
        cfg.train.global_loss_normalizer = False
        proc = ContinuousForwardProcess(cfg)
        resolver = _StructureTokenResolver(
            args.struct_tokenizer, device=str(device)
        )
        struct_cache: dict[str, np.ndarray] = {}
        for task, stable_id in todo:
            record = by_id[stable_id]
            if stable_id not in struct_cache:
                struct_cache[stable_id] = resolver.resolve(record)
            seed = unit_seed(
                args.seed,
                _unit_key((task, stable_id)),
                namespace="flagship-entropy-profile",
            )
            result = profile_unit(
                model,
                cfg,
                proc,
                device,
                sequence=record.native_seq,
                struct_index=struct_cache[stable_id],
                task=task,
                num_bins=args.num_bins,
                samples_per_unit=args.samples_per_unit,
                seed=seed,
            )
            result["stable_id"] = stable_id
            result["length"] = int(record.length)
            result["seed"] = int(seed)
            writer.write(_unit_key((task, stable_id)), result, kind="profile")

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return 0
    write_entropy_profile([record for _key, record in pairs], meta, args.out)
    print(f"Saved entropy profile report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
