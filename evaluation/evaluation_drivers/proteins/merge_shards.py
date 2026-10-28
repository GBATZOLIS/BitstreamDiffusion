"""Merge per-shard evaluation records back into one report."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluation_drivers.proteins.sharding import (  # noqa: E402
    ShardLayoutError,
    ShardSet,
    order_records,
    read_shard_dir,
)

__all__ = ["merge", "main"]


class IncompleteShardSet(RuntimeError):
    """
    Raised when the shard set does not cover every work unit exactly
    once.
    """


def _check_complete(shard_set: ShardSet) -> None:
    """
    Refuse to aggregate anything but a complete, duplicate-free shard
    set.
    """
    problems = []
    if shard_set.duplicates:
        sample = list(shard_set.duplicates.items())[:5]
        problems.append(
            f"{len(shard_set.duplicates)} work unit(s) were written by "
            f"more than one shard, for example {sample}. Two shards ran "
            f"the same partition slice; the aggregate would double-count "
            f"them."
        )
    missing = shard_set.missing_keys
    if missing:
        absent = shard_set.missing_shards
        detail = (
            f" Shards {absent} produced no records at all."
            if absent
            else (
                " Every shard produced records, so the gap is inside one of "
                "them."
            )
        )
        problems.append(
            f"{len(missing)} of {len(shard_set.keys)} work units are "
            f"missing, for example {missing[:5]}.{detail} Rerun the "
            f"responsible shard with --resume; do not merge a partial "
            f"matrix."
        )
    if problems:
        raise IncompleteShardSet("\n".join(problems))


def _sharding_block(shard_set: ShardSet) -> dict[str, object]:
    """Describe the layout that produced the report."""
    return {
        "num_shards": shard_set.num_shards,
        "shard_ids": list(shard_set.present_shards),
        "num_work_units": len(shard_set.keys),
        "note": (
            "Merged from independent shard processes. Publication "
            "drivers use stable work-unit-keyed random seeds, so their "
            "samples are invariant to shard count, GPU assignment, and "
            "resume position."
        ),
    }


# -----------------------------------------------------------------------------
# Per-task aggregation
# -----------------------------------------------------------------------------


def _merge_folding(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.folding import (
        aggregate_folding,
    )
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    meta = shard_set.meta
    result = aggregate_folding(
        order_records(shard_set), int(meta["num_samples"])
    )
    result["config"] = meta["config"]
    result["checkpoint"] = meta["checkpoint"]
    result["targets_manifest"] = meta["targets_manifest"]
    result["device"] = meta["device"]
    result["sharding"] = _sharding_block(shard_set)
    atomic_evaluation_json(out, result, task="forward_folding")
    return True


def _merge_inverse_folding(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.inverse_folding import (
        aggregate_inverse_folding,
    )
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    meta = shard_set.meta
    methods = list(meta["methods"])
    # The record kind is the method name, so regrouping needs no key
    # parsing.
    entries_by_method = {
        method: order_records(shard_set, kind=method) for method in methods
    }
    results = aggregate_inverse_folding(
        entries_by_method,
        methods,
        num_samples=int(meta["num_samples"]),
        sc_designs=int(meta["sc_designs"]),
    )
    payload = dict(meta["payload"])
    payload["results"] = results
    payload["sharding"] = _sharding_block(shard_set)
    atomic_evaluation_json(out, payload, task="inverse_folding")
    return True


def _merge_cogeneration(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.cogeneration import (
        _json_default,
        _load_reference_sequences,
        _load_reference_structures,
        aggregate_cogeneration,
        build_reference_index,
        finalize_cogeneration_set_metrics,
    )
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    meta = shard_set.meta
    records = [
        record for samples in order_records(shard_set) for record in samples
    ]
    records.sort(
        key=lambda record: (int(record["length"]), int(record["index"]))
    )
    arguments = dict(meta.get("arguments", {}))
    reference_structures = arguments.get("reference_structures")
    reference_sequences = arguments.get("reference_sequences")
    reference_ca = _load_reference_structures(
        Path(reference_structures) if reference_structures else None
    )
    reference_seq = _load_reference_sequences(
        Path(reference_sequences) if reference_sequences else None,
        lengths=meta["length_grid"],
    )
    reference_ca_by_length, reference_seq_by_length = build_reference_index(
        reference_ca, reference_seq
    )
    from evaluation.evaluation_drivers.proteins import structure_metrics

    records = finalize_cogeneration_set_metrics(
        records,
        reference_ca_by_length=reference_ca_by_length,
        reference_seq_by_length=reference_seq_by_length,
        structure_metrics=structure_metrics,
    )
    foldseek_summary = None
    foldseek_threads = int(
        os.environ.get(
            "PROTEBIT_FOLDSEEK_THREADS",
            meta.get("foldseek_threads", 16),
        )
    )
    if meta.get("foldseek_db"):
        from evaluation.evaluation_drivers.proteins.foldseek import (
            annotate_with_foldseek,
        )

        foldseek_summary = annotate_with_foldseek(
            records,
            foldseek_bin=meta["foldseek_bin"],
            database=meta["foldseek_db"],
            structures_dir=meta["structures_dir"],
            work_dir=meta["foldseek_work_dir"],
            threads=foldseek_threads,
        )
    mmseqs_summary = None
    if meta.get("mmseqs_reference"):
        from evaluation.evaluation_drivers.proteins.mmseqs_novelty import (
            annotate_with_mmseqs,
        )

        mmseqs_summary = annotate_with_mmseqs(
            records,
            mmseqs_bin=meta["mmseqs_bin"],
            reference_fasta=meta["mmseqs_reference"],
            work_dir=meta["mmseqs_work_dir"],
            threads=int(meta.get("mmseqs_threads", 16)),
            gpu=bool(meta.get("mmseqs_gpu", False)),
        )
    results = aggregate_cogeneration(
        records,
        length_grid=meta["length_grid"],
        samples_per_length=int(meta["samples_per_length"]),
        n_design_seqs=int(meta["n_design_seqs"]),
        sc_temperature=float(meta["sc_temperature"]),
        seed=int(meta["seed"]),
        has_references=bool(meta.get("has_references", False)),
        foldseek_summary=foldseek_summary,
        mmseqs_summary=mmseqs_summary,
    )
    payload = dict(meta["payload"])
    payload["results"] = results
    payload["postprocessing_runtime"] = {
        "foldseek_threads_configured": int(meta.get("foldseek_threads", 16)),
        "foldseek_threads_effective": foldseek_threads,
        "mmseqs_threads_effective": int(meta.get("mmseqs_threads", 16)),
        "mmseqs_gpu": bool(meta.get("mmseqs_gpu", False)),
    }
    payload["sharding"] = _sharding_block(shard_set)
    atomic_evaluation_json(
        out,
        payload,
        task="cogeneration",
        default=_json_default,
    )
    return True


def _merge_scaffolding(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.scaffolding import (
        _atomic_write_json,
        _load_structure_metrics,
        aggregate_scaffolding,
        write_scaffold_info_files,
    )

    meta = shard_set.meta
    units = order_records(shard_set)
    results = aggregate_scaffolding(
        units,
        _load_structure_metrics(),
        motif_rmsd_threshold=float(meta["motif_rmsd_threshold"]),
        sc_tm_threshold=float(meta["sc_tm_threshold"]),
    )
    payload = dict(meta["payload"])
    payload["results"] = results
    payload["sharding"] = _sharding_block(shard_set)
    structures_dir = payload.get("motifbench_scaffolds_dir")
    write_scaffold_info_files(
        units, Path(structures_dir) if structures_dir else None
    )
    _atomic_write_json(out, payload)
    return True


def _merge_sequence_marginal(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.sequence_marginal import (
        aggregate_sequence_marginal,
        write_sequence_marginal_outputs,
    )

    records = order_records(shard_set)
    payload = dict(shard_set.meta["payload"])
    payload["results"] = aggregate_sequence_marginal(records)
    payload["sharding"] = _sharding_block(shard_set)
    write_sequence_marginal_outputs(out, payload, records)
    return True


def _merge_inpainting(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.inpainting import (
        aggregate_records,
    )
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    payload = dict(shard_set.meta["payload"])
    payload["results"] = aggregate_records(order_records(shard_set))
    payload["sharding"] = _sharding_block(shard_set)
    atomic_evaluation_json(out, payload, task="inpainting")
    return True


def _merge_entropy_profile(shard_set: ShardSet, out: Path) -> bool:
    from evaluation.evaluation_drivers.proteins.entropy_profile import (
        write_entropy_profile,
    )

    write_entropy_profile(
        order_records(shard_set, kind="profile"), shard_set.meta, out
    )
    return True


_MERGERS: dict[str, Callable[[ShardSet, Path], bool]] = {
    "forward_folding": _merge_folding,
    "inverse_folding": _merge_inverse_folding,
    "cogeneration": _merge_cogeneration,
    "motif_scaffolding": _merge_scaffolding,
    "sequence_marginal": _merge_sequence_marginal,
    "inpainting": _merge_inpainting,
    "entropy_profile": _merge_entropy_profile,
}


def merge(shard_dir, out) -> tuple[str, bool]:
    """Merge one shard directory into out; returns (task, all_gates_passed)."""
    shard_set = read_shard_dir(shard_dir)
    if shard_set.task not in _MERGERS:
        raise ShardLayoutError(
            f"no merger registered for task {shard_set.task!r}; known "
            f"tasks are {sorted(_MERGERS)}."
        )
    _check_complete(shard_set)
    return shard_set.task, _MERGERS[shard_set.task](shard_set, Path(out))


def main(argv=None) -> int:
    """
    Command-line entry point for merging a shard directory into one
    report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        required=True,
        type=Path,
        help="Directory of shard-<id>-of-<n>.jsonl files written by a driver",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path of the merged report, matching the driver's --out",
    )
    args = parser.parse_args(argv)

    try:
        task, all_passed = merge(args.shard_dir, args.out)
    except (ShardLayoutError, IncompleteShardSet) as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        return 2
    # rc=1 is reserved for metric gates
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"merge failed operationally: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    print(f"merged {task} shards from {args.shard_dir} into {args.out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
