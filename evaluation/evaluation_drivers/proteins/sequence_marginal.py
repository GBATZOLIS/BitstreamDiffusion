"""
Unconditional ProteBit sequence generation on frozen EvoDiff UniRef50
rtest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.protein_structure_codec import (  # noqa: E402
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
)
from evaluation.evaluation_drivers.proteins.io import (  # noqa: E402
    atomic_evaluation_json,
    input_identities,
    read_fasta,
    sha256,
    write_fasta,
)
from evaluation.evaluation_drivers.proteins.metrics import (  # noqa: E402
    basic_metrics,
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
from evaluation.evaluation_drivers.proteins.statistics import (  # noqa: E402
    bootstrap_mean_ci,
)

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class SequenceMarginalUnit:
    key: str
    length: int
    reference: str


def load_units(manifest_path: Path) -> tuple[list[SequenceMarginalUnit], dict]:
    """Load the compact prepared benchmark and verify its content identity."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fasta = Path(manifest["references_fasta"])
    if not fasta.is_absolute():
        candidate = manifest_path.parent / fasta
        fasta = candidate if candidate.is_file() else Path(fasta)
    if sha256(fasta) != manifest["references_fasta_sha256"]:
        raise ValueError(
            "sequence-marginal reference FASTA hash does not match manifest"
        )
    references = read_fasta(fasta)
    lengths = [int(value) for value in manifest["lengths"]]
    samples_per_length = int(manifest["samples_per_length"])
    expected_lengths = [
        length for length in lengths for _ in range(samples_per_length)
    ]
    if len(references) != int(manifest["num_samples"]):
        raise ValueError(
            "sequence-marginal manifest count does not match FASTA"
        )
    if [len(sequence) for sequence in references] != expected_lengths:
        raise ValueError(
            "sequence-marginal references are not in the frozen length grid"
        )
    counters: dict[int, int] = {length: 0 for length in lengths}
    units = []
    for reference, length in zip(references, expected_lengths, strict=True):
        index = counters[length]
        counters[length] += 1
        units.append(
            SequenceMarginalUnit(
                key=f"L{length}/{index:04d}",
                length=length,
                reference=reference,
            )
        )
    return units, manifest


def _invalid_code_fraction(bits, length: int) -> float:
    array = np.asarray(torch.as_tensor(bits).cpu(), dtype=np.uint8)
    sequence_bits = array.reshape(-1, int(length), PATCH_BITS_PER_RESIDUE)[
        :, :, :SEQ_BITS_PER_RESIDUE
    ]
    weights = (1 << np.arange(SEQ_BITS_PER_RESIDUE - 1, -1, -1)).astype(
        np.uint8
    )
    ids = sequence_bits @ weights
    return float(np.mean(ids >= len(CANONICAL_AA)))


def _position_identity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    return float(
        sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    )


def evaluate_unit(
    model, cfg, unit: SequenceMarginalUnit, device, *, num_steps=None
) -> dict:
    from evaluation.evaluation_drivers.proteins.generate_multimodal import (
        generate,
    )

    generated = generate(
        model,
        cfg,
        "sequence_marginal",
        unit.length,
        1,
        device,
        observed=None,
        guidance_scale=0.0,
        num_steps=num_steps,
    )
    sequence = str(generated["seq_strings"][0])
    return {
        "key": unit.key,
        "length": unit.length,
        "generated": sequence,
        "reference": unit.reference,
        "valid_canonical": bool(
            sequence and set(sequence) <= set(CANONICAL_AA)
        ),
        "unique_code_invalid_fraction": _invalid_code_fraction(
            generated["bits"], unit.length
        ),
        # A random same-length natural chain is not a target.  This is
        # retained only as a negative-control diagnostic and is
        # explicitly named as such.
        "position_identity_to_random_rtest_reference": _position_identity(
            sequence, unit.reference
        ),
    }


def _summary(values: Sequence[float]) -> dict:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "ci95": bootstrap_mean_ci(finite.tolist()),
    }


def aggregate_sequence_marginal(records: Sequence[dict]) -> dict:
    records = list(records)
    generated = [str(record["generated"]) for record in records]
    references = [str(record["reference"]) for record in records]
    by_length = {}
    for length in sorted({int(record["length"]) for record in records}):
        selected = [
            record for record in records if int(record["length"]) == length
        ]
        by_length[str(length)] = basic_metrics(
            [str(record["generated"]) for record in selected],
            [str(record["reference"]) for record in selected],
        )
    return {
        "protocol": {
            "name": "sequence_marginal_p_s",
            "observed_modality": "none",
            "sampled_modality": "sequence (5 bits per residue)",
            "benchmark": "EvoDiff UniRef50 March-2020 held-out rtest",
            "comparison_note": (
                "This is unconditional generation; sequence inpainting is the "
                "separate sequence-to-sequence task."
            ),
        },
        "overall": basic_metrics(generated, references),
        "by_length": by_length,
        "diagnostics": {
            "raw_invalid_5bit_code_fraction": _summary(
                [record["unique_code_invalid_fraction"] for record in records]
            ),
            "position_identity_to_random_rtest_reference": _summary(
                [
                    record["position_identity_to_random_rtest_reference"]
                    for record in records
                ]
            ),
        },
        "samples": records,
    }


def write_sequence_marginal_outputs(
    out: Path, payload: dict, records: Sequence[dict]
) -> None:
    """
    Write the report plus generated FASTA used by optional heavyweight
    scorers.
    """
    out = Path(out)
    fasta = out.with_suffix(".generated.fasta")
    write_fasta(
        fasta, [str(record["generated"]) for record in records], "protebit"
    )
    payload = dict(payload)
    payload["generated_fasta"] = {"path": str(fasta), "sha256": sha256(fasta)}
    atomic_evaluation_json(out, payload, task="sequence_marginal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=(
            "Optional override; omit for the validation-locked flagship value"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True, type=Path)
    add_shard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = resolve_shard(args)
    units, benchmark = load_units(args.benchmark)
    all_keys = unit_keys(units, key=lambda unit: unit.key)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = {
        "schema_version": 1,
        "task": "sequence_marginal",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "benchmark_manifest": str(args.benchmark),
        "benchmark": benchmark,
        "parameters": {
            "num_steps_override": args.num_steps,
            "seed": int(args.seed),
            "device": str(device),
        },
    }
    writer = ShardWriter(
        plan,
        task="sequence_marginal",
        meta={
            "payload": payload,
            "input_identities": input_identities(
                [
                    args.config,
                    args.checkpoint,
                    args.benchmark,
                    benchmark["references_fasta"],
                ]
            ),
        },
    )
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()
    mine = partition(
        units, plan.shard_id, plan.num_shards, key=lambda unit: unit.key
    )
    todo = [unit for unit in mine if unit.key not in done]

    model = cfg = None
    if todo:
        from evaluation.evaluation_drivers.proteins.io import (
            load_binary_protein_model,
        )

        model, cfg = load_binary_protein_model(
            args.config,
            args.checkpoint,
            device,
            num_steps=args.num_steps,
            context="sequence marginal evaluation",
        )
    for unit in todo:
        random_seed = unit_seed(
            args.seed, unit.key, namespace="sequence_marginal"
        )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed % (2**32))
        record = evaluate_unit(
            model, cfg, unit, device, num_steps=args.num_steps
        )
        record["random_seed"] = int(random_seed)
        writer.write(unit.key, record)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return 0
    records = [record for _key, record in pairs]
    payload["results"] = aggregate_sequence_marginal(records)
    write_sequence_marginal_outputs(args.out, payload, records)
    print(f"Saved sequence-marginal report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
