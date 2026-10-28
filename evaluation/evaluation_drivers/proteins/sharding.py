"""Split evaluation work into resumable shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ShardPlan",
    "ShardWriter",
    "ShardSet",
    "ShardLayoutError",
    "add_shard_args",
    "resolve_shard",
    "partition",
    "unit_keys",
    "shard_rng",
    "unit_seed",
    "shard_root",
    "read_jsonl",
    "read_shard_dir",
    "order_records",
    "finalise",
]

_SHARD_FILE_RE = re.compile(r"^shard-(\d+)-of-(\d+)\.jsonl$")
_MANIFEST_FILE_RE = re.compile(r"^manifest-(\d+)-of-(\d+)\.json$")


class ShardLayoutError(RuntimeError):
    """Raised when a shard directory cannot describe one coherent run."""


def _comparable_meta(value):
    """Drop only intentionally volatile fields before manifest comparison."""
    if isinstance(value, dict):
        return {
            key: _comparable_meta(item)
            for key, item in value.items()
            if key != "created_at_utc"
        }
    if isinstance(value, list):
        return [_comparable_meta(item) for item in value]
    return value


# -----------------------------------------------------------------------------
# Argument plumbing
# -----------------------------------------------------------------------------


def add_shard_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shard/resume flags to a driver's parser."""
    group = parser.add_argument_group("sharding")
    group.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Index of this shard (default: SLURM_PROCID, else RANK, else 0)",
    )
    group.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help=(
            "Total number of shards (default: SLURM_NTASKS, else "
            "WORLD_SIZE, else 1)"
        ),
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Skip work units already recorded in this shard's JSONL",
    )
    try:
        parser.add_argument(
            "--out", required=True, type=Path, help="Output report JSON path"
        )
    except argparse.ArgumentError:
        pass
    return parser


def _env_int(name: str) -> int | None:
    """
    Read an integer environment variable, ignoring unset or unparsable
    values.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def shard_root(out) -> Path:
    """Directory holding the per-shard artefacts for a report path."""
    path = Path(out)
    base = path.with_suffix("") if path.suffix else path
    return base / "shards"


@dataclass(frozen=True)
class ShardPlan:
    """Resolved sharding for one driver process."""

    shard_id: int
    num_shards: int
    resume: bool
    out: Path
    shard_dir: Path

    @property
    def is_single(self) -> bool:
        """
        True when this process owns every work unit and writes the
        report itself.
        """
        return self.num_shards == 1


def resolve_shard(args: argparse.Namespace) -> ShardPlan:
    """
    Resolve the shard identity from the command line and the Slurm
    environment.
    """
    num_shards = getattr(args, "num_shards", None)
    if num_shards is None:
        num_shards = _env_int("SLURM_NTASKS")
    if num_shards is None:
        num_shards = _env_int("WORLD_SIZE")
    if num_shards is None:
        num_shards = 1

    shard_id = getattr(args, "shard_id", None)
    if shard_id is None:
        shard_id = _env_int("SLURM_PROCID")
    if shard_id is None:
        shard_id = _env_int("RANK")
    if shard_id is None:
        shard_id = 0

    num_shards = int(num_shards)
    shard_id = int(shard_id)
    if num_shards < 1:
        raise ValueError(f"--num-shards must be at least 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"--shard-id {shard_id} is outside [0, {num_shards}); check "
            f"SLURM_PROCID against SLURM_NTASKS."
        )

    out = Path(args.out)
    return ShardPlan(
        shard_id=shard_id,
        num_shards=num_shards,
        resume=bool(getattr(args, "resume", False)),
        out=out,
        shard_dir=shard_root(out),
    )


def shard_rng(seed: int, plan: ShardPlan) -> np.random.Generator:
    """Per-shard numpy generator for controls and other nuisance draws."""
    if plan.num_shards == 1:
        return np.random.default_rng(int(seed))
    return np.random.default_rng([int(seed), int(plan.shard_id)])


def unit_seed(
    seed: int, key: object, *, namespace: str = "protein-eval"
) -> int:
    """
    Derive a stable per-work-unit seed, independent of sharding and
    resume.
    """
    material = f"{int(seed)}\0{namespace}\0{key}".encode()
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


# -----------------------------------------------------------------------------
# Partitioning
# -----------------------------------------------------------------------------


def unit_keys(
    work_units: Sequence, key: Callable[[object], str] = str
) -> list[str]:
    """
    Return the work-unit keys in the caller's order, rejecting
    duplicates.
    """
    keys = [str(key(unit)) for unit in work_units]
    seen: dict[str, int] = {}
    for position, value in enumerate(keys):
        if value in seen:
            raise ValueError(
                f"duplicate work-unit key {value!r} at positions "
                f"{seen[value]} and {position}; work-unit keys must be "
                f"unique for resume to be correct."
            )
        seen[value] = position
    return keys


def partition(
    work_units: Sequence,
    shard_id: int,
    num_shards: int,
    *,
    key: Callable[[object], str] = str,
) -> list:
    """Return the work units belonging to one shard."""
    units = list(work_units)
    if num_shards < 1:
        raise ValueError(f"num_shards must be at least 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} is outside [0, {num_shards})")
    if num_shards == 1:
        unit_keys(units, key)  # still reject duplicates, so resume stays sound
        return units

    keys = unit_keys(units, key)
    by_key = sorted(range(len(units)), key=lambda i: keys[i])
    mine = set(by_key[shard_id::num_shards])
    return [unit for i, unit in enumerate(units) if i in mine]


# -----------------------------------------------------------------------------
# Writing
# -----------------------------------------------------------------------------


def _json_default(value):
    """Coerce numpy scalars and arrays so any driver's record serialises."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


class ShardWriter:
    """
    Append-only JSONL sink for one shard, plus the manifest that
    describes it.
    """

    def __init__(
        self, plan: ShardPlan, *, task: str, meta: dict | None = None
    ):
        self.plan = plan
        self.task = str(task)
        self.meta = dict(meta or {})
        # CLI arguments alone do not identify a final model system: the
        # same config/checkpoint path can load a different
        # validation-locked sampler after a selection file changes.
        # Persist the launcher-provided hashes in every manifest so
        # resume rejects cross-recipe shard mixtures.
        execution_identity = {
            key: os.environ[name]
            for key, name in (
                ("evaluation_id", "PROTEBIT_EVAL_ID"),
                ("evaluation_protocol_id", "PROTEBIT_EVAL_PROTOCOL_ID"),
                ("checkpoint_sha256", "PROTEBIT_CHECKPOINT_SHA256"),
                ("sampler_protocol_id", "PROTEBIT_SAMPLER_PROTOCOL_ID"),
                (
                    "sampler_selection_sha256",
                    "PROTEBIT_SAMPLER_SELECTION_SHA256",
                ),
            )
            if os.environ.get(name)
        }
        if execution_identity:
            self.meta["execution_identity"] = execution_identity
        self.shard_dir = Path(plan.shard_dir)
        self._handle = None

    @property
    def path(self) -> Path:
        return self.shard_dir / (
            f"shard-{self.plan.shard_id}-of-{self.plan.num_shards}.jsonl"
        )

    @property
    def manifest_path(self) -> Path:
        return self.shard_dir / (
            f"manifest-{self.plan.shard_id}-of-{self.plan.num_shards}.json"
        )

    def write_manifest(self, all_keys: Sequence[str]) -> None:
        """
        Record the full key universe and the driver metadata for this
        run.
        """
        payload = {
            "task": self.task,
            "shard_id": int(self.plan.shard_id),
            "num_shards": int(self.plan.num_shards),
            "keys": [str(key) for key in all_keys],
            "meta": self.meta,
        }
        if self.plan.resume and self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            previous_identity = {
                "task": previous.get("task"),
                "shard_id": previous.get("shard_id"),
                "num_shards": previous.get("num_shards"),
                "keys": previous.get("keys"),
                "meta": _comparable_meta(previous.get("meta", {})),
            }
            current_identity = {
                "task": payload["task"],
                "shard_id": payload["shard_id"],
                "num_shards": payload["num_shards"],
                "keys": payload["keys"],
                "meta": _comparable_meta(payload["meta"]),
            }
            if previous_identity != current_identity:
                raise ShardLayoutError(
                    f"{self.manifest_path} belongs to an incompatible "
                    f"evaluation configuration. Refusing to reuse its "
                    f"records; choose a new --out or launch once with "
                    f"PROTEBIT_EVAL_FRESH=1."
                )
            return
        if (
            self.plan.resume
            and self.path.is_file()
            and self.path.stat().st_size > 0
        ):
            raise ShardLayoutError(
                f"{self.path} has resumable records but its manifest is "
                f"missing; their configuration cannot be verified. Choose "
                f"a new --out or launch once with PROTEBIT_EVAL_FRESH=1."
            )
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
        os.replace(temporary, self.manifest_path)

    def completed_keys(self) -> set[str]:
        """Keys already recorded in this shard's JSONL, for the resume skip."""
        return {key for key, _kind, _record in read_jsonl(self.path)}

    def completed_records(self) -> dict[str, dict]:
        """Every record this shard has written, keyed by work-unit key."""
        return {key: record for key, _kind, record in read_jsonl(self.path)}

    def write(self, key: str, record, *, kind: str = "unit") -> None:
        """Append one completed work unit and force it to disk."""
        if self._handle is None:
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        line = json.dumps(
            {"key": str(key), "kind": str(kind), "record": record},
            default=_json_default,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def ensure_file(self) -> None:
        """Create the shard file even when this shard had no units to run."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def reset(self) -> None:
        """Discard this shard's records so a non-resumed run starts clean."""
        self.close()
        if self.path.exists():
            self.path.unlink()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Reading and merging
# -----------------------------------------------------------------------------


def read_jsonl(path) -> list[tuple[str, str, dict]]:
    """Read a shard JSONL into (key, kind, record) triples."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    out: list[tuple[str, str, dict]] = []
    for position, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if position == len(lines) - 1:
                # Killed mid-write. The unit is simply not done;
                # --resume redoes it.
                continue
            raise ShardLayoutError(
                f"{path}: line {position + 1} is not valid JSON and is not"
                f" the final line, so this is corruption rather than an "
                f"interrupted write."
            ) from exc
        out.append(
            (
                str(payload["key"]),
                str(payload.get("kind", "unit")),
                payload["record"],
            )
        )
    return out


@dataclass
class ShardSet:
    """
    Every shard artefact found under one shard directory, structurally
    checked.
    """

    task: str
    num_shards: int
    keys: list[str]
    meta: dict
    records: dict[str, dict] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    duplicates: dict[str, list[int]] = field(default_factory=dict)
    present_shards: list[int] = field(default_factory=list)

    @property
    def missing_keys(self) -> list[str]:
        return [key for key in self.keys if key not in self.records]

    @property
    def missing_shards(self) -> list[int]:
        return [
            i
            for i in range(self.num_shards)
            if i not in set(self.present_shards)
        ]


def read_shard_dir(shard_dir) -> ShardSet:
    """
    Load every shard JSONL under shard_dir and check they describe one
    run.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        raise ShardLayoutError(f"shard directory does not exist: {shard_dir}")

    manifests: dict[int, dict] = {}
    layouts: set[int] = set()
    for path in sorted(shard_dir.iterdir()):
        match = _MANIFEST_FILE_RE.match(path.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        manifests[int(match.group(1))] = payload
        layouts.add(int(match.group(2)))

    if not manifests:
        raise ShardLayoutError(
            f"no manifest-<id>-of-<n>.json under {shard_dir}; the drivers "
            f"write one before doing any work, so either nothing ran or "
            f"the path is wrong."
        )
    if len(layouts) != 1:
        raise ShardLayoutError(
            f"{shard_dir} mixes shard layouts {sorted(layouts)}. Two runs "
            f"with different --num-shards partition the work differently; "
            f"delete the stale shards and rerun rather than merging across"
            f" layouts."
        )
    num_shards = layouts.pop()

    reference_id = min(manifests)
    reference = manifests[reference_id]
    task = str(reference["task"])
    keys = [str(key) for key in reference["keys"]]
    reference_meta = _comparable_meta(reference.get("meta", {}))
    for shard_id, payload in manifests.items():
        if str(payload["task"]) != task:
            raise ShardLayoutError(
                f"{shard_dir}: shard {shard_id} ran task "
                f"{payload['task']!r} but shard {reference_id} ran "
                f"{task!r}."
            )
        if [str(key) for key in payload["keys"]] != keys:
            raise ShardLayoutError(
                f"{shard_dir}: shard {shard_id} enumerated a different set"
                f" of work units than shard {reference_id}. The shards "
                f"were launched with different inputs (a changed manifest,"
                f" --max-examples, --splits or similar); rerun them with "
                f"identical arguments."
            )
        if _comparable_meta(payload.get("meta", {})) != reference_meta:
            raise ShardLayoutError(
                f"{shard_dir}: shard {shard_id} has different evaluation "
                f"parameters or provenance from shard {reference_id}. "
                f"Refusing to combine records from incompatible runs."
            )

    shard_set = ShardSet(
        task=task,
        num_shards=num_shards,
        keys=keys,
        meta=dict(reference.get("meta", {})),
    )
    seen_in: dict[str, list[int]] = {}
    for path in sorted(shard_dir.iterdir()):
        match = _SHARD_FILE_RE.match(path.name)
        if not match:
            continue
        shard_id, layout = int(match.group(1)), int(match.group(2))
        if layout != num_shards:
            raise ShardLayoutError(
                f"{path.name} does not match the manifest layout of "
                f"{num_shards} shards."
            )
        shard_set.present_shards.append(shard_id)
        for key, kind, record in read_jsonl(path):
            seen_in.setdefault(key, []).append(shard_id)
            shard_set.records[key] = record
            shard_set.kinds[key] = kind

    shard_set.present_shards.sort()
    shard_set.duplicates = {
        key: sorted(shards)
        for key, shards in seen_in.items()
        if len(shards) > 1
    }
    return shard_set


def finalise(
    plan: ShardPlan, writer: ShardWriter, all_keys: Sequence[str]
) -> list[tuple[str, dict]] | None:
    """
    Close a shard and return the records for the report, or None if it
    is partial.
    """
    writer.close()
    writer.ensure_file()
    if not plan.is_single:
        print(
            f"shard {plan.shard_id}/{plan.num_shards} wrote {writer.path}"
            "\nMerge with: python -m "
            "evaluation.evaluation_drivers.proteins.merge_shards "
            f"--shard-dir {plan.shard_dir} --out {plan.out}"
        )
        return None

    records = writer.completed_records()
    missing = [key for key in all_keys if key not in records]
    if missing:
        raise ShardLayoutError(
            f"{len(missing)} of {len(all_keys)} work units are missing "
            f"from {writer.path} (first: {missing[:5]}). Rerun with "
            f"--resume."
        )
    return [(key, records[key]) for key in all_keys]


def order_records(shard_set: ShardSet, kind: str | None = None) -> list[dict]:
    """
    Return the merged records in the manifest's original enumeration
    order.
    """
    return [
        shard_set.records[key]
        for key in shard_set.keys
        if key in shard_set.records
        and (kind is None or shard_set.kinds.get(key) == kind)
    ]
