from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def repository_commit(root: Path) -> str | None:
    """Read HEAD without invoking git or scanning a shared Lustre worktree."""
    overridden = os.environ.get("PROTEBIT_GIT_COMMIT")
    if overridden:
        return overridden.strip() or None
    git_dir = Path(root) / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir:"):
            return None
        git_dir = (git_dir.parent / text.split(":", 1)[1].strip()).resolve()
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    reference = head.split(":", 1)[1].strip()
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip() or None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, name = line.split(" ", 1)
            if name == reference:
                return commit
    return None


def file_identity(path: Path | str) -> dict:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def input_identities(paths: Sequence[Path | str | None]) -> list[dict]:
    """
    Capture fail-closed identities for files and immediate directory
    members.
    """
    identities: list[dict] = []
    seen: set[str] = set()
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        candidates = (
            sorted(item for item in path.iterdir() if item.is_file())
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            if not candidate.is_file():
                identities.append({"path": resolved, "missing": True})
            else:
                identities.append(file_identity(candidate))
    return identities


def verify_checkpoint_identity(path: Path | str) -> None:
    """Reject a checkpoint whose file identity changed after full preflight."""
    identity_path = os.environ.get("PROTEBIT_CHECKPOINT_IDENTITY")
    if not identity_path:
        return
    payload = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    checkpoint = Path(path)
    current = file_identity(checkpoint)
    expected = payload.get("file_identity", {})
    expected_sha = os.environ.get("PROTEBIT_CHECKPOINT_SHA256")
    if expected_sha and payload.get("sha256") != expected_sha:
        raise RuntimeError(
            "checkpoint identity cache does not match "
            "PROTEBIT_CHECKPOINT_SHA256"
        )
    if current != expected:
        raise RuntimeError(
            "checkpoint changed after publication preflight; refusing to"
            " load it"
        )


def read_fasta(path: Path) -> list[str]:
    sequences, pieces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pieces:
                    sequences.append("".join(pieces).upper())
                    pieces = []
            else:
                pieces.append(line)
    if pieces:
        sequences.append("".join(pieces).upper())
    return sequences


def write_fasta(path: Path, sequences: Sequence[str], prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(
                f">{prefix}_{index} length={len(sequence)}\n{sequence}\n"
            )


def atomic_json(path: Path, payload: dict, *, default=str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
        handle.write("\n")
    os.replace(temporary, path)


def mark_evaluation_complete(payload: dict, *, task: str) -> dict:
    """
    Attach the completion and frozen-protocol identity used by launch
    checks.
    """
    report = dict(payload)
    report["completion"] = {
        "complete": True,
        "task": str(task),
        "evaluation_id": os.environ.get("PROTEBIT_EVAL_ID"),
        "protocol_id": os.environ.get("PROTEBIT_EVAL_PROTOCOL_ID"),
        "checkpoint_sha256": os.environ.get("PROTEBIT_CHECKPOINT_SHA256"),
        "sampler_protocol_id": os.environ.get("PROTEBIT_SAMPLER_PROTOCOL_ID"),
        "sampler_selection_sha256": os.environ.get(
            "PROTEBIT_SAMPLER_SELECTION_SHA256"
        ),
    }
    return report


def atomic_evaluation_json(
    path: Path, payload: dict, *, task: str, default=str
) -> None:
    """
    Write a complete evaluation report atomically with an identity
    marker.
    """
    atomic_json(
        path,
        mark_evaluation_complete(payload, task=task),
        default=default,
    )


def load_binary_protein_model(
    config_path,
    checkpoint_path,
    device,
    *,
    num_steps=None,
    context: str = "this evaluation",
    return_checkpoint_metadata: bool = False,
):
    """
    Load an 18-bit multimodal bitstream model and its config, EMA
    applied.
    """
    import torch

    from evaluation.utils import load_checkpoint, load_config, unwrap_all
    from models import create_model
    from utils.ema import EMA

    verify_checkpoint_identity(checkpoint_path)
    cfg = load_config(str(config_path))
    representation = str(getattr(cfg.data, "representation", "")).lower()
    if representation != "binary":
        raise ValueError(
            f"{context} requires the binary 18-bit patch config "
            f"(cfg.data.representation == 'binary'); got "
            f"{representation!r}"
        )
    # The command-line checkpoint is authoritative.  Several historical
    # configs point at a generic run root, which previously also made
    # entropic table discovery look in the wrong directory.
    cfg.evaluation.checkpoint_path = str(checkpoint_path)

    if num_steps is not None:
        cfg.evaluation.num_sampling_steps = int(num_steps)

    resolved = torch.device(device)
    model = create_model(cfg).to(resolved)
    ema = EMA(unwrap_all(model), decay=0.0)
    checkpoint_metadata = load_checkpoint(
        model, ema, Path(str(checkpoint_path)), resolved, apply_ema=True
    )
    model.eval()
    if return_checkpoint_metadata:
        return model, cfg, checkpoint_metadata
    return model, cfg


class EvoDiffReferenceStore:
    """Read natural references directly from the frozen EvoDiff archive."""

    def __init__(self, root: Path):
        self.root = root
        with (root / "frozen_manifest.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.manifest = json.load(handle)
        data_dir = Path(self.manifest["dataset"]["data_dir"])
        self.data_dir = data_dir if data_dir.is_absolute() else root / data_dir
        metadata = np.load(
            self.data_dir / "lengths_and_offsets.npz", allow_pickle=False
        )
        self.lengths = metadata["ells"]
        self.offsets = metadata["seq_offsets"]
        with (self.data_dir / "splits.json").open(
            "r", encoding="utf-8"
        ) as handle:
            split_data = json.load(handle)
        self.rtest_indices = np.asarray(split_data["rtest"], dtype=np.int64)

    def _read(self, handle, source_index: int, length: int) -> str:
        handle.seek(int(self.offsets[source_index]))
        sequence = handle.readline().rstrip(b"\r\n").decode("ascii")
        if len(sequence) < length:
            raise ValueError(
                "Selected reference is shorter than requested length"
            )
        return sequence[:length]

    def length_matched(
        self,
        target_lengths: Sequence[int],
        *,
        split: str = "rtest",
        seed: int = 0,
    ) -> list[str]:
        if split != "rtest":
            raise ValueError(
                "Publication evaluation must use the held-out rtest split"
            )
        indices = self.rtest_indices
        lengths = np.asarray(self.lengths[indices], dtype=np.int64)
        rng = np.random.default_rng(seed)
        output: list[str | None] = [None] * len(target_lengths)
        with (self.data_dir / "consensus.fasta").open("rb") as handle:
            for target in sorted(set(int(x) for x in target_lengths)):
                positions = np.flatnonzero(
                    np.asarray(target_lengths) == target
                )
                exact = indices[lengths == target]
                candidates = (
                    exact
                    if len(exact) >= len(positions)
                    else indices[lengths >= target]
                )
                if len(candidates) == 0:
                    raise ValueError(
                        f"No rtest reference is at least length {target}"
                    )
                selected = rng.choice(
                    candidates,
                    size=len(positions),
                    replace=len(candidates) < len(positions),
                )
                for position, source_index in zip(
                    positions, selected, strict=True
                ):
                    output[int(position)] = self._read(
                        handle, int(source_index), target
                    )
        return [sequence for sequence in output if sequence is not None]
