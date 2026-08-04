from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def load_binary_protein_model(
    config_path,
    checkpoint_path,
    device,
    *,
    num_steps=None,
    context: str = "this evaluation",
):
    """Load an 18-bit multimodal bitstream model and its config, EMA applied.

    This is the single loader shared by the forward-folding, inverse-folding,
    co-generation, and motif-scaffolding drivers. The training-stack imports (the
    model factory, the checkpoint loader, and the EMA container) are deferred so
    importing this module needs only numpy and torch. The config must select the
    binary 18-bit patch representation; ``num_steps`` overrides the sampling step
    count when given. Returns the evaluated ``(model, cfg)`` pair.
    """
    import torch

    from evaluation.utils import load_checkpoint, load_config, unwrap_all
    from models import create_model
    from utils.ema import EMA

    cfg = load_config(str(config_path))
    representation = str(getattr(cfg.data, "representation", "")).lower()
    if representation != "binary":
        raise ValueError(
            f"{context} requires the binary 18-bit patch config "
            f"(cfg.data.representation == 'binary'); got {representation!r}"
        )
    if num_steps is not None:
        cfg.evaluation.num_sampling_steps = int(num_steps)

    resolved = torch.device(device)
    model = create_model(cfg).to(resolved)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(
        model, ema, Path(str(checkpoint_path)), resolved, apply_ema=True
    )
    model.eval()
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
                for position, source_index in zip(positions, selected):
                    output[int(position)] = self._read(
                        handle, int(source_index), target
                    )
        return [sequence for sequence in output if sequence is not None]
