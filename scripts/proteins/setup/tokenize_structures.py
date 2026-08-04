#!/usr/bin/env python3
"""Offline frozen-encoder cache builder for paired sequence and structure shards.

This tool implements plan section 7.8. It reads a raw backbone-coordinate corpus
(the M1 and M2 datasets), passes every chain through the frozen DPLM-2 LFQ
structure tokenizer to obtain the compact ``uint16`` structure token per residue,
and writes the paired shards that ``data.protein_multimodal`` consumes at
training time. The heavy tokenizer runs only here, in the separate
DPLM-inference environment; the training GPUs never load it.

Two persistence layers make the build cheap to resume and impossible to run
twice on the same chain:

  1. A per-chain token cache keyed by the coordinate hash. Once a chain has been
     encoded, its ``uint16`` tokens live in ``out_dir/token_cache/<hash>.npy``
     and are reused verbatim, never re-encoded, on any later run.
  2. A ``progress.json`` that records which coordinate hashes have already been
     flushed into a shard and the next shard index, so an interrupted build
     continues exactly where it stopped when re-run with ``--resume``.

The frozen tokenizer is loaded lazily through
``evaluation.proteins.dplm_struct_tokenizer.load_struct_tokenizer`` (wrapped by
``data.protein_structure_codec.DPLMStructureTokenizer``). That import only
happens the first time a chain actually needs encoding, so this module, and its
``--help`` output, import cleanly with only numpy and torch present. If every
chain is already cached, the tokenizer is never loaded at all.

Expected coordinate corpus layout under ``--coords-dir`` (either form, mixed):

  Per-chain ``.npz`` (one chain per file)
    coords     float [L, 4, 3]  backbone N, CA, C, O (or [L, 12], reshaped)
    seq_ids    int   [L]        canonical-20 amino-acid ids (optional)
       or aatype [L] int, or sequence/seq a length-L amino-acid string
    seq_mask   bool  [L]        residue has a valid amino-acid identity (optional)
    struct_mask or coord_mask bool [L] residue has valid backbone density (optional)
    scalar metadata keys (optional): stable_id, source, split, cluster_id,
       accession, chain, release, plddt, resolution

  Ragged ``.npz`` (many chains per file, has an ``offsets`` array)
    coords     float [T, 4, 3]  concatenated backbone atoms for all chains
    offsets    int   [N + 1]    row boundaries into the flat arrays
    seq_ids/aatype int [T], struct_mask/coord_mask bool [T] (optional)
    a sibling ``<stem>.meta.json`` of ``{"rows": [{...}, ...]}`` supplies the
       per-chain metadata described above (optional)

The source label (used for the training-time source-balanced sampler) defaults
to the immediate sub-directory of ``--coords-dir``, so laying the corpus out as
``coords_dir/m1/...`` and ``coords_dir/m2/...`` tags each chain automatically;
``--source`` overrides it for every chain.

The dataset manifest written by this tool pins both the LFQ bit convention and
the tokenizer revision, so any drift in the frozen encoder is detectable later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.protein_multimodal import (  # noqa: E402  (repo root added above)
    CANONICAL_AA,
    RowRecord,
    write_manifest,
    write_shard,
)
from data.protein_structure_codec import (  # noqa: E402
    DEFAULT_STRUCT_CODEC,
    DPLM_STRUCT_TOKENIZER_HF_REPO,
    STRUCT_CODEBOOK_SIZE,
    DPLMStructureTokenizer,
    TokenizerProvenance,
)

LOG_PREFIX = "[tokenize_structures]"
BACKBONE_ATOMS = 4  # N, CA, C, O
NONCANONICAL = 255  # sentinel for an amino-acid code outside the canonical 20

# Files under the tokenizer path that identify its revision in the manifest.
_TOKENIZER_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".safetensors")
_TOKENIZER_CONFIG_NAMES = ("config.json", "config.yaml", "config.yml")


def _canonical_aa_lookup() -> np.ndarray:
    """Return a 256-entry table mapping ASCII codes to canonical-20 ids."""
    table = np.full(256, NONCANONICAL, dtype=np.uint8)
    for i, aa in enumerate(CANONICAL_AA):
        table[ord(aa)] = i
    return table


_AA_LUT = _canonical_aa_lookup()


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON to ``path`` atomically via a temporary file and rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Write a numpy array to ``path`` atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def _load_json(path: Path) -> Optional[dict]:
    """Return the parsed JSON at ``path`` or None when it does not exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(value: object) -> object:
    """Unwrap a 0-d numpy array or numpy scalar into a plain Python value."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return value
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coord_hash(coords: np.ndarray) -> str:
    """Stable content hash of one chain's backbone coordinates (the cache key)."""
    contiguous = np.ascontiguousarray(coords, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _seq_hash(seq_ids: np.ndarray) -> str:
    """Short content hash of a chain's amino-acid ids, for the cache index."""
    return hashlib.sha256(
        np.asarray(seq_ids, dtype=np.uint8).tobytes()
    ).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Coordinate corpus reader
# -----------------------------------------------------------------------------


class ChainInput:
    """One raw backbone chain read from the coordinate corpus, before encoding."""

    def __init__(
        self,
        *,
        stable_id: str,
        coords: np.ndarray,
        seq_ids: np.ndarray,
        seq_mask: np.ndarray,
        struct_mask: np.ndarray,
        source: str,
        split: str,
        cluster_id: str = "",
        accession: str = "",
        chain: str = "",
        release: str = "",
        plddt: float = float("nan"),
        resolution: float = float("nan"),
    ) -> None:
        self.stable_id = stable_id
        self.coords = coords
        self.seq_ids = seq_ids
        self.seq_mask = seq_mask
        self.struct_mask = struct_mask
        self.source = source
        self.split = split
        self.cluster_id = cluster_id
        self.accession = accession
        self.chain = chain
        self.release = release
        self.plddt = plddt
        self.resolution = resolution

    @property
    def length(self) -> int:
        return int(self.coords.shape[0])


def _as_backbone(array: object) -> np.ndarray:
    """Coerce a coordinate array to ``[L, 4, 3]`` float32 (N, CA, C, O per residue)."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (BACKBONE_ATOMS, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == BACKBONE_ATOMS * 3:
        return arr.reshape(-1, BACKBONE_ATOMS, 3)
    raise ValueError(
        f"coords must have shape [L, {BACKBONE_ATOMS}, 3] or [L, {BACKBONE_ATOMS * 3}]; "
        f"got {arr.shape}"
    )


def _sequence_ids_from_string(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Map an amino-acid string to canonical-20 ids and a validity mask."""
    codes = np.frombuffer(str(seq).encode("ascii", "replace"), dtype=np.uint8)
    mapped = _AA_LUT[codes]
    mask = mapped != NONCANONICAL
    ids = np.where(mask, mapped, 0).astype(np.uint8)
    return ids, mask


def _sequence_ids_from_ints(array: object) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp integer amino-acid ids to the canonical-20 range and mask the rest."""
    values = np.asarray(array).astype(np.int64).reshape(-1)
    mask = (values >= 0) & (values < len(CANONICAL_AA))
    ids = np.where(mask, values, 0).astype(np.uint8)
    return ids, mask


def _resolve_sequence(
    npz: Dict[str, object],
    row_meta: Optional[dict],
    length: int,
    row_slice: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve per-residue amino-acid ids and mask for one chain.

    Prefers integer ids (``seq_ids`` then ``aatype``); falls back to a
    per-chain amino-acid string (``sequence`` or ``seq``); otherwise returns a
    structure-only chain with all-zero ids and an all-false sequence mask.
    """
    for key in ("seq_ids", "aatype"):
        if key in npz:
            values = npz[key]
            if row_slice is not None:
                values = np.asarray(values)[row_slice[0] : row_slice[1]]
            ids, mask = _sequence_ids_from_ints(values)
            if ids.shape[0] != length:
                raise ValueError(
                    f"{key} length {ids.shape[0]} does not match chain length {length}"
                )
            return ids, mask
    seq_str: Optional[str] = None
    if row_meta is not None:
        seq_str = row_meta.get("sequence") or row_meta.get("seq")
    if seq_str is None:
        for key in ("sequence", "seq"):
            if key in npz and row_slice is None:
                seq_str = _scalar(npz[key])
                break
    if isinstance(seq_str, str) and seq_str:
        ids, mask = _sequence_ids_from_string(seq_str)
        if ids.shape[0] != length:
            raise ValueError(
                f"sequence length {ids.shape[0]} does not match chain length {length}"
            )
        return ids, mask
    return np.zeros(length, dtype=np.uint8), np.zeros(length, dtype=bool)


def _resolve_struct_mask(
    npz: Dict[str, object],
    coords: np.ndarray,
    row_slice: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Mark residues with fully finite backbone density, intersected with any
    provided ``struct_mask`` or ``coord_mask``."""
    finite = np.isfinite(coords).all(axis=(1, 2))
    for key in ("struct_mask", "coord_mask"):
        if key in npz:
            provided = np.asarray(npz[key])
            if row_slice is not None:
                provided = provided[row_slice[0] : row_slice[1]]
            provided = provided.reshape(-1).astype(bool)
            if provided.shape[0] == finite.shape[0]:
                finite = finite & provided
            break
    return finite.astype(bool)


def _meta_value(
    npz: Dict[str, object],
    rows_meta: Optional[List[dict]],
    row_idx: int,
    key: str,
    default: object,
) -> object:
    """Look up per-chain metadata from a ragged npz array, sidecar rows, or default."""
    if key in npz:
        arr = np.asarray(npz[key])
        if arr.ndim >= 1 and arr.shape[0] > row_idx:
            return _scalar(arr[row_idx])
    if rows_meta is not None and row_idx < len(rows_meta):
        value = rows_meta[row_idx].get(key)
        if value is not None:
            return value
    return default


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_sidecar(path: Path) -> Optional[dict]:
    """Load a ``<stem>.meta.json`` or ``<stem>.json`` metadata sidecar for a file."""
    for candidate in (
        path.with_suffix(".meta.json"),
        path.with_suffix(".json"),
    ):
        payload = _load_json(candidate)
        if payload is not None:
            return payload
    return None


def _chains_from_npz(
    path: Path, default_source: str, source_override: Optional[str]
) -> Iterator[ChainInput]:
    """Yield every chain stored in one ``.npz`` file (per-chain or ragged)."""
    with np.load(path, allow_pickle=True) as loaded:
        npz = {key: loaded[key] for key in loaded.files}
    sidecar = _load_sidecar(path)

    if "coords" not in npz:
        raise ValueError(f"{path}: missing required 'coords' array")

    if "offsets" in npz:
        coords_flat = _as_backbone(npz["coords"])
        offsets = np.asarray(npz["offsets"]).astype(np.int64).reshape(-1)
        rows_meta = sidecar.get("rows") if isinstance(sidecar, dict) else None
        n_rows = int(offsets.shape[0] - 1)
        for row_idx in range(n_rows):
            start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
            coords = coords_flat[start:end]
            length = int(coords.shape[0])
            row_meta = (
                rows_meta[row_idx]
                if (rows_meta is not None and row_idx < len(rows_meta))
                else None
            )
            seq_ids, seq_mask = _resolve_sequence(
                npz, row_meta, length, row_slice=(start, end)
            )
            struct_mask = _resolve_struct_mask(
                npz, coords, row_slice=(start, end)
            )
            source = source_override or str(
                _meta_value(npz, rows_meta, row_idx, "source", default_source)
            )
            yield ChainInput(
                stable_id=str(
                    _meta_value(
                        npz,
                        rows_meta,
                        row_idx,
                        "stable_id",
                        f"{path.stem}:{row_idx}",
                    )
                ),
                coords=coords,
                seq_ids=seq_ids,
                seq_mask=seq_mask,
                struct_mask=struct_mask,
                source=source,
                split=str(
                    _meta_value(npz, rows_meta, row_idx, "split", "train")
                ),
                cluster_id=str(
                    _meta_value(npz, rows_meta, row_idx, "cluster_id", "")
                ),
                accession=str(
                    _meta_value(npz, rows_meta, row_idx, "accession", "")
                ),
                chain=str(_meta_value(npz, rows_meta, row_idx, "chain", "")),
                release=str(
                    _meta_value(npz, rows_meta, row_idx, "release", "")
                ),
                plddt=_float_or_nan(
                    _meta_value(npz, rows_meta, row_idx, "plddt", float("nan"))
                ),
                resolution=_float_or_nan(
                    _meta_value(
                        npz, rows_meta, row_idx, "resolution", float("nan")
                    )
                ),
            )
        return

    coords = _as_backbone(npz["coords"])
    length = int(coords.shape[0])
    meta = sidecar if isinstance(sidecar, dict) else {}
    seq_ids, seq_mask = _resolve_sequence(npz, meta, length)
    struct_mask = _resolve_struct_mask(npz, coords)

    def pick(key: str, default: object) -> object:
        if key in npz:
            return _scalar(npz[key])
        if key in meta:
            return meta[key]
        return default

    source = source_override or str(pick("source", default_source))
    yield ChainInput(
        stable_id=str(pick("stable_id", path.stem)),
        coords=coords,
        seq_ids=seq_ids,
        seq_mask=seq_mask,
        struct_mask=struct_mask,
        source=source,
        split=str(pick("split", "train")),
        cluster_id=str(pick("cluster_id", "")),
        accession=str(pick("accession", "")),
        chain=str(pick("chain", "")),
        release=str(pick("release", "")),
        plddt=_float_or_nan(pick("plddt", float("nan"))),
        resolution=_float_or_nan(pick("resolution", float("nan"))),
    )


def iter_coordinate_chains(
    coords_dir: Path, source_override: Optional[str]
) -> Iterator[ChainInput]:
    """Iterate over every chain in the coordinate corpus in a deterministic order.

    Files are visited in sorted path order so shard assignment is reproducible.
    The default source label for a file is the first path component beneath
    ``coords_dir`` (the ``m1``/``m2`` sub-directory) or the corpus directory
    name when the file sits directly in ``coords_dir``.
    """
    files = sorted(coords_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz coordinate files found under {coords_dir}. Each file must "
            "contain a 'coords' array of shape [L, 4, 3] (backbone N, CA, C, O), "
            "optionally with 'offsets' for a ragged multi-chain shard. See the "
            "module docstring for the full layout."
        )
    for path in files:
        relative = path.relative_to(coords_dir)
        default_source = (
            relative.parts[0] if len(relative.parts) > 1 else coords_dir.name
        )
        yield from _chains_from_npz(path, default_source, source_override)


# -----------------------------------------------------------------------------
# Tokenizer identity (recorded in the manifest without loading the heavy model)
# -----------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_files(tokenizer_path: Path) -> List[Path]:
    """Return the identifying weight and config files under the tokenizer path."""
    if tokenizer_path.is_file():
        return [tokenizer_path]
    if not tokenizer_path.is_dir():
        return []
    files: List[Path] = []
    for candidate in sorted(tokenizer_path.rglob("*")):
        if not candidate.is_file():
            continue
        if (
            candidate.suffix in _TOKENIZER_WEIGHT_SUFFIXES
            or candidate.name in _TOKENIZER_CONFIG_NAMES
        ):
            files.append(candidate)
    return files


def resolve_tokenizer_identity(
    tokenizer_path: Optional[Path], explicit_revision: Optional[str]
) -> Tuple[str, Dict[str, object]]:
    """Determine the tokenizer revision string and per-file provenance hashes.

    When ``--tokenizer-revision`` is given it is trusted and only cheap file
    sizes are recorded. Otherwise the revision is derived from the content
    hashes of the tokenizer's weight and config files, so a silent checkpoint
    swap changes the recorded revision. This never imports the DPLM stack.
    """
    if tokenizer_path is None or not tokenizer_path.exists():
        return (explicit_revision or "unpinned"), {}
    files = _tokenizer_files(tokenizer_path)
    file_hashes: Dict[str, object] = {}
    if explicit_revision:
        for path in files:
            file_hashes[path.name] = {"bytes": path.stat().st_size}
        return explicit_revision, file_hashes
    combined = hashlib.sha256()
    for path in files:
        content = _sha256_file(path)
        file_hashes[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": content,
        }
        combined.update(path.name.encode("utf-8"))
        combined.update(content.encode("utf-8"))
    if not files:
        return "unpinned", {}
    return "sha256:" + combined.hexdigest()[:16], file_hashes


# -----------------------------------------------------------------------------
# Encoding and shard flushing
# -----------------------------------------------------------------------------


class _TokenizerHandle:
    """Holds the frozen tokenizer and loads it lazily on first encode."""

    def __init__(
        self,
        model_path: Optional[Path],
        provenance: TokenizerProvenance,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.provenance = provenance
        self.device = device
        self._tokenizer: Optional[DPLMStructureTokenizer] = None

    def encode(self, coords: np.ndarray) -> np.ndarray:
        """Encode one chain's backbone coordinates to ``uint16`` structure ids."""
        if self._tokenizer is None:
            if self.model_path is None or not Path(self.model_path).exists():
                raise RuntimeError(
                    "Encoding requires the frozen DPLM-2 structure tokenizer, but "
                    f"--tokenizer-path {self.model_path!r} was not found. Download it and "
                    "install the DPLM-inference environment via "
                    "scripts/proteins/setup/setup_evaluation.sh dplm "
                    "(the dplm-inference extra (uv sync --extra dplm-inference)), then re-run. The tokenizer "
                    f"corresponds to the Hugging Face repo {DPLM_STRUCT_TOKENIZER_HF_REPO}."
                )
            self._tokenizer = DPLMStructureTokenizer(
                model_path=self.model_path,
                provenance=self.provenance,
                device=self.device,
            )
        tokens = np.asarray(
            self._tokenizer.encode(coords), dtype=np.uint16
        ).reshape(-1)
        return tokens


def _build_row(
    chain: ChainInput, tokens: np.ndarray, revision: str
) -> RowRecord:
    """Assemble a :class:`RowRecord` from one chain and its structure tokens."""
    length = chain.length
    if tokens.shape[0] != length:
        raise ValueError(
            f"chain {chain.stable_id}: tokenizer returned {tokens.shape[0]} tokens for "
            f"{length} residues"
        )
    if tokens.size and int(tokens.max()) >= STRUCT_CODEBOOK_SIZE:
        raise ValueError(
            f"chain {chain.stable_id}: structure token id {int(tokens.max())} is out of "
            f"range [0, {STRUCT_CODEBOOK_SIZE - 1}]"
        )
    return RowRecord(
        stable_id=chain.stable_id,
        seq_ids=np.asarray(chain.seq_ids, dtype=np.uint8),
        struct_index=tokens,
        seq_mask=np.asarray(chain.seq_mask, dtype=bool),
        struct_mask=np.asarray(chain.struct_mask, dtype=bool),
        source=chain.source,
        cluster_id=chain.cluster_id,
        split=chain.split,
        accession=chain.accession,
        chain=chain.chain,
        release=chain.release,
        plddt=chain.plddt,
        resolution=chain.resolution,
    )


def _flush_shard(
    out_dir: Path,
    progress: dict,
    rows: List[RowRecord],
    hashes: List[str],
    cache_entries: dict,
    revision: str,
) -> None:
    """Write the buffered rows as one shard and checkpoint resume state."""
    if not rows:
        return
    shard_name = f"shard_{progress['next_shard_index']:05d}"
    write_shard(out_dir, shard_name, rows, tokenizer_revision=revision)
    progress["shards"].append(shard_name)
    progress["emitted_hashes"].extend(hashes)
    progress["next_shard_index"] += 1
    _atomic_write_json(out_dir / "progress.json", progress)
    _atomic_write_json(
        out_dir / "cache_index.json",
        {"tokenizer_revision": revision, "entries": cache_entries},
    )
    print(
        f"{LOG_PREFIX} wrote {shard_name} with {len(rows)} chains", flush=True
    )
    rows.clear()
    hashes.clear()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coords-dir",
        type=Path,
        required=True,
        help="Directory of raw backbone-coordinate .npz files (the M1/M2 corpus).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Local path to the frozen DPLM-2 LFQ structure tokenizer checkpoint. "
        "Only needed when chains still have to be encoded.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the token cache, paired shards, and manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the tokenizer (for example cpu or cuda:0).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Number of chains per output shard and per resume checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted build in --out-dir instead of starting fresh.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help="Explicit tokenizer revision to record; derived from file hashes if omitted.",
    )
    parser.add_argument(
        "--tokenizer-hf-repo",
        type=str,
        default=DPLM_STRUCT_TOKENIZER_HF_REPO,
        help="Hugging Face repo id recorded in the tokenizer provenance.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override the source label for every chain (default: per-file sub-directory).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Skip chains shorter than this many residues.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="Skip chains longer than this many residues (0 disables the cap).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after processing this many chains (0 disables; for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    coords_dir = args.coords_dir
    if not coords_dir.exists():
        raise FileNotFoundError(f"--coords-dir {coords_dir} does not exist")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    token_cache_dir = out_dir / "token_cache"
    token_cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    existing_progress = _load_json(progress_path)
    if existing_progress is not None and not args.resume:
        print(
            f"{LOG_PREFIX} {progress_path} already exists from a previous run. "
            "Pass --resume to continue it, or choose a new --out-dir.",
            file=sys.stderr,
        )
        return 2
    progress = existing_progress or {
        "shards": [],
        "emitted_hashes": [],
        "next_shard_index": 0,
    }
    emitted = set(progress["emitted_hashes"])

    revision, file_hashes = resolve_tokenizer_identity(
        args.tokenizer_path, args.tokenizer_revision
    )
    provenance = TokenizerProvenance(
        hf_repo=args.tokenizer_hf_repo,
        hf_revision=revision,
        file_hashes=file_hashes,
    )
    tokenizer = _TokenizerHandle(args.tokenizer_path, provenance, args.device)
    print(f"{LOG_PREFIX} tokenizer revision: {revision}", flush=True)

    cache_entries = dict((existing_progress or {}).get("cache_entries", {}))
    prior_index = _load_json(out_dir / "cache_index.json")
    if isinstance(prior_index, dict):
        cache_entries.update(prior_index.get("entries", {}))

    buffer_rows: List[RowRecord] = []
    buffer_hashes: List[str] = []
    source_counts: Dict[str, int] = {}
    n_seen = n_encoded = n_cache_hit = n_already_emitted = n_skipped_len = 0

    for chain in iter_coordinate_chains(coords_dir, args.source):
        length = chain.length
        if length < max(1, args.min_len) or (
            args.max_len > 0 and length > args.max_len
        ):
            n_skipped_len += 1
            continue
        n_seen += 1
        chash = _coord_hash(chain.coords)

        if chash in emitted:
            n_already_emitted += 1
            continue

        token_file = token_cache_dir / f"{chash}.npy"
        if token_file.exists():
            tokens = np.asarray(np.load(token_file), dtype=np.uint16).reshape(
                -1
            )
            n_cache_hit += 1
        else:
            tokens = tokenizer.encode(
                np.ascontiguousarray(chain.coords, dtype=np.float32)
            )
            if tokens.shape[0] != length:
                raise ValueError(
                    f"chain {chain.stable_id}: tokenizer returned {tokens.shape[0]} tokens for "
                    f"{length} residues"
                )
            _atomic_write_npy(token_file, tokens)
            n_encoded += 1

        row = _build_row(chain, tokens, revision)
        buffer_rows.append(row)
        buffer_hashes.append(chash)
        emitted.add(chash)
        source_counts[chain.source] = source_counts.get(chain.source, 0) + 1
        cache_entries[chash] = {
            "stable_id": chain.stable_id,
            "length": length,
            "source": chain.source,
            "split": chain.split,
            "sequence_hash": _seq_hash(chain.seq_ids),
            "tokenizer_revision": revision,
        }

        if len(buffer_rows) >= max(1, args.batch):
            _flush_shard(
                out_dir,
                progress,
                buffer_rows,
                buffer_hashes,
                cache_entries,
                revision,
            )

        if args.limit and n_seen >= args.limit:
            break

    _flush_shard(
        out_dir, progress, buffer_rows, buffer_hashes, cache_entries, revision
    )

    manifest_extra = {
        "tokenizer": provenance.to_dict(),
        "tokenizer_revision": revision,
        "tokenizer_path": str(args.tokenizer_path)
        if args.tokenizer_path
        else None,
        "device": args.device,
        "coords_dir": str(coords_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_chains": len(progress["emitted_hashes"]),
        "source_counts": source_counts,
        "counts": {
            "seen_this_run": n_seen,
            "encoded_this_run": n_encoded,
            "cache_hits_this_run": n_cache_hit,
            "already_emitted": n_already_emitted,
            "skipped_by_length": n_skipped_len,
        },
    }
    manifest_path = write_manifest(
        out_dir,
        progress["shards"],
        codec=DEFAULT_STRUCT_CODEC,
        extra=manifest_extra,
    )
    _atomic_write_json(
        out_dir / "cache_index.json",
        {"tokenizer_revision": revision, "entries": cache_entries},
    )

    print(
        f"{LOG_PREFIX} done. chains total={len(progress['emitted_hashes'])} "
        f"encoded={n_encoded} cache_hits={n_cache_hit} already_emitted={n_already_emitted} "
        f"shards={len(progress['shards'])}",
        flush=True,
    )
    print(f"{LOG_PREFIX} manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
