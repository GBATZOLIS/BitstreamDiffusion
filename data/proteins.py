from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


# -----------------------------------------------------------------------------
# Char amino-acid vocabulary (offline fallback tokenizer)
# -----------------------------------------------------------------------------
# 4 specials + 20 canonical residues + 5 ambiguity/rare symbols = 29 tokens.
PROTEIN_TOKENS: List[str] = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "B", "Z", "X", "U", "O",
]
TOKEN_TO_ID: Dict[str, int] = {tok: i for i, tok in enumerate(PROTEIN_TOKENS)}
ID_TO_TOKEN: Dict[int, str] = {i: tok for i, tok in enumerate(PROTEIN_TOKENS)}

PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
UNK_ID = TOKEN_TO_ID["<unk>"]

SPECIAL_IDS = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
VOCAB_SIZE = len(PROTEIN_TOKENS)

RESIDUE_CHARS: List[str] = [t for t in PROTEIN_TOKENS if len(t) == 1]
FALLBACK_RESIDUE_ID = TOKEN_TO_ID["X"]

DEFAULT_ESM2_MODEL = "nvidia/esm2_t6_8M_UR50D"


# -----------------------------------------------------------------------------
# Tokenizer abstraction
# -----------------------------------------------------------------------------

class ProteinTokenizer:
    """
    Maps residues <-> integer token ids for the bitstream path. Two backends:
      - "char": self-contained 29-token amino-acid vocabulary (5 bits).
      - "esm2": ESM-2 amino-acid tokenizer from the nvidia/bionemo HF collection
                (33 tokens, 6 bits). cls=<bos>, eos, pad, unk, mask.
    Only the id space matters here; the encoder framing is BOS + residues + EOS.
    """

    def __init__(
        self,
        *,
        name: str,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        unk_id: int,
        char_to_id: np.ndarray,
        id_to_char: Dict[int, str],
        residue_ids: Set[int],
        special_ids: Set[int],
    ):
        self.name = str(name)
        self.vocab_size = int(vocab_size)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)
        self.unk_id = int(unk_id)
        self.char_to_id = char_to_id  # [256] int64, unknown residues -> X
        self.id_to_char = dict(id_to_char)  # residue ids only
        self.residue_ids = set(int(i) for i in residue_ids)
        self.special_ids = set(int(i) for i in special_ids)

    def encode_residues(self, seq: str) -> np.ndarray:
        b = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        return self.char_to_id[b]


def make_char_tokenizer() -> ProteinTokenizer:
    char_to_id = np.full(256, FALLBACK_RESIDUE_ID, dtype=np.int64)
    id_to_char: Dict[int, str] = {}
    residue_ids: Set[int] = set()
    for c in RESIDUE_CHARS:
        tid = TOKEN_TO_ID[c]
        char_to_id[ord(c)] = tid
        char_to_id[ord(c.lower())] = tid
        id_to_char[tid] = c
        residue_ids.add(tid)
    return ProteinTokenizer(
        name="char",
        vocab_size=VOCAB_SIZE,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        residue_ids=residue_ids,
        special_ids=set(SPECIAL_IDS),
    )


def make_esm2_tokenizer(model_id: str = DEFAULT_ESM2_MODEL) -> ProteinTokenizer:
    from transformers import AutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(
            f"[proteins] could not load ESM-2 tokenizer from {model_id!r} ({e}); "
            f"falling back to facebook/esm2_t6_8M_UR50D"
        )
        hf = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    vocab = hf.get_vocab()  # token -> id
    vocab_size = max(int(i) for i in vocab.values()) + 1

    char_to_id = np.full(256, -1, dtype=np.int64)
    id_to_char: Dict[int, str] = {}
    residue_ids: Set[int] = set()
    for token, tid in vocab.items():
        if len(token) == 1 and "A" <= token <= "Z":
            char_to_id[ord(token)] = int(tid)
            char_to_id[ord(token.lower())] = int(tid)
            id_to_char[int(tid)] = token
            residue_ids.add(int(tid))

    x_id = int(char_to_id[ord("X")])
    if x_id < 0:
        x_id = int(hf.unk_token_id)
    char_to_id[char_to_id < 0] = x_id

    return ProteinTokenizer(
        name="esm2",
        vocab_size=vocab_size,
        bos_id=int(hf.cls_token_id),
        eos_id=int(hf.eos_token_id),
        pad_id=int(hf.pad_token_id),
        unk_id=int(hf.unk_token_id),
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        residue_ids=residue_ids,
        special_ids=set(int(i) for i in hf.all_special_ids),
    )


def tokenizer_tag_from_config(config: config_dict.ConfigDict) -> str:
    name = str(getattr(config.data, "tokenizer", "esm2")).lower().strip()
    if name in {"esm", "esm2", "bionemo"}:
        return "esm2"
    if name in {"char", "aa", "chars"}:
        return "char"
    raise ValueError(f"Unknown cfg.data.tokenizer={name!r}. Use 'esm2' or 'char'.")


def make_protein_tokenizer(config: config_dict.ConfigDict) -> ProteinTokenizer:
    tag = tokenizer_tag_from_config(config)
    if tag == "char":
        return make_char_tokenizer()
    model_id = str(getattr(config.data, "esm2_model", DEFAULT_ESM2_MODEL))
    return make_esm2_tokenizer(model_id)


# -----------------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------------

def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_rank_world() -> Tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _is_rank0() -> bool:
    rank, _ = _ddp_rank_world()
    return rank == 0


def _dist_barrier() -> None:
    if _ddp_is_on():
        dist.barrier()


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


# -----------------------------------------------------------------------------
# Token <-> bits
# -----------------------------------------------------------------------------

def build_token_to_bits_table(vocab_size: int, bits_per_token: int) -> torch.Tensor:
    # Big-endian raw-binary code for each token id. Row i = bits of id i.
    table = torch.zeros((int(vocab_size), int(bits_per_token)), dtype=torch.long)
    for tid in range(int(vocab_size)):
        code = int(tid)
        for k in range(int(bits_per_token)):
            table[tid, bits_per_token - 1 - k] = (code >> k) & 1
    return table


def bitstreams_to_token_ids(bits: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    # bits: [S] or [B, S] in {0,1} (long or float). Returns [B, T] token ids.
    if bits.dim() == 1:
        bits = bits.unsqueeze(0)
    if bits.is_floating_point():
        bits = (bits > 0.5).long()
    else:
        bits = (bits != 0).long()

    B = bits.size(0)
    m = int(bits_per_token)
    T = bits.size(1) // m
    data_bits = bits[:, : T * m].reshape(B, T, m)
    powers = 2 ** torch.arange(m - 1, -1, -1, device=bits.device)
    return (data_bits * powers).sum(dim=-1).long()


def token_ids_to_sequences(
    token_ids: torch.Tensor,
    tokenizer: ProteinTokenizer,
    *,
    stop_at_eos: bool = True,
    drop_leading_bos: bool = True,
) -> List[str]:
    # Map token ids to amino-acid strings. Specials and invalid codes contribute
    # no residue. Reading stops at the first EOS when stop_at_eos is True.
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    ids = token_ids.long().cpu().tolist()

    out: List[str] = []
    for row in ids:
        start = 1 if (drop_leading_bos and len(row) > 0 and row[0] == tokenizer.bos_id) else 0
        chars: List[str] = []
        for tid in row[start:]:
            if stop_at_eos and tid == tokenizer.eos_id:
                break
            c = tokenizer.id_to_char.get(tid)
            if c is not None:
                chars.append(c)
        out.append("".join(chars))
    return out


def decode_bitstreams_to_sequences(
    bits: torch.Tensor,
    bits_per_token: int,
    tokenizer: ProteinTokenizer,
    *,
    stop_at_eos: bool = True,
    drop_leading_bos: bool = True,
) -> List[str]:
    # Convenience: generated bitstreams -> amino-acid strings.
    token_ids = bitstreams_to_token_ids(bits, bits_per_token)
    return token_ids_to_sequences(
        token_ids, tokenizer, stop_at_eos=stop_at_eos, drop_leading_bos=drop_leading_bos
    )


# -----------------------------------------------------------------------------
# FASTA parsing + deterministic split
# -----------------------------------------------------------------------------

def _open_fasta(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def iter_fasta(path: Path):
    # Yield (header_id, sequence) pairs. Sequence lines are concatenated and
    # uppercased; whitespace is dropped.
    header = None
    chunks: List[str] = []
    with _open_fasta(path) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                parts = line[1:].strip().split()
                header = parts[0] if parts else ""
                chunks = []
            else:
                chunks.append(line.strip().upper())
    if header is not None:
        yield header, "".join(chunks)


def split_for_id(header_id: str, val_fraction: float, test_fraction: float) -> str:
    # Deterministic hash split by accession so splits are stable across runs.
    h = hashlib.md5(header_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(0xFFFFFFFF)
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def _sequence_to_windows(seq: str, content_len: int, min_len: int) -> List[str]:
    # Non-overlapping windows so long proteins are used instead of discarded.
    windows: List[str] = []
    for i in range(0, len(seq), content_len):
        w = seq[i : i + content_len]
        if len(w) >= min_len:
            windows.append(w)
    return windows


def _encode_block(window: str, seq_len_tokens: int, tokenizer: ProteinTokenizer) -> np.ndarray:
    # block = [BOS] + residues + [EOS], right-padded with PAD to seq_len_tokens.
    row = np.full((seq_len_tokens,), tokenizer.pad_id, dtype=np.uint8)
    row[0] = tokenizer.bos_id
    ids = tokenizer.encode_residues(window)[: seq_len_tokens - 2]
    n = int(ids.shape[0])
    row[1 : 1 + n] = ids.astype(np.uint8)
    row[1 + n] = tokenizer.eos_id
    return row


# -----------------------------------------------------------------------------
# Token cache (fixed-width uint8 blocks, one file per split, tokenizer-tagged)
# -----------------------------------------------------------------------------

def _cache_paths(
    root: Path, split: str, tok_tag: str, seq_len_tokens: int, min_len: int
) -> Tuple[Path, Path]:
    stem = f"cache_{split}_{tok_tag}_tokens_L{seq_len_tokens}_min{min_len}"
    return root / f"{stem}.uint8", root / f"{stem}.meta.json"


def _raw_fasta_path(root: Path, fasta_name: str) -> Path:
    p = root / "raw" / fasta_name
    if p.exists():
        return p
    if str(p).endswith(".gz"):
        alt = Path(str(p)[:-3])
        if alt.exists():
            return alt
    return p


def _caches_valid(
    root: Path,
    tok_tag: str,
    seq_len_tokens: int,
    min_len: int,
    *,
    fasta_name: str,
    val_fraction: float,
    test_fraction: float,
    max_windows_per_seq: Optional[int],
) -> bool:
    # Reusable only if all three splits exist AND were built with the same
    # cache-affecting params (tokenizer, fasta, fractions, windowing, lengths).
    for split in ("train", "val", "test"):
        cache_path, meta_path = _cache_paths(root, split, tok_tag, seq_len_tokens, min_len)
        if not (cache_path.exists() and meta_path.exists()):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        if (
            meta.get("tokenizer") != tok_tag
            or meta.get("fasta_name") != fasta_name
            or float(meta.get("val_fraction", -1.0)) != float(val_fraction)
            or float(meta.get("test_fraction", -1.0)) != float(test_fraction)
            or meta.get("max_windows_per_seq", "missing") != max_windows_per_seq
            or int(meta.get("seq_len_tokens", -1)) != int(seq_len_tokens)
            or int(meta.get("min_len", -1)) != int(min_len)
        ):
            return False
    return True


def _build_all_caches(
    *,
    root: Path,
    tokenizer: ProteinTokenizer,
    fasta_name: str,
    seq_len_tokens: int,
    min_len: int,
    val_fraction: float,
    test_fraction: float,
    max_windows_per_seq: Optional[int],
) -> None:
    # Single streaming pass over the FASTA writing all three split caches.
    from tqdm.auto import tqdm

    content_len = seq_len_tokens - 2
    if content_len <= 0:
        raise ValueError("sequence_len_tokens must be >= 3")

    fasta_path = _raw_fasta_path(root, fasta_name)
    if not fasta_path.exists():
        raise RuntimeError(
            f"Missing Swiss-Prot FASTA at {fasta_path}. Download it with:\n"
            f"  mkdir -p {root / 'raw'}\n"
            f"  wget -O {root / 'raw' / fasta_name} \\\n"
            f"    https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
            f"knowledgebase/complete/{fasta_name}"
        )

    tok_tag = tokenizer.name
    tmp_handles = {}
    counts = {"train": 0, "val": 0, "test": 0}
    tmp_paths = {
        s: _cache_paths(root, s, tok_tag, seq_len_tokens, min_len)[0].with_suffix(".uint8.tmp")
        for s in ("train", "val", "test")
    }
    for s, p in tmp_paths.items():
        p.unlink(missing_ok=True)
        tmp_handles[s] = open(p, "wb")

    n_proteins = 0
    n_dropped_short = 0
    try:
        for header_id, seq in tqdm(iter_fasta(fasta_path), desc="Parsing FASTA", unit="seq"):
            n_proteins += 1
            if len(seq) < min_len:
                n_dropped_short += 1
                continue
            split = split_for_id(header_id, val_fraction, test_fraction)
            windows = _sequence_to_windows(seq, content_len, min_len)
            if max_windows_per_seq is not None:
                windows = windows[: int(max_windows_per_seq)]
            for w in windows:
                row = _encode_block(w, seq_len_tokens, tokenizer)
                tmp_handles[split].write(row.tobytes())
                counts[split] += 1
    finally:
        for h in tmp_handles.values():
            h.close()

    for split in ("train", "val", "test"):
        cache_path, meta_path = _cache_paths(root, split, tok_tag, seq_len_tokens, min_len)
        if counts[split] <= 0:
            tmp_paths[split].unlink(missing_ok=True)
            raise RuntimeError(
                f"Split {split!r} is empty after parsing (n_proteins={n_proteins}). "
                f"Check the FASTA and split fractions."
            )
        os.replace(tmp_paths[split], cache_path)
        meta = {
            "dataset": "swissprot",
            "split": split,
            "tokenizer": tok_tag,
            "fasta_name": fasta_name,
            "seq_len_tokens": int(seq_len_tokens),
            "min_len": int(min_len),
            "n_sequences": int(counts[split]),
            "vocab_size": int(tokenizer.vocab_size),
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
            "max_windows_per_seq": max_windows_per_seq,
            "n_proteins_total": int(n_proteins),
            "n_dropped_short": int(n_dropped_short),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(
        f"[proteins] built caches ({tok_tag}): "
        f"train={counts['train']:,} val={counts['val']:,} test={counts['test']:,} "
        f"(proteins={n_proteins:,}, dropped_short={n_dropped_short:,}, "
        f"seq_len_tokens={seq_len_tokens}, min_len={min_len})"
    )


def ensure_swissprot_caches_ready(config: config_dict.ConfigDict) -> None:
    # DDP-safe: rank 0 builds all splits, other ranks wait then are told the
    # outcome so a build failure fails fast instead of hanging in the barrier.
    root = Path(getattr(config.data, "root", "datasets/swissprot"))
    tok_tag = tokenizer_tag_from_config(config)
    fasta_name = str(getattr(config.data, "fasta_name", "uniprot_sprot.fasta.gz"))
    seq_len_tokens = int(getattr(config.data, "sequence_len_tokens", 256))
    min_len = int(getattr(config.data, "min_len", 20))
    val_fraction = float(getattr(config.data, "val_fraction", 0.01))
    test_fraction = float(getattr(config.data, "test_fraction", 0.01))
    max_windows = getattr(config.data, "max_windows_per_seq", None)
    max_windows = None if max_windows in (None, 0) else int(max_windows)

    def _valid() -> bool:
        return _caches_valid(
            root,
            tok_tag,
            seq_len_tokens,
            min_len,
            fasta_name=fasta_name,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_windows_per_seq=max_windows,
        )

    if _valid():
        return

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"cache_build_{tok_tag}_L{seq_len_tokens}_min{min_len}.lock"

    def _build_locked():
        with FileLock(str(lock_path)):
            if _valid():
                return
            tokenizer = make_protein_tokenizer(config)
            _build_all_caches(
                root=root,
                tokenizer=tokenizer,
                fasta_name=fasta_name,
                seq_len_tokens=seq_len_tokens,
                min_len=min_len,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                max_windows_per_seq=max_windows,
            )

    if not _ddp_is_on():
        _build_locked()
        return

    # Only rank 0 builds. Catch its failure so it still reaches the barrier, then
    # broadcast success/failure so non-zero ranks fail fast instead of hanging in
    # the barrier until the NCCL watchdog fires.
    err: Optional[Exception] = None
    if _is_rank0():
        try:
            _build_locked()
        except Exception as e:  # noqa: BLE001 - re-raised after sync below
            err = e
            print(f"[proteins] rank0 cache build failed: {e}")

    _dist_barrier()

    ok = 1 if err is None else 0
    flag = torch.tensor([ok], dtype=torch.long)
    if torch.cuda.is_available():
        flag = flag.to(torch.device("cuda", torch.cuda.current_device()))
    dist.broadcast(flag, src=0)

    if int(flag.item()) == 0:
        if err is not None:
            raise err
        raise RuntimeError(
            "Swiss-Prot cache build failed on rank 0; see rank 0 logs for the cause."
        )


def load_token_id_cache(
    root: Path, split: str, tok_tag: str, seq_len_tokens: int, min_len: int
) -> Tuple[np.memmap, dict]:
    # Load a split's fixed-width uint8 token-id cache as a memmap.
    root = Path(root)
    cache_path, meta_path = _cache_paths(root, split, tok_tag, seq_len_tokens, min_len)
    if not (cache_path.exists() and meta_path.exists()):
        raise RuntimeError(f"Missing protein token cache: {cache_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["n_sequences"])
    mm = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=(n, int(seq_len_tokens)))
    return mm, meta


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ProteinFastaDataset(Dataset):
    """
    Unconditional protein bitstream dataset.

    Path: amino-acid tokens -> raw-binary bits -> flattened [seq_len_tokens * bits_per_token].
    The tokenizer is only needed at cache-build / decode time; __getitem__ reads
    cached token ids and expands them to bits via a lookup table.
    """

    is_text_dataset = False

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.config = config
        self.split = split

        self.root = Path(getattr(config.data, "root", "datasets/swissprot"))
        self.repr = str(getattr(config.data, "representation", "binary")).lower().strip()
        if self.repr != "binary":
            raise ValueError("ProteinFastaDataset only supports representation='binary'.")

        self.tok_tag = tokenizer_tag_from_config(config)
        self.seq_len_tokens = int(getattr(config.data, "sequence_len_tokens", 256))
        self.min_len = int(getattr(config.data, "min_len", 20))

        ensure_swissprot_caches_ready(config)
        self.mm, self.meta = load_token_id_cache(
            self.root, split, self.tok_tag, self.seq_len_tokens, self.min_len
        )
        self.num_sequences = int(self.mm.shape[0])
        self.vocab_size = int(self.meta["vocab_size"])

        data_bits_default = _ceil_log2(self.vocab_size)
        bits_cfg = getattr(config.data, "bits_per_token", None)
        self.bits_per_token = int(bits_cfg) if bits_cfg is not None else int(data_bits_default)
        if self.bits_per_token < data_bits_default:
            raise ValueError(
                f"Need at least {data_bits_default} bits to encode {self.vocab_size} tokens, "
                f"got cfg.data.bits_per_token={self.bits_per_token}."
            )

        self.seq_len_bits = self.seq_len_tokens * self.bits_per_token
        cfg_seq_len = int(getattr(config.data, "sequence_len", self.seq_len_bits))
        if cfg_seq_len != self.seq_len_bits:
            raise ValueError(
                f"cfg.data.sequence_len={cfg_seq_len} != "
                f"sequence_len_tokens*bits_per_token={self.seq_len_bits}."
            )

        self.token_to_bits_table = build_token_to_bits_table(
            self.vocab_size, self.bits_per_token
        )
        tokenizer = make_protein_tokenizer(config)
        self.pad_id = int(tokenizer.pad_id)
        self.residue_id_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.residue_id_mask[list(sorted(tokenizer.residue_ids))] = True

        print(
            f"[proteins] split={split} tokenizer={self.tok_tag} vocab={self.vocab_size} "
            f"seq_tokens={self.seq_len_tokens} bits/token={self.bits_per_token} "
            f"seq_bits={self.seq_len_bits} num_seq={self.num_sequences} min_len={self.min_len}"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        row = np.array(self.mm[idx], dtype=np.int64, copy=True)
        toks_t = torch.from_numpy(row)
        bits = self.token_to_bits_table[toks_t].view(-1)
        nonpad_mask = toks_t.ne(self.pad_id).repeat_interleave(self.bits_per_token)
        residue_mask = self.residue_id_mask[toks_t].repeat_interleave(
            self.bits_per_token
        )
        return bits, nonpad_mask, residue_mask


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Frozen DiMA-compatible whole-sequence dataset
# -----------------------------------------------------------------------------

DIMA_CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


class FrozenDimaSwissProtDataset(Dataset):
    """Exact upstream rows, truncated to 254 aa, with no BOS/EOS/PAD objective."""

    is_text_dataset = False

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.requested_split = split
        self.split = "test" if split == "val" else split
        self.root = Path(
            getattr(config.data, "root", "datasets/swissprot_dima_bf4b2f13")
        )
        manifest_path = self.root / "frozen_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        expected_revision = str(
            getattr(
                config.data,
                "hf_revision",
                "bf4b2f131664fe87ef5e0fc9e53f01f0d030dcab",
            )
        )
        actual_revision = self.manifest["upstream"]["huggingface_revision"]
        if actual_revision != expected_revision:
            raise ValueError(
                f"Frozen dataset revision {actual_revision} != configured {expected_revision}"
            )

        protocol_dir = self.root / "protocol"
        binary_path = protocol_dir / f"{self.split}.sequences.bin"
        offsets_path = protocol_dir / f"{self.split}.offsets.npy"
        lengths_path = protocol_dir / f"{self.split}.lengths.npy"
        self.sequence_bytes = np.memmap(binary_path, dtype=np.uint8, mode="r")
        self.offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        self.lengths = np.load(lengths_path, mmap_mode="r", allow_pickle=False)
        limit_name = "limit_train" if self.split == "train" else "limit_eval"
        limit = int(getattr(config.data, limit_name, 0))
        if limit > 0:
            limit = min(limit, len(self.lengths))
            self.lengths = self.lengths[:limit]
            self.offsets = self.offsets[: limit + 1]

        self.representation = str(
            getattr(config.data, "representation", "binary")
        ).lower()
        self.prepend_bos = bool(getattr(config.data, "prepend_bos", False))
        if self.prepend_bos and self.representation != "tokens":
            raise ValueError("prepend_bos is only supported for token representation")
        if self.representation not in {"binary", "tokens"}:
            raise ValueError(
                f"Unsupported DiMA representation: {self.representation}"
            )
        self.bits_per_token = int(getattr(config.data, "bits_per_token", 5))
        if self.bits_per_token != 5:
            raise ValueError("The 20-residue raw-binary protocol requires 5 bits/token")
        self.vocab_size = len(DIMA_CANONICAL_AA)
        self.token_to_bits_table = build_token_to_bits_table(
            self.vocab_size, self.bits_per_token
        )
        self.byte_to_id = np.full(256, -1, dtype=np.int16)
        for token_id, residue in enumerate(DIMA_CANONICAL_AA):
            self.byte_to_id[ord(residue)] = token_id

        if len(self.offsets) != len(self.lengths) + 1:
            raise ValueError("Corrupt frozen sequence index: offsets/lengths disagree")
        print(
            f"[proteins:dima] requested={self.requested_split} source={self.split} "
            f"n={len(self.lengths):,} length={int(self.lengths.min())}-"
            f"{int(self.lengths.max())} revision={actual_revision[:12]}"
        )

    def __len__(self) -> int:
        return int(len(self.lengths))

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        residue_bytes = np.asarray(self.sequence_bytes[start:end], dtype=np.uint8)
        token_ids = self.byte_to_id[residue_bytes]
        if np.any(token_ids < 0):
            raise ValueError(f"Noncanonical residue in frozen row {idx}")
        ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
        if self.representation == "tokens":
            if self.prepend_bos:
                ids = torch.cat([torch.tensor([20], dtype=torch.long), ids])
            return ids
        return self.token_to_bits_table[ids].reshape(-1)


class DistributedLengthBucketBatchSampler(Sampler):
    """Same-length batches, synchronised by length and batch size across ranks."""

    def __init__(
        self,
        lengths,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.batch_size <= 0 or self.world_size <= 0:
            raise ValueError("batch_size and world_size must be positive")
        self._buckets = {
            int(length): np.flatnonzero(self.lengths == length).astype(np.int64)
            for length in np.unique(self.lengths)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _num_batches_for_size(self, size: int) -> int:
        global_batch = self.batch_size * self.world_size
        full, remainder = divmod(int(size), global_batch)
        return full + int(remainder >= self.world_size)

    def __len__(self) -> int:
        return sum(self._num_batches_for_size(len(v)) for v in self._buckets.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        global_batch = self.batch_size * self.world_size
        batches = []
        for indices_source in self._buckets.values():
            indices = indices_source.copy()
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), global_batch):
                group = indices[start : start + global_batch]
                usable = (len(group) // self.world_size) * self.world_size
                if usable == 0:
                    continue
                group = group[:usable]
                local_size = usable // self.world_size
                local = group[
                    self.rank * local_size : (self.rank + 1) * local_size
                ]
                batches.append(local.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def get_dima_loader(
    config: config_dict.ConfigDict,
    *,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    seed: int = 42,
) -> DataLoader:
    dataset = FrozenDimaSwissProtDataset(config, split=split)
    if shuffle is None:
        shuffle = split == "train"
    rank, world_size = _ddp_rank_world()
    batch_sampler = DistributedLengthBucketBatchSampler(
        dataset.lengths,
        batch_size=int(batch_size or config.train.batch_size),
        shuffle=bool(shuffle),
        seed=int(seed),
        rank=rank,
        world_size=world_size,
    )
    num_workers = int(getattr(config.data, "num_workers", 8))
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            int(getattr(config.data, "prefetch_factor", 4))
            if num_workers > 0
            else None
        ),
    )


# Dataloaders
# -----------------------------------------------------------------------------

def get_dataloaders(
    config: config_dict.ConfigDict,
    *,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ensure_swissprot_caches_ready(config)

    batch = int(batch_size or config.train.batch_size)

    train_ds = ProteinFastaDataset(config, split="train")
    val_ds = ProteinFastaDataset(config, split="val")
    test_ds = ProteinFastaDataset(config, split="test")

    num_workers = int(getattr(config.data, "num_workers", 8))
    prefetch_factor = int(getattr(config.data, "prefetch_factor", 4))
    pin_memory = bool(getattr(config.data, "pin_memory", True))
    persistent_workers = num_workers > 0

    g = torch.Generator()
    g.manual_seed(int(seed))

    def _worker_init_fn(worker_id: int) -> None:
        base = int(seed) + int(worker_id)
        np.random.seed(base % (2**32 - 1))
        torch.manual_seed(base)

    rank, world_size = _ddp_rank_world()

    def make_loader(ds: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        sampler = None
        loader_shuffle = shuffle
        if _ddp_is_on():
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=int(seed),
            )
            loader_shuffle = False

        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=loader_shuffle,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            generator=g if (shuffle and sampler is None) else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
