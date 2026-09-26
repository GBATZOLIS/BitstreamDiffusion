from __future__ import annotations

from pathlib import Path

import math
import json
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


from utils.ecc_secded import ecc_chunk_len, ecc_encode_data_bits, ecc_from_cfg

def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


def _load_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(
            "Missing text-audio token cache files. Run scripts/text_audio/prepare_train.py "
            "or scripts/text_audio/prepare_eval.py first.\n"
            f"Expected: {cache_path} and {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta['n_sequences'])
    seq_len = int(meta['seq_len_tokens'])
    mm = np.memmap(cache_path, dtype=np.uint32, mode='r', shape=(n,seq_len))
    return mm, meta

def _build_token_to_bits_table(vocab_size: int, bits_per_token: int) -> torch.Tensor:
    ids = torch.arange(vocab_size, dtype=torch.long)
    shifts = torch.arange(bits_per_token-1, -1, -1, dtype=torch.long)
    return (ids.unsqueeze(1) >> shifts) & 1

def _val_geometry_suffix(config: config_dict.ConfigDict) -> str:
    seq_len_tokens = int(getattr(config.data, 'seq_len_tokens', 1000))
    return f'_{seq_len_tokens}'


def _test_geometry_suffix(config: config_dict.ConfigDict) -> str:
    seq_len_tokens = int(getattr(config.data, 'seq_len_tokens', 1000))
    return f'_{seq_len_tokens}'


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def _ddp_rank_world() -> tuple[int, int]:
    if not _ddp_is_on():
        return 0,1
    return int(dist.get_rank()), int(dist.get_world_size())

class _PackedTokenCacheDataset(Dataset):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        cache_path: Path,
        meta_path: Path,
        expected_cache_format: str,
        expected_seq_len_tokens: Optional[int] = None,
        log_tag: str = "textaudio",
    ):
        super().__init__()
        self.config = config

        self.repr = str(getattr(config.data, 'representation', 'binary')).lower().strip()
        self.binarization = str(getattr(config.data, 'binarization', 'raw_binary')).lower().strip()
        self.token_space = str(getattr(config.data, 'token_space', 'raw')).lower().strip()

        if self.repr == "binary":
            if self.binarization not in {"semantic", "raw_binary"}:
                raise ValueError(
                    f"Unknown cfg.data.binarization={self.binarization!r} for Text-audio binary mode. "
                    "Supported: 'semantic', 'raw_binary'."
                )
        elif self.repr == "tokens":
            if self.token_space not in {
                "semantic_rank",
                "semantic",
                "rank",
                "tokenizer_id",
                "raw",
                "tokenizer",
            }:
                raise ValueError(f"Unknown cfg.data.token_space={self.token_space}")
        else:
            raise ValueError(f"Unsupported Text-Audio representation={self.repr}")

        self.vocab_size_base = int(getattr(config.data, 'vocab_size_base', 250773))
        self.pad_token_text_id = self.vocab_size_base-2
        self.pad_token_speech_id = self.vocab_size_base-1
        self.unk_token_id = None

        self.ecc = ecc_from_cfg(config)
        self.ecc_enabled = bool(self.ecc.enabled)

        data_bits_default = _ceil_log2(self.vocab_size_base)
        data_bits_cfg = getattr(config.data, "bits_per_token", None)
        self.data_bits_per_token = int(data_bits_cfg) if data_bits_cfg is not None else int(data_bits_default)

        min_bits_needed = _ceil_log2(self.vocab_size_base)
        if self.data_bits_per_token < min_bits_needed:
            raise ValueError(
                f"Text-audio data requires at least {min_bits_needed} data bits to encode vocab_size={self.vocab_size_base}, "
                f"but cfg.data.bits_per_token={self.data_bits_per_token}."
            )

        if self.ecc_enabled:
            if int(self.ecc.data_bits) != int(self.data_bits_per_token):
                raise ValueError(
                    f"ECC enabled but cfg.data.ecc.data_bits={int(self.ecc.data_bits)} does not match "
                    f"bits_per_token={int(self.data_bits_per_token)}."
                )
            self.bits_per_token = int(ecc_chunk_len(self.ecc))
        else:
            self.bits_per_token = int(self.data_bits_per_token)

        if self.repr == 'binary':
            self.token_to_bits_table = _build_token_to_bits_table(
                self.vocab_size_base, self.bits_per_token
            )

        mm, meta = _load_memmap(cache_path, meta_path)
        self.mm = mm

        cache_format = str(meta.get("cache_format", "legacy_unknown"))
        if cache_format != expected_cache_format:
            raise RuntimeError(
                f"{cache_path} has cache_format={cache_format!r}, expected {expected_cache_format!r}.\n"
                f"Delete it and rebuild with the matching scripts/text_audio/prepare_*.py script."
            )
        self.cache_format = expected_cache_format

        self.seq_len_tokens = int(meta["seq_len_tokens"])
        if expected_seq_len_tokens is not None and self.seq_len_tokens != int(expected_seq_len_tokens):
            raise RuntimeError(
                f"{cache_path} seq_len_tokens={self.seq_len_tokens} but config expects "
                f"{expected_seq_len_tokens}."
            )
        self.seq_len_bits = self.seq_len_tokens*self.bits_per_token

        self.start = 0
        self.end = int(self.mm.shape[0])
        self.num_sequences = int(self.end-self.start)

        print(
            f"[{log_tag}] cache={cache_path} cache_format={self.cache_format} "
            f"repr={self.repr} "
            f"{'binarization=' + self.binarization if self.repr == 'binary' else 'token_space=' + self.token_space} "
            f"vocab_base={self.vocab_size_base} seq_tokens={self.seq_len_tokens} "
            f"bits/token={self.bits_per_token} num_seq={self.num_sequences} "
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.array(self.mm[self.start+idx], dtype=np.int64, copy=True)
        toks_t = torch.from_numpy(row)
        bits = self.token_to_bits_table[toks_t]
        return bits.view(-1)


class TextAudioDataset(_PackedTokenCacheDataset):
    TASK = 'asr'

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        assert split in {'train', 'val', 'test'}
        self.split = split
        root = Path(getattr(config.data, 'root', 'datasets/'))

        if split == 'train':
            dataset_name = str(config.data.dataset)
            stem = f'{dataset_name}/cache_{dataset_name}_train'
            cache_path = root / f'{stem}.uint32'
            meta_path = root / f'{stem}.meta.json'
        elif split == 'val':
            stem = f'validation/cache_val_{self.TASK}{_val_geometry_suffix(config)}'
            cache_path = root / f'{stem}.uint32'
            meta_path = root / f'{stem}.meta.json'
        else:
            test_partition = str(getattr(config.data, 'partition', 'clean'))
            stem = f'test_common/cache_test_{test_partition}_{self.TASK}{_test_geometry_suffix(config)}'
            cache_path = root / f'{stem}.uint32'
            meta_path = root / f'{stem}.meta.json'

        log_tag = 'asr' if split == 'val' else 'textaudio' 

        super().__init__(
            config,
            cache_path=cache_path,
            meta_path=meta_path,
            expected_cache_format='packed_multimodal_blocks',
            expected_seq_len_tokens=int(getattr(config.data, 'seq_len_tokens', 1000)),
            log_tag=log_tag,
        )


class TextAudioTTSDataset(_PackedTokenCacheDataset):
    TASK = 'tts'

    def __init__(
        self,
        config: config_dict.ConfigDict,
        *,
        split: str,
        cache_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        log_tag: str = 'tts',
    ):
        if split not in {'val', 'test'}:
            raise ValueError(
                f"TextAudioTTSDataset has no split={split!r}; only 'val'/'test' exist."
            )
        root = Path(getattr(config.data, 'root', 'datasets/'))

        if (cache_path is None) != (meta_path is None):
            raise ValueError('cache_path and meta_path must be provided together.')

        if cache_path is not None:
            if split != 'test':
                raise ValueError('A custom TTS cache is only supported for split="test".')
            cache_path = Path(cache_path)
            meta_path = Path(meta_path)
        elif split == 'val':
            stem = f'validation/cache_val_{self.TASK}{_val_geometry_suffix(config)}'
            cache_path = root / f'{stem}.uint32'
            meta_path = root / f'{stem}.meta.json'
        else:
            test_partition = str(getattr(config.data, 'partition', 'clean'))
            stem = f'test_common/cache_test_{test_partition}_{self.TASK}{_test_geometry_suffix(config)}'
            cache_path = root / f'{stem}_common.uint32'
            meta_path = root / f'{stem}.meta.json'

        text_seq_len = int(getattr(config.data, 'text_seq_len', 168))
        speaker_seq_len = int(getattr(config.data, 'speaker_seq_len', 32))

        super().__init__(
            config,
            cache_path=cache_path,
            meta_path=meta_path,
            expected_cache_format='packed_tts_blocks',
            expected_seq_len_tokens=text_seq_len + speaker_seq_len,
            log_tag=log_tag,
        )
        self.split = split


class TextAudioContinuationDataset(_PackedTokenCacheDataset):
    TASK = 'cont'

    def __init__(self, config: config_dict.ConfigDict, *, split: str, cache_name: str | None = None):
        if split not in {'val', 'test'}:
            raise ValueError(
                f"TextAudioContinuationDataset has no split={split!r}; only 'val'/'test' exist."
            )
        root = Path(getattr(config.data, 'root', 'datasets/'))

        if cache_name is not None:
            stem = f'test/{cache_name}'
        elif split == 'val':
            stem = f'validation/cache_val_{self.TASK}'
        else:
            stem = f'test_common/cache_test_{self.TASK}'

        super().__init__(
            config,
            cache_path=root / f'{stem}.uint32',
            meta_path=root / f'{stem}.meta.json',
            expected_cache_format='packed_continuation_prefix_blocks',
            expected_seq_len_tokens=None,
            log_tag='cont',
        )
        self.split = split


def get_dataloaders(
    config: config_dict.ConfigDict,
    *,
    batch_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if str(config.data.dataset) not in {'textaudio', 'libri'}:
        raise NotImplementedError("config.data.dataset must be 'textaudio' or 'libri'")

    batch = int(batch_size or config.train.batch_size)

    train_ds = TextAudioDataset(config, split='train')
    val_ds = TextAudioDataset(config, split='val')
    test_ds = TextAudioDataset(config, split='test')

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
                num_replicas = world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=int(seed)
            )
            loader_shuffle=False
        
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
