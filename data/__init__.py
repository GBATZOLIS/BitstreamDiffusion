from __future__ import annotations

from typing import Tuple, Literal

from torch.utils.data import DataLoader
from ml_collections import config_dict

from .openwebtext import OpenWebTextDataset
from .lm1b import LM1BDataset

Split = Literal["train", "val", "test"]


def _norm_dataset_name(name: object) -> str:
    return str(name or "").strip()


def get_loader(
    config: config_dict.ConfigDict,
    *,
    split: Split,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    """
    Return a single DataLoader for the requested split only.
    """
    assert split in {"train", "val", "test"}, f"Unknown split: {split}"
    name = _norm_dataset_name(getattr(config.data, "dataset", None))

    batch = int(batch_size or config.train.batch_size)
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split != "test")

    def _make_direct_loader(ds):
        num_workers = int(getattr(config.data, "num_workers", 4))
        prefetch_factor = int(getattr(config.data, "prefetch_factor", 2))
        pin_memory = bool(getattr(config.data, "pin_memory", True))
        persistent_workers = num_workers > 0

        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    # ---------------- OpenWebText ----------------
    # Old names "OpenWebTextFLM" / "openwebtext_flm" are kept as aliases so
    # archived configs from prior runs continue to load against the renamed dataset.
    if name in {"OpenWebText", "openwebtext", "owt", "OpenWebTextFLM", "openwebtext_flm"}:
        ds = OpenWebTextDataset(config, split=split)
        return _make_direct_loader(ds)

    # ---------------- LM1B ----------------
    if name in {"LM1B", "lm1b", "OneBillionWords", "one_billion_words"}:
        ds = LM1BDataset(config, split=split)
        return _make_direct_loader(ds)

    # ---------------- Sudoku (task) ----------------
    if name in {"sudoku", "Sudoku"}:
        from .sudoku import SudokuDataset
        return _make_direct_loader(SudokuDataset(config, split=split))

    # ---------------- TinyGSM / GSM8K (task) ----------------
    if name in {"tinygsm", "TinyGSM", "gsm8k", "GSM8K"}:
        from .tinygsm import TinyGSMDataset
        return _make_direct_loader(TinyGSMDataset(config, split=split))

    raise NotImplementedError(
        f"Unknown dataset '{name}'. Supported: 'OpenWebText', 'LM1B', 'sudoku', 'tinygsm'."
    )


def get_dataloaders(
    config: config_dict.ConfigDict,
    *,
    batch_size: int | None = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Generic dataloader factory.
    """
    name = _norm_dataset_name(getattr(config.data, "dataset", None))

    if name in {"OpenWebText", "openwebtext", "owt", "OpenWebTextFLM", "openwebtext_flm"}:
        from .openwebtext import get_dataloaders as _owt_get_dataloaders
        return _owt_get_dataloaders(config, batch_size=batch_size, seed=seed)

    if name in {"LM1B", "lm1b", "OneBillionWords", "one_billion_words"}:
        from .lm1b import get_dataloaders as _lm1b_get_dataloaders
        return _lm1b_get_dataloaders(config, batch_size=batch_size, seed=seed)

    if name in {"sudoku", "Sudoku"}:
        from .sudoku import get_dataloaders as _sudoku_get_dataloaders
        return _sudoku_get_dataloaders(config, batch_size=batch_size, seed=seed)

    if name in {"tinygsm", "TinyGSM", "gsm8k", "GSM8K"}:
        from .tinygsm import get_dataloaders as _tinygsm_get_dataloaders
        return _tinygsm_get_dataloaders(config, batch_size=batch_size, seed=seed)

    raise NotImplementedError(
        f"Unknown dataset '{name}'. Supported: 'OpenWebText', 'LM1B', 'sudoku', 'tinygsm'."
    )
