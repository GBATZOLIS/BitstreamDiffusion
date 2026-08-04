from __future__ import annotations

from typing import Tuple, Literal

from torch.utils.data import DataLoader
from ml_collections import config_dict

from .openwebtext import OpenWebTextDataset
from .lm1b import LM1BDataset
from .proteins import ProteinFastaDataset, get_dima_loader
from .uniref50 import EvoDiffUniRef50Dataset, get_evodiff_uniref50_loader

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

    # ---------------- Frozen DiMA Swiss-Prot protocol ----------------
    if name in {"SwissProtDiMA", "swissprot_dima", "dima_swissprot"}:
        return get_dima_loader(
            config,
            split=split,
            batch_size=batch,
            shuffle=shuffle,
            seed=seed,
        )

    # ---------------- Frozen EvoDiff/DPLM UniRef50 protocol ----------------
    if name in {"EvoDiffUniRef50", "evodiff_uniref50", "uniref50_evodiff"}:
        return get_evodiff_uniref50_loader(
            config, split=split, batch_size=batch, shuffle=shuffle, seed=seed
        )

    # ---------------- Multimodal paired 18-bit protocol ----------------
    if name in {"ProteinMultimodalLFQ", "protein_multimodal", "dplm_paired"}:
        from .protein_multimodal import get_multimodal_loader
        return get_multimodal_loader(
            config, split=split, batch_size=batch, shuffle=shuffle, seed=seed
        )

    # ---------------- Swiss-Prot proteins ----------------

    if name in {"Proteins", "proteins", "SwissProt", "swissprot"}:
        ds = ProteinFastaDataset(config, split=split)
        return _make_direct_loader(ds)

    raise NotImplementedError(
        f"Unknown dataset '{name}'. Supported: 'OpenWebText', 'LM1B', 'SwissProt'."
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

    if name in {"Proteins", "proteins", "SwissProt", "swissprot"}:
        from .proteins import get_dataloaders as _proteins_get_dataloaders
        return _proteins_get_dataloaders(config, batch_size=batch_size, seed=seed)


    if name in {"SwissProtDiMA", "swissprot_dima", "dima_swissprot"}:
        train = get_dima_loader(config, split="train", batch_size=batch_size, shuffle=True, seed=seed)
        val = get_dima_loader(config, split="val", batch_size=batch_size, shuffle=False, seed=seed)
        test = get_dima_loader(config, split="test", batch_size=batch_size, shuffle=False, seed=seed)
        return train, val, test
    if name in {"EvoDiffUniRef50", "evodiff_uniref50", "uniref50_evodiff"}:
        train = get_evodiff_uniref50_loader(config, split="train", batch_size=batch_size, shuffle=True, seed=seed)
        val = get_evodiff_uniref50_loader(config, split="val", batch_size=batch_size, shuffle=False, seed=seed)
        test = get_evodiff_uniref50_loader(config, split="test", batch_size=batch_size, shuffle=False, seed=seed)
        return train, val, test
    if name in {"ProteinMultimodalLFQ", "protein_multimodal", "dplm_paired"}:
        from .protein_multimodal import get_multimodal_loader
        train = get_multimodal_loader(config, split="train", batch_size=batch_size, shuffle=True, seed=seed)
        val = get_multimodal_loader(config, split="val", batch_size=batch_size, shuffle=False, seed=seed)
        return train, val, None
    raise NotImplementedError(
        f"Unknown dataset '{name}'. Supported: 'OpenWebText', 'LM1B', 'SwissProt'."
    )
