"""Load DPLM sequence modules without importing unrelated DPLM-2 stacks."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def load_dplm_class(dplm_root: Path):
    """Import the official sequence DPLM files from a pinned checkout.

    The upstream ``byprot`` package eagerly imports every data, structure, and
    multimodal module. Sequence-only generation does not need OpenFold,
    torch-scatter, or DPLM-2, so we register only the official DPLM sequence
    package and its model registry.
    """
    source = dplm_root / "src" / "byprot"
    if not source.exists():
        raise FileNotFoundError(f"DPLM source tree not found at {source}")

    byprot = types.ModuleType("byprot")
    byprot.__path__ = [str(source)]
    byprot.__package__ = "byprot"
    sys.modules["byprot"] = byprot

    models = types.ModuleType("byprot.models")
    models.__path__ = [str(source / "models")]
    models.__package__ = "byprot.models"
    models.MODEL_REGISTRY = {}

    def register_model(name):
        def decorator(cls):
            models.MODEL_REGISTRY[name] = cls
            return cls

        return decorator

    models.register_model = register_model
    sys.modules["byprot.models"] = models

    dplm_package = types.ModuleType("byprot.models.dplm")
    dplm_package.__path__ = [str(source / "models" / "dplm")]
    dplm_package.__package__ = "byprot.models.dplm"
    sys.modules["byprot.models.dplm"] = dplm_package

    importlib.import_module("byprot.models.dplm.modules.dplm_modeling_esm")
    module = importlib.import_module("byprot.models.dplm.dplm")
    return module.DiffusionProteinLanguageModel
