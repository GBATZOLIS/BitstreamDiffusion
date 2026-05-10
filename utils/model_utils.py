# utils/model_utils.py
import gc
import torch
import torch.nn as nn

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively peels off DDP (.module) and torch.compile (._orig_mod) wrappers.
    Crucial for preventing Dynamo from caching massive inference graphs during evaluation.
    """
    m = model
    while hasattr(m, "module") or hasattr(m, "_orig_mod"):
        m = getattr(m, "module", m)
        m = getattr(m, "_orig_mod", m)
    return m

def free_vram():
    """Aggressively force Python garbage collection and empty the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()