"""
evaluation/mnist_sum_classifier.py

The quadrant digit classifier that makes text->image and joint generation
EXACTLY scorable, plus the ceiling that must be reported beside every number
it produces.

Why a ceiling. The classifier is a measuring instrument, not part of the model,
so a t2i score of 96% means nothing until you know what the instrument reads on
REAL composites. That number is the ceiling, and any shortfall below it is the
generative model. This is the same discipline the paper already applies to the
rFID floor -- with the difference that here there is no codec loss, so the
ceiling should be near-perfect and the headroom is almost all real.

Train/test hygiene: the classifier is trained on MNIST's TRAIN digits, while
every val/val_holdout composite is built from MNIST's TEST digits, so the
ceiling is measured on glyphs the classifier has never seen. Binarisation is
`pixel > 127`, byte-identical to the corpus builder -- a classifier trained on
greyscale would read a different image than the one the model generates.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Candidate locations for MNIST's raw IDX files, tried in order. The CSD3 path
# came first because the harness was written there; on any other machine it does
# not exist, and a hardcoded absolute path fails at classifier-training time --
# after the checkpoint has loaded and compiled, minutes into a run. Resolve it
# instead, and say what was tried when nothing matches.
MNIST_RAW_CANDIDATES = (
    "/rds/project/rds-LlrDsbHU5UM/gb511/datasets/MNISTDataset/raw",  # CSD3
    "datasets/MNIST/raw",                                            # repo-local
    "datasets/MNISTDataset/raw",
)
_IDX_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
              "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")


def _has_idx(d: Path) -> bool:
    return all((d / f).is_file() for f in _IDX_FILES)


def resolve_mnist_raw(raw_dir: Optional[str] = None) -> str:
    """Locate MNIST raw IDX files: explicit arg, then $MNIST_RAW, then candidates.

    Both arms must read the SAME digits, so this resolves to one directory and
    the caller records which one in the results.
    """
    tried = []
    for cand in ([raw_dir] if raw_dir else []) + \
                ([os.environ["MNIST_RAW"]] if os.environ.get("MNIST_RAW") else []) + \
                list(MNIST_RAW_CANDIDATES):
        d = Path(cand)
        tried.append(str(d))
        if _has_idx(d):
            return str(d)
    raise FileNotFoundError(
        "MNIST raw IDX files not found. The quadrant classifier needs all four of "
        f"{', '.join(_IDX_FILES)}. Tried:\n  " + "\n  ".join(tried) +
        "\nPass --mnist_raw /path/to/raw or set $MNIST_RAW."
    )


class QuadrantCNN(nn.Module):
    """Small CNN over a single binarised 28x28 quadrant -> digit 0-9."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):                      # x [B,1,28,28] in {0,1}
        x = F.relu(self.c1(x)); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x)); x = F.max_pool2d(F.relu(self.c4(x)), 2)
        x = x.flatten(1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))


def load_binarised_mnist(raw_dir: str, split: str, threshold: int = 127):
    """Raw IDX -> binarised float tensors. Mirrors the corpus builder exactly."""
    raw = Path(raw_dir)
    img_f = "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
    lbl_f = "train-labels-idx1-ubyte" if split == "train" else "t10k-labels-idx1-ubyte"
    with open(raw / img_f, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2051
        n = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
        w = int.from_bytes(f.read(4), "big")
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w).copy()
    with open(raw / lbl_f, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2049
        int.from_bytes(f.read(4), "big")
        lbls = np.frombuffer(f.read(), dtype=np.uint8).copy()
    x = torch.from_numpy((imgs > threshold).astype(np.float32)).unsqueeze(1)
    return x, torch.from_numpy(lbls.astype(np.int64))


def train_classifier(device, *, raw_dir: Optional[str] = None, epochs: int = 4,
                     batch_size: int = 256, lr: float = 1e-3,
                     limit: Optional[int] = None, seed: int = 0, verbose: bool = True
                     ) -> Tuple[QuadrantCNN, Dict[str, float]]:
    torch.manual_seed(seed)
    raw_dir = resolve_mnist_raw(raw_dir)
    if verbose:
        print(f"[clf] MNIST raw: {raw_dir}", flush=True)
    xtr, ytr = load_binarised_mnist(raw_dir, "train")
    xte, yte = load_binarised_mnist(raw_dir, "test")
    if limit:                                   # smoke path only
        xtr, ytr, xte, yte = xtr[:limit], ytr[:limit], xte[:limit], yte[:limit]
    model = QuadrantCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        acc = _accuracy(model, xte, yte, device)
        if verbose:
            print(f"[clf] epoch {ep+1}/{epochs} loss={tot/n:.4f} mnist_test_acc={100*acc:.3f}%",
                  flush=True)
    return model, {"mnist_test_digit_accuracy": _accuracy(model, xte, yte, device)}


@torch.no_grad()
def _accuracy(model, x, y, device, bs: int = 1024) -> float:
    model.eval(); ok = 0
    for i in range(0, x.shape[0], bs):
        pred = model(x[i:i + bs].to(device)).argmax(1).cpu()
        ok += int((pred == y[i:i + bs]).sum())
    return ok / x.shape[0]


@torch.no_grad()
def classify_quadrants(model, quads: torch.Tensor, device, bs: int = 512) -> torch.Tensor:
    """[B,4,28,28] in [0,1] -> [B,4] predicted digits. Binarised to match training."""
    model.eval()
    B = quads.shape[0]
    flat = (quads.reshape(B * 4, 1, 28, 28) > 0.5).float()
    out = []
    for i in range(0, flat.shape[0], bs):
        out.append(model(flat[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out).reshape(B, 4)


@torch.no_grad()
def measure_ceiling(model, ds, device, n: int = 4096) -> Dict[str, float]:
    """
    The instrument's own accuracy on REAL composites from `ds`.

    This is the ceiling for every t2i / joint number: a generated-image score
    cannot exceed what the classifier reads on genuine data, so it must be
    reported alongside, and the shortfall below it is what the model owns.
    """
    n = min(n, len(ds))
    idx = np.arange(n)
    bits = torch.stack([ds[int(i)] for i in idx]).float()
    quads = ds.reconstruct_batch_from_bits(bits)["quadrants"]
    true, _ = ds.labels(idx)
    pred = classify_quadrants(model, quads, device)
    per_slot = (pred == true).float().mean(0)
    return {
        "n": float(n),
        "ceiling_all_four": float((pred == true).all(1).float().mean()),
        "ceiling_per_quadrant": float(per_slot.mean()),
        **{f"ceiling_{k}": float(per_slot[i])
           for i, k in enumerate(("top_left", "top_right", "bottom_left", "bottom_right"))},
    }


def save(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def load(path: str, device) -> QuadrantCNN:
    m = QuadrantCNN()
    m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return m.to(device).eval()
