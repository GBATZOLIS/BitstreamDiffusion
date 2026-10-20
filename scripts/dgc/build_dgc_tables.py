"""Build DGC-schedule entropy tables from a CoBit entropy table.

    python scripts/dgc/build_dgc_tables.py assets/entropy_tables/lm1b assets/entropy_tables/lm1b
    (args: <source table dir> <output prefix>; writes <prefix>_dgc/)

Shipped table: pdf(u) ∝ g(σ)·(h(σ)/σ²)^α on equal-width bins in u = log σ,
  h = per-bit denoising MSE, g = σ^n/(σ^n + c^n).  Released checkpoints: α=½, c=0.1, n=3.
Recover h ∝ σ²·(pdf/g)^(1/α).
DGC (Wainwright, arXiv:2607.26285; VE heat time t = σ²): optimal step density in log t ∝ √h'(t),
and h'(t) = (dh/du)/(2σ²), so in u the density is ∝ √((dh/du)/σ²).
"""
import os
import shutil
import sys

import numpy as np
import torch

src, prefix = sys.argv[1], sys.argv[2]
alpha = float(os.environ.get("ALPHA", 0.5))
c = float(os.environ.get("C", 0.1))
n = float(os.environ.get("N", 3.0))

ld = lambda f: torch.load(f"{src}/{f}", map_location="cpu", weights_only=True).double().numpy()
pdf, sig = ld("entropy_pdf.pt"), ld("entropy_sigmas.pt")
u = np.log(sig)
g = sig**n / (sig**n + c**n)

h = sig**2 * (pdf / g) ** (1.0 / alpha)
h /= h.max()
k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
k /= k.sum()
hs = np.convolve(np.pad(h, 6, mode="edge"), k, mode="valid")
dh = np.gradient(hs, u)
dgc = np.sqrt(np.clip(dh, 0, None) / sig**2)

print(f"h(σ) recovered: monotone fraction={np.mean(np.diff(hs) >= 0):.3f}, "
      f"bins with dh<0: {np.round(sig[dh < 0], 2).tolist()}")

dg = dgc / dgc.sum()
for name, p in [("dgc", dg)]:
    p = p / p.sum()
    out = f"{prefix}_{name}"
    os.makedirs(out, exist_ok=True)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    torch.save(torch.tensor(p, dtype=torch.float32), f"{out}/entropy_pdf.pt")
    torch.save(torch.tensor(cdf, dtype=torch.float32), f"{out}/entropy_cdf.pt")
    for f in ("entropy_sigmas.pt", "entropy_edges.pt"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
    print("wrote", out)
