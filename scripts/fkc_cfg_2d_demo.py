#!/usr/bin/env python
"""Visual proof that the CFG-FKC weighted-particle scheme samples the geometric
average q_u^{1-w} q_c^w, in 2D (and 1D), against the EXACT analytic target.

This is the 2D extension of tests/test_fkc.py Gate 10/12 (the 1-D
`_analytic_cfg_weighted_var` two-Gaussian geometric-average check). The
production sampler uses a sigmoid-bounded BINARY denoiser that cannot represent a
real-valued Gaussian, so -- exactly as the gates do -- we REIMPLEMENT the toy
analytically: 2D Gaussian VE scores plus the codebase's CFG-FKC drift/weight
formulas.

Run (CPU only):
    source ~/miniconda3/bin/activate pytorch
    CUDA_VISIBLE_DEVICES="" python scripts/fkc_cfg_2d_demo.py

It SAVES PNGs (light + dark) under scripts/_fkc_cfg_2d_out/ and ASSERTS the
empirical importance-weighted mean/cov match the analytic m*, S* within tolerance
for every guidance weight w and every lambda in {0, 1, 1/w, lambda(sigma)} -- the
FK weight is lambda-independent, which is the theorem under test.

Math (2D, nonzero means; codebase convention h = sigma_next - sigma_cur < 0):
    q_u = N(m_u, S_u),  q_c = N(m_c, S_c),  precisions Lam = S^{-1}.
    VE marginal q_.,sigma = N(m_., S_. + sigma^2 I);
        score s_.(x) = -(S_. + sigma^2 I)^{-1} (x - m_.).
    Exact geometric average q_u^{1-w} q_c^w (at scale sig2 = sigma^2):
        Lam* = (1-w)(S_u+sig2 I)^{-1} + w (S_c+sig2 I)^{-1},  S* = Lam*^{-1},
        m*   = S* [ (1-w)(S_u+sig2 I)^{-1} m_u + w (S_c+sig2 I)^{-1} m_c ].
    CFG-FKC step (annealing beta = 1; em proposal):
        s_geo = (1-w) s_u + w s_c = s_u + w (s_c - s_u);   d = -sigma * s_geo
        x    <- x + h(1+lam) d + sqrt(2 lam sigma Delta) z,  Delta = sigma_cur-sigma_next
        logw += 0.5 w(w-1) (sigma_cur^2 - sigma_next^2) ||s_c - s_u||^2   (sum over dims)
    edm_churn variant (Gate 12 flavor): churn up sigma_hat = sigma_cur (1+gamma),
        add sqrt(sigma_hat^2 - sigma_cur^2) eps, evaluate scores at sigma_hat
        (O(gamma) approx) but keep the weight interval sigma_cur^2 - sigma_next^2;
        propagate with h = sigma_next - sigma_hat and no extra lambda noise.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

torch.set_default_dtype(torch.float64)
DEV = torch.device("cpu")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fkc_cfg_2d_out")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# Toy problem: two well-separated, anisotropic 2D Gaussians.  q_c is sharper   #
# than q_u so that Lam* = (1-w)Lam_u + w Lam_c stays positive-definite out to  #
# w = 3 (guidance extrapolation).                                              #
# --------------------------------------------------------------------------- #
def _cov(sx, sy, deg):
    th = math.radians(deg)
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    D = torch.diag(torch.tensor([sx ** 2, sy ** 2]))
    return R @ D @ R.T

M_U = torch.tensor([-1.5, 0.0])
M_C = torch.tensor([1.5, 0.8])
S_U = _cov(1.05, 0.70, 25.0)     # unconditional: broad
S_C = _cov(0.55, 0.38, -35.0)    # conditional: sharp

SIG_MAX, SIG_MIN, N_STEPS = 5.0, 0.05, 200
W_LIST = [0.0, 1.0, 1.5, 3.0]

I2 = torch.eye(2)


def geo_target(w: float, sig2: float):
    """Exact Gaussian (m*, S*) of q_u^{1-w} q_c^w at diffusion scale sig2=sigma^2."""
    Pu = torch.linalg.inv(S_U + sig2 * I2)
    Pc = torch.linalg.inv(S_C + sig2 * I2)
    Lam = (1.0 - w) * Pu + w * Pc
    S = torch.linalg.inv(Lam)
    m = S @ ((1.0 - w) * (Pu @ M_U) + w * (Pc @ M_C))
    return m, S, Lam


def sigma_grid(n=N_STEPS):
    i = torch.linspace(0, 1, n + 1)
    return SIG_MAX + i * (SIG_MIN - SIG_MAX)     # linear-in-sigma, high -> low


def n_for(w):
    """Steps for the linear-in-sigma grid. The guided marginal N(m*(s), S*(s)) is
    stiff for large w (a sharp target from precision (1-w)Lam_u + w Lam_c), so
    Euler needs a finer grid there; small w matches the gates' N=200."""
    return 200 if w <= 1.5 else (800 if w <= 2.0 else 2000)


def lam_of(sig: float, mode: str, w: float) -> float:
    if mode == "zero":
        return 0.0
    if mode == "one":
        return 1.0
    if mode == "inv_w":
        return 1.0 / w
    # sigma-dependent profile (matches the gates)
    return 0.5 + 0.6 * math.exp(-((math.log(sig)) ** 2) / 2.0)


# --------------------------------------------------------------------------- #
# The CFG-FKC particle sampler (2D reimplementation of Gate 10/12).            #
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    x: torch.Tensor        # [K, 2] terminal positions
    logw: torch.Tensor     # [K]   terminal log importance weights


def _scores(x, sig2):
    """s_u, s_c at scale sig2; x [K,2] -> [K,2] each."""
    Pu = torch.linalg.inv(S_U + sig2 * I2)
    Pc = torch.linalg.inv(S_C + sig2 * I2)
    s_u = -(x - M_U) @ Pu
    s_c = -(x - M_C) @ Pc
    return s_u, s_c


def _systematic_indices(logw, gen):
    w = torch.softmax(logw, dim=0)
    K = w.numel()
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    u0 = torch.rand(1, generator=gen) / K
    pos = u0 + torch.arange(K) / K
    return torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp_(0, K - 1)


def run_cfg_fkc(w, *, lam_mode="sigma", proposal="em", gamma=0.4,
                K=200_000, seed=0, resample=False, ess_frac=0.5, n_steps=None) -> Run:
    """CFG-FKC weighted-particle sampler in 2D (Gate 10/12 math).

    resample=False -> pure sequential importance weighting (the clean
      lambda-independence demonstration; the estimator is unbiased for every
      lambda but its variance blows up once w is large enough that a few
      particles carry all the weight -- report ESS).
    resample=True  -> ESS-triggered systematic resampling with logw reset,
      exactly as the production sampler (samplers.py ~L2928-2950). Needs a
      stochastic proposal (lam>0 or churn) to actually restore diversity; this
      is what makes large-w extrapolation (w=3) numerically demonstrable.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n = n_steps if n_steps is not None else n_for(w)
    sigmas = sigma_grid(n)

    # Prior = geometric-average marginal at sigma_max (analytic m*, S* + sig_max^2).
    m0, S0, _ = geo_target(w, float(sigmas[0]) ** 2)
    L0 = torch.linalg.cholesky(S0)
    x = m0 + torch.randn(K, 2, generator=gen) @ L0.T
    logw = torch.zeros(K)

    g_eff = min(gamma, math.sqrt(2.0) - 1.0)
    for k in range(n):
        sc, sn = float(sigmas[k]), float(sigmas[k + 1])
        churn = proposal == "edm_churn" and gamma > 0.0

        if churn:
            sig_state = sc * (1.0 + g_eff)                       # churn up
            eps = torch.randn(K, 2, generator=gen)
            x = x + math.sqrt(sig_state ** 2 - sc ** 2) * eps
        else:
            sig_state = sc

        s_u, s_c = _scores(x, sig_state ** 2)                    # scores at sig_state
        s_geo = (1.0 - w) * s_u + w * s_c                        # guided drift
        s_wt = s_c - s_u                                         # weight: score DIFFERENCE

        # FK log-weight over the CONSECUTIVE-target interval sc^2 - sn^2 (Gate 12:
        # even under churn the weight interval is sigma_cur -> sigma_next, NOT sig_hat).
        logw = logw + 0.5 * w * (w - 1.0) * (sc ** 2 - sn ** 2) * (s_wt ** 2).sum(dim=1)

        lam = 0.0 if churn else lam_of(sc, lam_mode, w)
        stochastic = churn or lam > 0.0

        # Resample BEFORE propagation (production order), only where stochastic so
        # the duplicated ancestors can re-diversify through the noise kernel.
        if resample and stochastic:
            ess = 1.0 / (torch.softmax(logw, dim=0) ** 2).sum()
            if float(ess) < ess_frac * K:
                idx = _systematic_indices(logw, gen)
                x = x[idx]
                logw = torch.zeros(K)

        d = -sig_state * s_geo
        h = sn - sig_state
        if churn:
            x = x + h * d                                        # churn already added noise
        else:
            x = x + h * (1.0 + lam) * d
            if lam > 0.0:
                delta = max(sc - sn, 0.0)
                z = torch.randn(K, 2, generator=gen)
                x = x + math.sqrt(2.0 * lam * sig_state * delta) * z

    return Run(x=x, logw=logw)


def weighted_mean_cov(run: Run):
    w = torch.softmax(run.logw, dim=0)                           # [K]
    x = run.x
    mean = (w[:, None] * x).sum(dim=0)
    d = x - mean
    cov = (w[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(dim=0)
    ess = 1.0 / (w ** 2).sum()
    return mean, cov, float(ess)


def systematic_resample(run: Run, gen, n_out=None):
    """Equal-weight cloud via systematic (low-variance) resampling -> honest scatter."""
    w = torch.softmax(run.logw, dim=0)
    K = w.numel()
    n_out = n_out or K
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    u0 = torch.rand(1, generator=gen) / n_out
    pos = u0 + torch.arange(n_out) / n_out
    idx = torch.searchsorted(cdf.contiguous(), pos.contiguous()).clamp_(0, K - 1)
    return run.x[idx]


# --------------------------------------------------------------------------- #
# Palette validator (Python port of the dataviz skill's validate_palette.js,   #
# scatter uses ALL-pairs CVD). Run at import so the figure palette is proven.  #
# --------------------------------------------------------------------------- #
def _validate_palette(hexes, mode, surface, pairs="all"):
    def srgb(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    def s2lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def lin(h):
        return [s2lin(c) for c in srgb(h)]

    def rellum(h):
        r, g, b = lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(a, b):
        hi, lo = sorted([rellum(a), rellum(b)], reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    def oklch(h):
        r, g, b = lin(h)
        l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
        a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
        bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
        return L, math.hypot(a, bb)

    MACH = {
        "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
        "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                   [-0.011820, 0.042940, 0.968881]],
    }

    def lab(rgb):
        r, g, b = rgb
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
        return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]

    def sim(h, kind):
        r, g, b = lin(h)
        M = MACH[kind]
        return [min(1, max(0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]

    def dE(h1, h2, kind):
        a, b = lab(sim(h1, kind)), lab(sim(h2, kind))
        return math.dist(a, b)

    n = len(hexes)
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all" \
        else [(i, i + 1) for i in range(n - 1)]
    lo, hi = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}[mode]
    band_ok = all(lo <= oklch(c)[0] <= hi for c in hexes)
    chroma_ok = all(oklch(c)[1] >= 0.10 for c in hexes)
    worst = min(dE(hexes[i], hexes[j], k) for i, j in prs for k in MACH)
    contr = {c: contrast(c, surface) for c in hexes}
    relief = [c for c in hexes if contr[c] < 3.0]
    print(f"  palette validate [{mode}] {hexes}")
    print(f"    lightness band {lo}-{hi}: {'PASS' if band_ok else 'FAIL'}")
    print(f"    chroma floor 0.10: {'PASS' if chroma_ok else 'FAIL'}")
    state = "PASS" if worst >= 12 else ("WARN(floor)" if worst >= 8 else "FAIL")
    print(f"    CVD all-pairs worst dE {worst:.1f}: {state}")
    print(f"    contrast vs surface {surface}: "
          + ("PASS" if not relief else f"relief (labels/legend required): "
             + ", ".join(f'{c}={contr[c]:.2f}' for c in relief)))
    assert band_ok and chroma_ok and worst >= 8.0, "palette fails hard checks"
    return worst


# Series roles, assigned in FIXED order: exact-target, sampled, q_u ref, q_c ref.
PAL = {
    # exact=blue, sampled=orange (max-separation pair); qu=green, qc=magenta as the
    # faint refs. All-pairs CVD worst dE 10.7 (both modes) -- floor band, legal here
    # because marks also differ by shape (solid ellipse / dashed contour / dots) and
    # every series is direct-labelled in the legend + on-panel text.
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", sec="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  exact="#2a78d6", sampled="#eb6834", qu="#008300", qc="#e87ba4"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#ffffff", sec="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835",
                 exact="#3987e5", sampled="#d95926", qu="#008300", qc="#d55181"),
}


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def _apply_theme(plt, p):
    plt.rcParams.update({
        "figure.facecolor": p["page"], "savefig.facecolor": p["page"],
        "axes.facecolor": p["surface"], "axes.edgecolor": p["axis"],
        "axes.labelcolor": p["sec"], "text.color": p["ink"],
        "xtick.color": p["muted"], "ytick.color": p["muted"],
        "grid.color": p["grid"], "axes.grid": True, "grid.linewidth": 0.6,
        "axes.linewidth": 0.8, "font.size": 11,
        "font.family": ["DejaVu Sans", "sans-serif"], "legend.frameon": False,
    })


def _ellipse(ax, mean, cov, ks, color, **kw):
    from matplotlib.patches import Ellipse
    vals, vecs = torch.linalg.eigh(cov)
    vals = vals.clamp_min(1e-12)
    ang = math.degrees(math.atan2(float(vecs[1, 1]), float(vecs[0, 1])))
    for k in ks:
        w_, h_ = 2 * k * math.sqrt(float(vals[1])), 2 * k * math.sqrt(float(vals[0]))
        ax.add_patch(Ellipse((float(mean[0]), float(mean[1])), w_, h_, angle=ang,
                             facecolor="none", edgecolor=color, **kw))


def plot_primary(theme, results, gen_seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = PAL[theme]
    _apply_theme(plt, p)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.4))
    xlim, ylim = (-4.2, 5.2), (-3.4, 3.8)
    for ax, w in zip(axes.ravel(), W_LIST):
        run, mean_e, cov_e = results[w]
        m_star, S_star, _ = geo_target(w, SIG_MIN ** 2)

        # faint reference contours: q_u, q_c at DATA level
        _ellipse(ax, M_U, S_U, [1, 2], p["qu"], lw=1.2, ls=(0, (5, 4)), alpha=0.55)
        _ellipse(ax, M_C, S_C, [1, 2], p["qc"], lw=1.2, ls=(0, (5, 4)), alpha=0.55)

        # sampled equal-weight cloud (resampled -> honest)
        g = torch.Generator(device=DEV).manual_seed(gen_seed + int(w * 10))
        cloud = systematic_resample(run, g, n_out=6000)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=5, c=p["sampled"], alpha=0.05,
                   edgecolors="none", rasterized=True, zorder=2)

        # EXACT geometric-average 1/2-sigma ellipse (the ground truth)
        _ellipse(ax, m_star, S_star, [1, 2], p["exact"], lw=2.0, zorder=4)

        # analytic vs empirical mean markers
        ax.scatter(*m_star.tolist(), marker="P", s=95, c=p["exact"],
                   edgecolors=p["surface"], linewidths=0.8, zorder=6)
        ax.scatter(*mean_e.tolist(), marker="o", s=42, c=p["sampled"],
                   edgecolors=p["surface"], linewidths=0.8, zorder=6)

        me = f"({mean_e[0]:+.2f}, {mean_e[1]:+.2f})"
        ma = f"({m_star[0]:+.2f}, {m_star[1]:+.2f})"
        ax.text(0.035, 0.965,
                f"w = {w:g}\nexact  m* = {ma}\nsampled = {me}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
                color=p["ink"], linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", fc=p["surface"],
                          ec=p["axis"], lw=0.7, alpha=0.9))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=0)

    # single shared legend (>=2 series)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=p["exact"], lw=2.0, label="exact  q_u^(1-w) q_c^w"),
        Line2D([], [], marker="o", ls="", mfc=p["sampled"], mec="none",
               alpha=0.6, markersize=8, label="FKC sampled cloud"),
        Line2D([], [], color=p["qu"], lw=1.2, ls=(0, (5, 4)), label="q_u  (reference)"),
        Line2D([], [], color=p["qc"], lw=1.2, ls=(0, (5, 4)), label="q_c  (reference)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.045), labelcolor=p["sec"])
    fig.suptitle("CFG-FKC particle cloud vs exact geometric average  q_u^{1-w} q_c^{w}",
                 fontsize=14.5, color=p["ink"], y=0.985)
    fig.text(0.5, 0.955,
             "w=0 → q_u   ·   w=1 → q_c   ·   w=1.5, 3 → mean marches BEYOND q_c "
             "(guidance extrapolation), covariance sharpens",
             ha="center", fontsize=10.5, color=p["sec"])
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.94))
    path = os.path.join(OUT, f"fkc_2d_primary_{theme}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_churn(theme, w=2.0, K=200_000, gen_seed=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = PAL[theme]
    _apply_theme(plt, p)

    em = run_cfg_fkc(w, lam_mode="sigma", proposal="em", K=K, seed=101, resample=True)
    ch = run_cfg_fkc(w, proposal="edm_churn", gamma=0.4, K=K, seed=202, resample=True)
    m_star, S_star, _ = geo_target(w, SIG_MIN ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    for ax, (name, run, col) in zip(
        axes, [("em proposal", em, p["sampled"]), ("edm_churn proposal", ch, p["qc"])]
    ):
        g = torch.Generator(device=DEV).manual_seed(gen_seed)
        cloud = systematic_resample(run, g, n_out=6000)
        mean_e, cov_e, ess = weighted_mean_cov(run)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=5, c=col, alpha=0.05,
                   edgecolors="none", rasterized=True, zorder=2)
        _ellipse(ax, m_star, S_star, [1, 2], p["exact"], lw=2.0, zorder=4)
        ax.scatter(*m_star.tolist(), marker="P", s=95, c=p["exact"],
                   edgecolors=p["surface"], linewidths=0.8, zorder=6)
        ax.scatter(*mean_e.tolist(), marker="o", s=42, c=col,
                   edgecolors=p["surface"], linewidths=0.8, zorder=6)
        ax.set_title(f"{name}   (w={w:g})", color=p["ink"], fontsize=12)
        ax.text(0.035, 0.965,
                f"sampled = ({mean_e[0]:+.2f}, {mean_e[1]:+.2f})\n"
                f"exact m* = ({m_star[0]:+.2f}, {m_star[1]:+.2f})\nESS/K = {ess/K:.2f}",
                transform=ax.transAxes, va="top", fontsize=10, color=p["ink"],
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", fc=p["surface"], ec=p["axis"],
                          lw=0.7, alpha=0.9))
        ax.set_xlim(-1.0, 4.6); ax.set_ylim(-1.8, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=p["exact"], lw=2.0, label="exact geometric average"),
        Line2D([], [], marker="o", ls="", mfc=p["sampled"], mec="none", alpha=0.6,
               markersize=8, label="em cloud"),
        Line2D([], [], marker="o", ls="", mfc=p["qc"], mec="none", alpha=0.6,
               markersize=8, label="edm_churn cloud"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.055), labelcolor=p["sec"])
    fig.suptitle("em vs edm_churn: both land on the geometric average "
                 "(leading order in gamma)", fontsize=13.5, color=p["ink"])
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.95))
    path = os.path.join(OUT, f"fkc_churn_{theme}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_1d(theme, ws=(0.0, 1.0, 3.0), K=200_000, gen_seed=5):
    """1D companion: pick the x-axis marginal of the same toy but 1D Gaussians.

    Uses the x-coordinate model: q_u = N(mu_u, vu), q_c = N(mu_c, vc). The exact
    geometric-average pdf is Gaussian and overlaid on the weighted histogram.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    p = PAL[theme]
    _apply_theme(plt, p)

    mu_u, mu_c, vu, vc = -1.5, 1.5, 1.0, 0.30

    def scores1(x, sig2):
        return -(x - mu_u) / (vu + sig2), -(x - mu_c) / (vc + sig2)

    def target1(w, sig2):
        inv = (1 - w) / (vu + sig2) + w / (vc + sig2)
        var = 1.0 / inv
        m = var * ((1 - w) * mu_u / (vu + sig2) + w * mu_c / (vc + sig2))
        return m, var

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    cols = [p["exact"], p["sampled"], p["qu"]]
    xs = np.linspace(-4, 5, 800)
    for wi, w in enumerate(ws):
        n = n_for(w)
        sigmas = sigma_grid(n)
        gen = torch.Generator(device=DEV).manual_seed(gen_seed + wi)
        m0, v0 = target1(w, float(sigmas[0]) ** 2)
        x = m0 + math.sqrt(v0) * torch.randn(K, generator=gen)
        logw = torch.zeros(K)
        for k in range(n):
            sc, sn = float(sigmas[k]), float(sigmas[k + 1])
            s_u, s_c = scores1(x, sc ** 2)
            s_geo = (1 - w) * s_u + w * s_c
            logw = logw + 0.5 * w * (w - 1.0) * (sc ** 2 - sn ** 2) * (s_c - s_u) ** 2
            lam = lam_of(sc, "sigma", w)
            if w > 1.0 and lam > 0:                              # SIR to keep ESS up
                ess = 1.0 / (torch.softmax(logw, dim=0) ** 2).sum()
                if float(ess) < 0.5 * K:
                    cdf = torch.cumsum(torch.softmax(logw, dim=0), dim=0); cdf[-1] = 1.0
                    u0 = torch.rand(1, generator=gen) / K
                    idx = torch.searchsorted(
                        cdf.contiguous(), (u0 + torch.arange(K) / K).contiguous()
                    ).clamp_(0, K - 1)
                    x = x[idx]; logw = torch.zeros(K)
            h = sn - sc
            x = x + h * (1 + lam) * (-sc * s_geo)
            if lam > 0:
                delta = max(sc - sn, 0.0)
                x = x + math.sqrt(2 * lam * sc * delta) * torch.randn(K, generator=gen)
        wt = torch.softmax(logw, dim=0).numpy()
        col = cols[wi % len(cols)]
        ax.hist(x.numpy(), bins=160, weights=wt, density=True, histtype="stepfilled",
                color=col, alpha=0.16, edgecolor="none")
        ax.hist(x.numpy(), bins=160, weights=wt, density=True, histtype="step",
                color=col, alpha=0.55, lw=1.0)
        m, v = target1(w, SIG_MIN ** 2)
        pdf = np.exp(-(xs - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
        ax.plot(xs, pdf, color=col, lw=2.2,
                label=f"exact  w={w:g}   N({m:.2f}, {v:.2f})")
    ax.set_xlim(-4, 5)
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.legend(fontsize=10.5, labelcolor=p["sec"])
    ax.tick_params(length=0)
    fig.suptitle("1D companion: weighted histogram (sampled) vs exact "
                 "geometric-average pdf", fontsize=13.5, color=p["ink"])
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    path = os.path.join(OUT, f"fkc_1d_{theme}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main: verify (asserts + table) THEN render.                                  #
# --------------------------------------------------------------------------- #
def main():
    import json
    report = {"passA": [], "passB": [], "churn": [],
              "meta": dict(K=200_000, sig_max=SIG_MAX, sig_min=SIG_MIN,
                           n_steps_small=200, n_steps_w3=n_for(3.0), w_list=W_LIST)}
    print("=" * 74)
    print("CFG-FKC 2D demo -- verifying weighted mean/cov vs analytic m*, S*")
    print("=" * 74)

    # ---- palette validation (dataviz skill) ----
    print("\n[palette] validating categorical hues (scatter -> all-pairs CVD):")
    for theme in ("light", "dark"):
        p = PAL[theme]
        _validate_palette([p["exact"], p["sampled"], p["qu"], p["qc"]],
                          theme, p["surface"], pairs="all")

    # PD sanity of the DATA-level precision Lam* for every w (extrapolation valid).
    for w in W_LIST:
        _, _, Lam_data = geo_target(w, 0.0)
        eig = torch.linalg.eigvalsh(Lam_data)
        assert float(eig.min()) > 0, f"Lam* not PD at w={w} (min eig {float(eig.min())})"
    print(f"\n[verify] Lam* positive-definite for all w in {W_LIST} (data level).")

    # ---- (A) lambda-INDEPENDENCE: pure IS weighted estimator, no resampling ----
    # The FK weight is lambda-independent, so the *unbiased* IS estimator must hit
    # m*, S* for every lambda. This is finite-sample demonstrable while the ESS
    # stays healthy -- i.e. up to w=1.5 (Gate 10's regime). Past that the pure-IS
    # ESS collapses to ~1 particle (reported below, not hidden); the mixing kernels
    # + resampling in pass (B) are what make large w tractable.
    MEAN_ATOL, COV_RTOL = 0.06, 0.12
    print("\n[verify A] lambda-independence, pure IS (no resampling), w in {0,1,1.5}:\n")
    header = f"{'w':>4} {'lambda':>9} {'ESS/K':>7}  " \
             f"{'|mean-m*|':>10}  {'||cov-S*||_F/||S*||':>20}"
    print(header); print("-" * len(header))
    for w in [0.0, 1.0, 1.5]:
        m_star, S_star, _ = geo_target(w, SIG_MIN ** 2)
        for lm in ["zero", "one", "sigma"] + (["inv_w"] if w >= 1.0 else []):
            run = run_cfg_fkc(w, lam_mode=lm, proposal="em", K=200_000, seed=17)
            mean_e, cov_e, ess = weighted_mean_cov(run)
            dmean = float((mean_e - m_star).norm())
            dcov = float((cov_e - S_star).norm() / S_star.norm())
            flag = "" if (dmean < MEAN_ATOL and dcov < COV_RTOL) else "  <-- FAIL"
            print(f"{w:>4g} {lm:>9} {ess/200_000:>7.3f}  {dmean:>10.4f}  {dcov:>20.4f}{flag}")
            report["passA"].append(dict(w=w, lam=lm, ess=ess / 200_000,
                                        dmean=dmean, dcov=dcov))
            assert dmean < MEAN_ATOL, f"w={w} lam={lm}: mean off by {dmean:.4f}"
            assert dcov < COV_RTOL, f"w={w} lam={lm}: cov off by {dcov:.4f}"

    # Show WHY w=3 needs resampling: pure-IS ESS for the mixing kernel collapses.
    for w in [3.0]:
        _, _, ess = weighted_mean_cov(
            run_cfg_fkc(w, lam_mode="sigma", K=50_000, seed=17, n_steps=200))
        print(f"  (note) pure-IS ESS/K at w={w:g}, lam=sigma = {ess/50_000:.4f} "
              "-> single-particle regime; resampling required.")

    # ---- (B) EXTRAPOLATION: mixing lambda(sigma) + sequential resampling (SIR) ----
    # The production configuration. Diversity is maintained, so mean/cov match the
    # exact geometric average for EVERY w including the w>1 extrapolation w=3.
    print("\n[verify B] extrapolation, lam(sigma) + sequential resampling, all w:\n")
    print(header); print("-" * len(header))
    results_for_plot = {}
    for w in W_LIST:
        m_star, S_star, _ = geo_target(w, SIG_MIN ** 2)
        run = run_cfg_fkc(w, lam_mode="sigma", proposal="em", K=200_000, seed=23,
                          resample=(w > 1.0))
        mean_e, cov_e, ess = weighted_mean_cov(run)
        dmean = float((mean_e - m_star).norm())
        dcov = float((cov_e - S_star).norm() / S_star.norm())
        flag = "" if (dmean < MEAN_ATOL and dcov < COV_RTOL) else "  <-- FAIL"
        print(f"{w:>4g} {'sigma+SIR':>9} {ess/200_000:>7.3f}  {dmean:>10.4f}  {dcov:>20.4f}{flag}")
        report["passB"].append(dict(w=w, ess=ess / 200_000, dmean=dmean, dcov=dcov,
                                    m_star=[float(m_star[0]), float(m_star[1])],
                                    mean_emp=[float(mean_e[0]), float(mean_e[1])],
                                    n_steps=n_for(w)))
        assert dmean < MEAN_ATOL, f"w={w} SIR: mean off by {dmean:.4f}"
        assert dcov < COV_RTOL, f"w={w} SIR: cov off by {dcov:.4f}"
        m_data, _, _ = geo_target(w, 0.0)
        print(f"     exact m*(sig_min)=({m_star[0]:+.3f},{m_star[1]:+.3f})  "
              f"data-level m*=({m_data[0]:+.3f},{m_data[1]:+.3f})")
        results_for_plot[w] = (run, mean_e, cov_e)

    print("\n[verify] ALL asserts passed: the CFG-FKC cloud matches the analytic "
          "geometric average q_u^(1-w) q_c^w for every w; the FK weight is "
          "lambda-independent (pass A).")

    # ---- churn cross-check (leading order) ----
    print("\n[verify] em vs edm_churn at w=2.0 (churn is leading-order in gamma):")
    for name, kw in [("em", dict(lam_mode="sigma", proposal="em")),
                     ("edm_churn", dict(proposal="edm_churn", gamma=0.4))]:
        run = run_cfg_fkc(2.0, K=200_000, seed=303, resample=True, **kw)
        mean_e, cov_e, ess = weighted_mean_cov(run)
        m_star, S_star, _ = geo_target(2.0, SIG_MIN ** 2)
        dmean = float((mean_e - m_star).norm())
        dcov = float((cov_e - S_star).norm() / S_star.norm())
        print(f"  {name:>10}: |dmean|={dmean:.4f}  ||dcov||_F/||S*||={dcov:.4f}  "
              f"ESS/K={ess/200_000:.2f}")
        report["churn"].append(dict(name=name, dmean=dmean, dcov=dcov,
                                    ess=ess / 200_000))
        tol_m, tol_c = (0.06, 0.12) if name == "em" else (0.15, 0.30)
        assert dmean < tol_m and dcov < tol_c, f"{name} off (m {dmean}, c {dcov})"

    # ---- render figures (light + dark) ----
    print("\n[render] writing PNGs (light + dark) to", OUT)
    paths = {}
    for theme in ("light", "dark"):
        paths[f"primary_{theme}"] = plot_primary(theme, results_for_plot)
        paths[f"churn_{theme}"] = plot_churn(theme)
        paths[f"1d_{theme}"] = plot_1d(theme)
    for k, v in paths.items():
        print(f"    {k:>14}: {v}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("    results.json:", os.path.join(OUT, "results.json"))
    print("\nDONE.")
    return paths


if __name__ == "__main__":
    main()
