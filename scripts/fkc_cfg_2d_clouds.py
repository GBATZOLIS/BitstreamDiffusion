#!/usr/bin/env python
"""Companion figure: show the INGREDIENTS as real point clouds.

For each guidance weight w, overlay:
  - q_u sampled cloud   (green)   -- the unconditional source
  - q_c sampled cloud   (magenta) -- the conditional source
  - FKC final cloud     (orange)  -- the resampled particle result
  - exact geometric avg (blue)    -- the theoretical target ellipse + mean

Run: CUDA_VISIBLE_DEVICES="" python scripts/fkc_cfg_2d_clouds.py
"""
import math
import os
import sys
import importlib.util

import torch

_SPEC = importlib.util.spec_from_file_location(
    "demo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fkc_cfg_2d_demo.py"))
demo = importlib.util.module_from_spec(_SPEC)
sys.modules["demo"] = demo
_SPEC.loader.exec_module(demo)

OUT = demo.OUT
PAL = demo.PAL
W_LIST = [0.0, 1.0, 1.5, 3.0]


def _sample_gaussian(m, S, K, gen):
    L = torch.linalg.cholesky(S)
    return m + torch.randn(K, 2, generator=gen) @ L.T


def plot_clouds(theme, K=200_000, n_show=5000, seed=41):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    p = PAL[theme]
    demo._apply_theme(plt, p)

    gen = torch.Generator().manual_seed(seed)
    qu_cloud = _sample_gaussian(demo.M_U, demo.S_U, n_show, gen)   # data-level ingredients
    qc_cloud = _sample_gaussian(demo.M_C, demo.S_C, n_show, gen)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.4))
    xlim, ylim = (-4.4, 5.4), (-3.4, 3.8)
    for ax, w in zip(axes.ravel(), W_LIST):
        # FKC final cloud (resample the mixing-kernel run to equal weight)
        run = demo.run_cfg_fkc(w, lam_mode="sigma", proposal="em", K=K,
                               seed=23, resample=(w > 1.0))
        g2 = torch.Generator().manual_seed(seed + 7)
        fkc = demo.systematic_resample(run, g2, n_out=n_show)
        m_star, S_star, _ = demo.geo_target(w, demo.SIG_MIN ** 2)

        # ingredient clouds first (recessive backdrop)
        ax.scatter(qu_cloud[:, 0], qu_cloud[:, 1], s=4, c=p["qu"], alpha=0.045,
                   edgecolors="none", rasterized=True, zorder=1)
        ax.scatter(qc_cloud[:, 0], qc_cloud[:, 1], s=4, c=p["qc"], alpha=0.045,
                   edgecolors="none", rasterized=True, zorder=1)
        # FKC result on top
        ax.scatter(fkc[:, 0], fkc[:, 1], s=5, c=p["sampled"], alpha=0.06,
                   edgecolors="none", rasterized=True, zorder=3)
        # theoretical target
        demo._ellipse(ax, m_star, S_star, [1, 2], p["exact"], lw=2.0, zorder=5)
        ax.scatter(*m_star.tolist(), marker="P", s=95, c=p["exact"],
                   edgecolors=p["surface"], linewidths=0.8, zorder=6)
        # source means (faint anchors)
        for mm, cc in [(demo.M_U, p["qu"]), (demo.M_C, p["qc"])]:
            ax.scatter(*mm.tolist(), marker="x", s=46, c=cc, linewidths=1.6,
                       alpha=0.9, zorder=4)

        tag = {0.0: "= q_u", 1.0: "= q_c", 1.5: "beyond q_c",
               3.0: "far beyond q_c"}[w]
        ax.text(0.035, 0.965, f"w = {w:g}   ({tag})",
                transform=ax.transAxes, va="top", fontsize=12, color=p["ink"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc=p["surface"], ec=p["axis"],
                          lw=0.7, alpha=0.9))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=0)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=p["qu"], mec="none", alpha=0.7,
               markersize=8, label="q_u cloud  (unconditional)"),
        Line2D([], [], marker="o", ls="", mfc=p["qc"], mec="none", alpha=0.7,
               markersize=8, label="q_c cloud  (conditional)"),
        Line2D([], [], marker="o", ls="", mfc=p["sampled"], mec="none", alpha=0.7,
               markersize=8, label="FKC final cloud"),
        Line2D([], [], color=p["exact"], lw=2.0, label="exact geometric average"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.045), labelcolor=p["sec"])
    fig.suptitle("Ingredients and result: two source clouds -> the FKC cloud "
                 "on the geometric average", fontsize=14, color=p["ink"], y=0.985)
    fig.text(0.5, 0.955,
             "green = q_u,  magenta = q_c  (the two data clouds);  orange = FKC "
             "result;  blue = exact target.  The result marches q_u -> q_c -> beyond.",
             ha="center", fontsize=10.5, color=p["sec"])
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.94))
    path = os.path.join(OUT, f"fkc_clouds_{theme}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_match(theme, K=200_000, n_show=6000, seed=41):
    """Zoomed 'does it coincide?' view: per-w, the FKC cloud + the theoretical
    ellipse (solid) + the cloud's OWN empirical ellipse (dashed). Dashed on solid
    == coincidence, unambiguous even at high w where the cloud is small."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    p = PAL[theme]
    demo._apply_theme(plt, p)

    gen0 = torch.Generator().manual_seed(seed)
    qu_cloud = _sample_gaussian(demo.M_U, demo.S_U, n_show, gen0)   # source ingredients
    qc_cloud = _sample_gaussian(demo.M_C, demo.S_C, n_show, gen0)

    def _source_marker(ax, m, col, label, xlim, ylim):
        """Draw the source mean; if it is off-panel, point to it from the edge."""
        mx, my = float(m[0]), float(m[1])
        if xlim[0] <= mx <= xlim[1] and ylim[0] <= my <= ylim[1]:
            ax.scatter(mx, my, marker="x", s=52, c=col, linewidths=1.8, zorder=7)
            ax.annotate(label, (mx, my), textcoords="offset points",
                        xytext=(7, 6), color=col, fontsize=10, fontweight="bold")
            return
        cx, cy = sum(xlim) / 2, sum(ylim) / 2
        dx, dy = mx - cx, my - cy
        tx = (xlim[1] - cx) / dx if dx > 0 else (xlim[0] - cx) / dx if dx < 0 else 9e9
        ty = (ylim[1] - cy) / dy if dy > 0 else (ylim[0] - cy) / dy if dy < 0 else 9e9
        t = min(tx, ty) * 0.9
        ex, ey = cx + dx * t, cy + dy * t
        ax.annotate(f"{label} →" if dx > 0 else f"← {label}", xy=(ex, ey),
                    xytext=(ex - 0.28 * dx / abs(dx or 1), ey - 0.28 * dy / abs(dy or 1)),
                    color=col, fontsize=10, fontweight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4), zorder=7)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.4))
    for ax, w in zip(axes.ravel(), W_LIST):
        run = demo.run_cfg_fkc(w, lam_mode="sigma", proposal="em", K=K,
                               seed=23, resample=(w > 1.0))
        g2 = torch.Generator().manual_seed(seed + 7)
        cloud = demo.systematic_resample(run, g2, n_out=n_show)
        m_star, S_star, _ = demo.geo_target(w, demo.SIG_MIN ** 2)
        mean_e = cloud.mean(0)
        cov_e = cloud.T.cov()

        # per-panel window: cover the target's 2.6-sigma AND keep the nearby q_c
        # cloud in frame; q_u is marked with an edge arrow when it falls outside.
        cx, cy = float(m_star[0]), float(m_star[1])
        wt = 2.6 * float(torch.linalg.eigvalsh(S_star).max().sqrt())
        wc = 1.4 * float(torch.linalg.eigvalsh(demo.S_C).max().sqrt())
        R = max(wt, abs(float(demo.M_C[0]) - cx) + wc,
                abs(float(demo.M_C[1]) - cy) + wc)
        xlim, ylim = (cx - R, cx + R), (cy - R, cy + R)

        # source clouds (recessive backdrop) + FKC result on top
        ax.scatter(qu_cloud[:, 0], qu_cloud[:, 1], s=4, c=p["qu"], alpha=0.05,
                   edgecolors="none", rasterized=True, zorder=1)
        ax.scatter(qc_cloud[:, 0], qc_cloud[:, 1], s=4, c=p["qc"], alpha=0.05,
                   edgecolors="none", rasterized=True, zorder=1)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=6, c=p["sampled"], alpha=0.07,
                   edgecolors="none", rasterized=True, zorder=2)
        # theoretical (solid) and empirical (dashed) ellipses, same sigma levels
        demo._ellipse(ax, m_star, S_star, [1, 2], p["exact"], lw=2.2, zorder=4)
        demo._ellipse(ax, mean_e, cov_e, [1, 2], p["ink"], lw=1.5,
                      ls=(0, (4, 3)), zorder=5, alpha=0.85)
        ax.scatter(*m_star.tolist(), marker="P", s=110, c=p["exact"],
                   edgecolors=p["surface"], linewidths=0.9, zorder=6)
        ax.scatter(*mean_e.tolist(), marker="o", s=44, c=p["sampled"],
                   edgecolors=p["surface"], linewidths=0.9, zorder=6)
        _source_marker(ax, demo.M_U, p["qu"], "q_u", xlim, ylim)
        _source_marker(ax, demo.M_C, p["qc"], "q_c", xlim, ylim)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=0)

        dmean = float((mean_e - m_star).norm())
        dcov = float((cov_e - S_star).norm() / S_star.norm())
        ax.text(0.04, 0.96,
                f"w = {w:g}\n|mean − m*| = {dmean:.3f}\n‖Δcov‖/‖S*‖ = {dcov:.3f}",
                transform=ax.transAxes, va="top", fontsize=11, color=p["ink"],
                linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.4", fc=p["surface"], ec=p["axis"],
                          lw=0.7, alpha=0.92))

    handles = [
        Line2D([], [], marker="o", ls="", mfc=p["qu"], mec="none", alpha=0.7,
               markersize=8, label="q_u cloud"),
        Line2D([], [], marker="o", ls="", mfc=p["qc"], mec="none", alpha=0.7,
               markersize=8, label="q_c cloud"),
        Line2D([], [], marker="o", ls="", mfc=p["sampled"], mec="none", alpha=0.7,
               markersize=8, label="FKC cloud"),
        Line2D([], [], color=p["exact"], lw=2.2, label="exact target (1σ/2σ)"),
        Line2D([], [], color=p["ink"], lw=1.5, ls=(0, (4, 3)),
               label="empirical ellipse of cloud (1σ/2σ)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.045), labelcolor=p["sec"])
    fig.suptitle("Coincidence check (zoomed per w): the cloud's own ellipse lands "
                 "on the exact target", fontsize=14, color=p["ink"], y=0.985)
    fig.text(0.5, 0.955,
             "dashed = ellipse fitted to the sampled cloud;  solid = exact analytic "
             "target.  They overlap at every w.",
             ha="center", fontsize=10.5, color=p["sec"])
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.94))
    path = os.path.join(OUT, f"fkc_match_{theme}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for theme in ("light", "dark"):
        print("wrote", plot_clouds(theme))
        print("wrote", plot_match(theme))
