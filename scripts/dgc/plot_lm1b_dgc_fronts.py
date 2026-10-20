"""Two clean LM1B GenPPL-entropy figures: the paper's CoBit curves plus one DGC-grid curve.

    python scripts/dgc/plot_lm1b_dgc_fronts.py <all_points.csv> <out_dir>

<all_points.csv> is the table written by plot_lm1b_dgc_sweep.py (nfe, s_churn, gamma, genppl, entropy).
The DGC point used at each NFE is fixed below by its S_churn, so the figures are reproducible:
  FRONT_384  NFE 64..384, the churn point nearest this run's real-data point at each NFE;
  FRONT_512  NFE 64..512, with lower-GenPPL choices at NFE 256 and 512.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FRONT_384 = {64: 3.0, 96: 10.0, 128: 11.0, 192: 30.0, 256: 51.0, 384: 40.0}
FRONT_512 = {64: 3.0, 96: 10.0, 128: 11.0, 192: 30.0, 256: 64.0, 384: 40.0, 512: 30.0}

here = os.path.dirname(os.path.abspath(__file__))
pts_csv, out_dir = sys.argv[1], sys.argv[2]
D = json.load(open(os.path.join(here, "lm1b_fig2_points.json")))
dgc = {}
for r in csv.DictReader(open(pts_csv)):
    if r["s_churn"] != "det":
        dgc[(int(r["nfe"]), float(r["s_churn"]))] = (float(r["entropy"]), float(r["genppl"]))

C_DET_LINE, C_DET, C_STO, C_DGC = "#377eb8", "#ff9d2e", "#d96cb8", "#111111"
C_REAL, C_AR, C_LF, C_FLM, C_DUO, C_MDLM = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
INK, MUTED = "#222222", "#555555"
# one colour per NFE from one ordered map (viridis, yellow end trimmed so faint points stay visible)
NFE_COL = {n: plt.cm.viridis(x) for n, x in zip((64, 96, 128, 192, 256, 384, 512), (0.0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84))}


def draw(front, out, label_off=None, cloud=False):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    ax.grid(True, color="#b0b0b0", lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    det, sto = D["cobit_deterministic"], D["cobit_stochastic"]
    ax.plot([p["entropy"] for p in det], [p["genppl"] for p in det], ls=(0, (3.7, 1.6)), lw=1.0, color=C_DET_LINE, zorder=2)
    ax.scatter([p["entropy"] for p in det], [p["genppl"] for p in det], marker="^", s=32, c=C_DET, edgecolors="black", linewidths=0.5, zorder=3)
    ax.plot([p["entropy"] for p in sto], [p["genppl"] for p in sto], lw=1.5, color=C_STO, zorder=2)
    ax.scatter([p["entropy"] for p in sto], [p["genppl"] for p in sto], marker="X", s=48, c=C_STO, edgecolors="black", linewidths=0.5, zorder=3)
    det_off = {64: (-14, 4), 96: (5, 0), 128: (5, -2), 192: (-16, 3), 256: (-17, 2), 384: (5, -7), 512: (-17, -7)}
    sto_off = {64: (6, 2), 96: (-6, -3), 128: (-6, -3), 192: (-6, -3), 256: (-6, -3), 384: (-6, -3), 512: (-6, -3)}
    for p in det:
        ax.annotate(str(p["nfe"]), (p["entropy"], p["genppl"]), xytext=det_off[p["nfe"]], textcoords="offset points", fontsize=6.8, color=MUTED)
    for p in sto:
        off = sto_off[p["nfe"]]
        ax.annotate(str(p["nfe"]), (p["entropy"], p["genppl"]), xytext=off, textcoords="offset points", fontsize=6.8, color=MUTED,
                    ha="left" if off[0] > 0 else "right")

    R = D["references"]
    ax.scatter([R["ar_baseline"]["entropy"]], [R["ar_baseline"]["genppl"]], marker="s", s=28, c=C_AR, zorder=4)
    ax.scatter([R["langflow_128"]["entropy"]], [R["langflow_128"]["genppl"]], marker="P", s=40, c=C_LF, edgecolors="black", linewidths=0.4, zorder=4)
    for key, col in (("flm_1024", C_FLM), ("duo", C_DUO), ("mdlm", C_MDLM)):
        ax.scatter([R[key]["entropy"]], [R[key]["genppl"]], marker="D", s=32, c=col, zorder=4)
    for key, txt, off in (("ar_baseline", "AR", (6, -3)), ("langflow_128", "LangFlow\n(128)", (22, -8)), ("flm_1024", "FLM (1024)", (6, -3)),
                          ("duo", "Duo", (-6, -3)), ("mdlm", "MDLM", (6, -3))):
        ax.annotate(txt, (R[key]["entropy"], R[key]["genppl"]), xytext=off, textcoords="offset points", fontsize=7, color=INK,
                    ha="left" if off[0] > 0 else "right")

    rr = D["this_run"]["real_data"]
    ax.scatter([rr["entropy"]], [rr["genppl"]], marker="D", s=46, c=C_REAL, edgecolors="black", linewidths=0.5, zorder=6)
    ax.annotate("Real data", (rr["entropy"], rr["genppl"]), xytext=(-8, -3), textcoords="offset points", fontsize=7.2, color=INK, ha="right")

    if cloud:  # every churn value tried, one colour per NFE (fixed across figures), faint
        for n in sorted({k[0] for k in dgc}):
            if n <= max(front):
                xs = [e for (m, _), (e, _) in dgc.items() if m == n]
                ys = [g for (m, _), (_, g) in dgc.items() if m == n]
                ax.scatter(xs, ys, marker="o", s=34, color=NFE_COL[n], alpha=0.42, edgecolors="none", zorder=4)
    pts = [(n, dgc[(n, s)]) for n, s in sorted(front.items())]
    ax.plot([e for _, (e, _) in pts], [g for _, (_, g) in pts], lw=1.8, color=C_DGC, zorder=5)
    ax.scatter([e for _, (e, _) in pts], [g for _, (_, g) in pts], marker="o", s=44, c=C_DGC, edgecolors="white", linewidths=1.0, zorder=6)
    for n, (e, g) in pts:
        off = (label_off or {}).get(n, (7, -3))
        ax.annotate(str(n), (e, g), xytext=off, textcoords="offset points", fontsize=7, color=C_DGC, fontweight="bold",
                    ha="left" if off[0] > 0 else "right")

    ax.set_xlim(4.24, 4.365)
    ax.set_ylim(42, 113)
    ax.set_xlabel("Token-frequency entropy")
    ax.set_ylabel("Generative PPL (lower is better)")
    ax.set_title("LM1B: GenPPL–entropy comparison")
    handles = [
        Line2D([], [], marker="D", ls="", mfc=C_REAL, mec="black", mew=0.5, ms=6, label="Real data"),
        Line2D([], [], marker="s", ls="", mfc=C_AR, mec=C_AR, ms=5, label="AR baseline"),
        Line2D([], [], marker="P", ls="", mfc=C_LF, mec="black", mew=0.4, ms=6, label="LangFlow"),
        Line2D([], [], marker="D", ls="", mfc=C_FLM, mec=C_FLM, ms=5.5, label="Other reported DLMs"),
        Line2D([], [], marker="^", color=C_DET_LINE, ls=(0, (3.7, 1.6)), mfc=C_DET, mec="black", mew=0.5, ms=6, label="Ours deterministic"),
        Line2D([], [], marker="X", color=C_STO, mfc=C_STO, mec="black", mew=0.5, ms=7, label="Ours stochastic"),
        Line2D([], [], marker="o", color=C_DGC, mfc=C_DGC, mec="white", ms=6.5, lw=1.8, label="Ours stochastic, DGC grid"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.95, edgecolor="#cccccc")
    if cloud:
        ax.add_artist(leg)
        nfe_handles = [Line2D([], [], marker="o", ls="", color=NFE_COL[n], alpha=0.6, ms=6, label=f"NFE {n}")
                       for n in sorted(NFE_COL) if n <= max(front)]
        ax.legend(handles=nfe_handles, title="DGC, all γ", title_fontsize=7, loc="lower left", fontsize=7, ncol=2,
                  framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
        print("wrote", f"{out}.{ext}")
    plt.close(fig)


draw(FRONT_384, os.path.join(out_dir, "lm1b_dgc_front_nfe384"))
draw(FRONT_512, os.path.join(out_dir, "lm1b_dgc_front_nfe512"), label_off={256: (-7, -3)})
draw(FRONT_384, os.path.join(out_dir, "lm1b_dgc_front_nfe384_cloud"), cloud=True)
draw(FRONT_512, os.path.join(out_dir, "lm1b_dgc_front_nfe512_cloud"), label_off={256: (-7, -3)}, cloud=True)
