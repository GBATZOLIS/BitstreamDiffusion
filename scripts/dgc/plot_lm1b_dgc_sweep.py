"""Paper LM1B GenPPL-entropy figure + the DGC-grid sweep (deterministic and churn, per NFE).

    python scripts/dgc/plot_lm1b_dgc_sweep.py <out_prefix> <results_root> [<results_root> ...]

Reads every <root>/**/dgc_ab/dgc/seed*/results.jsonl (long-format rows from run_eval and
compute_entropy_from_caches), parses NFE and S_churn from the spec tag, and draws:
  * the paper's CoBit curves and baselines (lm1b_fig2_points.json, paper real-data point omitted);
  * the DGC deterministic curve (dashed, hollow triangles);
  * the DGC churn points, one thin line per NFE through its S_churn values, shaded by NFE;
  * this run's real-data point and the same-pipeline CoBit-grid point (NFE 256, gamma 0.200).
Also writes <out_prefix>.csv with every DGC point.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here = os.path.dirname(os.path.abspath(__file__))
out, roots = sys.argv[1], sys.argv[2:]
D = json.load(open(os.path.join(here, "lm1b_fig2_points.json")))

TAG = re.compile(r"tnfe-(\d+)_.*?term-1\.10(?:_stoch-ch([\d.]+)-)?")
pts = defaultdict(dict)  # (nfe, s_churn or None) -> {metric: value}
for root in roots:
    for f in glob.glob(f"{root}/**/dgc_ab/dgc/seed*/results.jsonl", recursive=True):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            m = TAG.search(str(r.get("tag", "")))
            if not m or r.get("metric") not in ("gen_full_external_ppl", "gen_full_token_unigram_entropy"):
                continue
            key = (int(m.group(1)), float(m.group(2)) if m.group(2) else None)
            pts[key]["ppl" if "ppl" in r["metric"] else "ent"] = float(r["value"])
pts = {k: v for k, v in pts.items() if "ppl" in v and "ent" in v}
nfes = sorted({k[0] for k in pts})
with open(f"{out}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["nfe", "s_churn", "gamma", "genppl", "entropy"])
    for (n, s), v in sorted(pts.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        w.writerow([n, "det" if s is None else s, 0 if s is None else round(s / (n - 1), 4), round(v["ppl"], 3), round(v["ent"], 4)])
print(f"{len(pts)} DGC points over NFEs {nfes}; wrote {out}.csv")

C_DET_LINE, C_DET, C_STO = "#377eb8", "#ff9d2e", "#d96cb8"
C_REAL, C_AR, C_LF, C_FLM, C_DUO, C_MDLM = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
INK, MUTED = "#222222", "#555555"
# DGC in one strong ink with NFE text labels, the convention the paper uses for its CoBit curves
C_DGC = "#111111"
ncol = {n: C_DGC for n in nfes}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200)
ax.grid(True, color="#b0b0b0", lw=0.6, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# paper: CoBit curves (kept recessive so the DGC data reads first)
det, sto = D["cobit_deterministic"], D["cobit_stochastic"]
ax.plot([p["entropy"] for p in det], [p["genppl"] for p in det], ls=(0, (3.7, 1.6)), lw=1.0, color=C_DET_LINE, alpha=0.7, zorder=2)
ax.scatter([p["entropy"] for p in det], [p["genppl"] for p in det], marker="^", s=26, c=C_DET, edgecolors="black", linewidths=0.4, alpha=0.8, zorder=3)
ax.plot([p["entropy"] for p in sto], [p["genppl"] for p in sto], lw=1.5, color=C_STO, alpha=0.8, zorder=2)
ax.scatter([p["entropy"] for p in sto], [p["genppl"] for p in sto], marker="X", s=40, c=C_STO, edgecolors="black", linewidths=0.4, zorder=3)
for p in sto:
    off = (6, 2) if p["nfe"] == 64 else (-6, -3)
    ax.annotate(str(p["nfe"]), (p["entropy"], p["genppl"]), xytext=off, textcoords="offset points", fontsize=6.5, color=C_STO,
                ha="left" if off[0] > 0 else "right")
R = D["references"]
ax.scatter([R["ar_baseline"]["entropy"]], [R["ar_baseline"]["genppl"]], marker="s", s=24, c=C_AR, zorder=4)
ax.scatter([R["langflow_128"]["entropy"]], [R["langflow_128"]["genppl"]], marker="P", s=34, c=C_LF, edgecolors="black", linewidths=0.4, zorder=4)
for key, col in (("flm_1024", C_FLM), ("duo", C_DUO), ("mdlm", C_MDLM)):
    ax.scatter([R[key]["entropy"]], [R[key]["genppl"]], marker="D", s=28, c=col, zorder=4)
for key, txt, off in (("ar_baseline", "AR", (6, -3)), ("langflow_128", "LangFlow (128)", (-6, 4)), ("flm_1024", "FLM (1024)", (6, -3)),
                      ("duo", "Duo", (-6, -3)), ("mdlm", "MDLM", (6, -3))):
    ax.annotate(txt, (R[key]["entropy"], R[key]["genppl"]), xytext=off, textcoords="offset points", fontsize=6.8, color=MUTED,
                ha="right" if off[0] < 0 else "left")

# DGC deterministic curve
dd = [(n, pts[(n, None)]) for n in nfes if (n, None) in pts]
ax.plot([v["ent"] for _, v in dd], [v["ppl"] for _, v in dd], ls=(0, (3.7, 1.6)), lw=1.2, color=C_DGC, zorder=4)
for n, v in dd:
    ax.scatter([v["ent"]], [v["ppl"]], marker="^", s=40, facecolors="white", edgecolors=ncol[n], linewidths=1.4, zorder=5)

# DGC churn points: every gamma as a faint per-NFE line; the lowest-GenPPL point per NFE forms the DGC frontier
best = []
for n in nfes:
    ss = sorted(((s, v) for (m, s), v in pts.items() if m == n and s is not None), key=lambda t: t[0])
    if not ss:
        continue
    ax.plot([v["ent"] for _, v in ss], [v["ppl"] for _, v in ss], lw=0.8, color="#9a9a9a", alpha=0.6, zorder=4)
    ax.scatter([v["ent"] for _, v in ss], [v["ppl"] for _, v in ss], marker="o", s=16, c="#8a8a8a", edgecolors="white", linewidths=0.5, zorder=4)
    best.append((n,) + min(ss, key=lambda t: t[1]["ppl"]))
ax.plot([v["ent"] for _, _, v in best], [v["ppl"] for _, _, v in best], lw=2.0, color=C_DGC, zorder=6)
ax.scatter([v["ent"] for _, _, v in best], [v["ppl"] for _, _, v in best], marker="o", s=52, c=C_DGC, edgecolors="white", linewidths=1.2, zorder=7)
BEST_OFF = {256: (-8, -13)}  # label placement tweaks where fronts crowd
for n, s_b, v in best:
    off = BEST_OFF.get(n, (7, -3))
    ax.annotate(f"{n} (γ {s_b / (n - 1):.2f})", (v["ent"], v["ppl"]), xytext=off, textcoords="offset points", fontsize=7, color=C_DGC,
                fontweight="bold", ha="right" if off[0] < 0 else "left")

# DGC near-real front: per NFE, the churn point closest to this run's real-data point, with
# ENT_RATE GenPPL points per nat of entropy (default 500: 0.01 nats counts as 5 GenPPL points)
ENT_RATE = float(os.environ.get("ENT_RATE", 500))
C_NEAR = "#0f766e"
real_ppl, real_ent = D["this_run"]["real_data"]["genppl"], D["this_run"]["real_data"]["entropy"]
near = []
for n in nfes:
    ss = [(s, v) for (m, s), v in pts.items() if m == n and s is not None]
    if ss:
        near.append((n,) + min(ss, key=lambda t: ((t[1]["ppl"] - real_ppl) ** 2 + (ENT_RATE * (t[1]["ent"] - real_ent)) ** 2) ** 0.5))
ax.plot([v["ent"] for _, _, v in near], [v["ppl"] for _, _, v in near], lw=1.8, color=C_NEAR, zorder=6)
ax.scatter([v["ent"] for _, _, v in near], [v["ppl"] for _, _, v in near], marker="s", s=95, facecolors="none", edgecolors=C_NEAR, linewidths=1.7, zorder=8)
best_keys = {(n, s_b) for n, s_b, _ in best}
for n, s_b, v in near:
    if (n, s_b) not in best_keys:
        off = {256: (-4, 12), 384: (9, -13)}.get(n, (-8, 4))
        ax.annotate(f"{n} (γ {s_b / (n - 1):.2f})", (v["ent"], v["ppl"]), xytext=off, textcoords="offset points", fontsize=7,
                    color=C_NEAR, fontweight="bold", ha="right" if off[0] < 0 else "left")
with open(f"{out}_fronts.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["front", "nfe", "s_churn", "gamma", "genppl", "entropy"])
    for name, rows in (("best_genppl", best), ("near_real", near)):
        for n, s_b, v in rows:
            w.writerow([name, n, s_b, round(s_b / (n - 1), 4), round(v["ppl"], 3), round(v["ent"], 4)])

# same-pipeline anchors from this run
T = D["this_run"]
rr, cb = T["real_data"], T["cobit_grid"]
ax.scatter([rr["entropy"]], [rr["genppl"]], marker="D", s=46, facecolors="white", edgecolors=C_REAL, linewidths=1.6, zorder=7)
ax.annotate("Real data", (rr["entropy"], rr["genppl"]), xytext=(-46, -3), textcoords="offset points", fontsize=7.2, color=INK)
ax.scatter([cb["entropy"]], [cb["genppl"]], marker="*", s=170, c=C_STO, edgecolors="white", linewidths=1.0, zorder=7)
ax.annotate("CoBit grid, this run\n(NFE 256, γ 0.200)", (cb["entropy"], cb["genppl"]), xytext=(-86, 4), textcoords="offset points", fontsize=6.8, color=INK)

allx = [v["ent"] for v in pts.values()] + [p["entropy"] for p in det + sto] + [rr["entropy"], R["mdlm"]["entropy"]]
ally = [v["ppl"] for v in pts.values()] + [p["genppl"] for p in det + sto] + [R["mdlm"]["genppl"]]
ax.set_xlim(min(4.24, min(allx) - 0.004), max(4.35, max(allx) + 0.006))
ax.set_ylim(min(42, min(ally) - 3), max(113, max(ally) + 4))
ax.set_xlabel("Token-frequency entropy")
ax.set_ylabel("Generative PPL (lower is better)")
ax.set_title("LM1B: CoBit (paper) vs the DGC σ-grid, NFE " + "/".join(str(n) for n in nfes))

handles = [
    Line2D([], [], marker="^", color=C_DET_LINE, ls=(0, (3.7, 1.6)), mfc=C_DET, mec="black", mew=0.4, ms=5.5, label="CoBit deterministic (paper)"),
    Line2D([], [], marker="X", color=C_STO, mfc=C_STO, mec="black", mew=0.4, ms=6.5, label="CoBit stochastic frontier (paper)"),
    Line2D([], [], marker="^", color=C_DGC, ls=(0, (3.7, 1.6)), mfc="white", mec=C_DGC, mew=1.4, ms=6, label="DGC deterministic"),
    Line2D([], [], marker="o", color=C_DGC, mfc=C_DGC, mec="white", ms=6, lw=2, label="DGC best per NFE (lowest GenPPL)"),
    Line2D([], [], marker="s", color=C_NEAR, mfc="none", mec=C_NEAR, mew=1.7, ms=8, lw=1.8, label="DGC nearest real data (0.01 nats = 5 GenPPL)"),
    Line2D([], [], marker="o", color="#9a9a9a", mfc="#8a8a8a", mec="white", ms=4, lw=0.8, label="DGC, every churn value tried"),
    Line2D([], [], marker="*", ls="", mfc=C_STO, mec="white", ms=11, label="CoBit grid, this run"),
    Line2D([], [], marker="D", ls="", mfc="white", mec=C_REAL, mew=1.6, ms=6, label="Real data (LM1B test, this run)"),
    Line2D([], [], marker="D", ls="", mfc=C_FLM, mec=C_FLM, ms=5, label="Reported baselines (paper)"),
]
ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.95, edgecolor="#cccccc")
fig.text(0.01, 0.005,
         "Paper points read from Figure 2's vector paths (paper real-data point omitted). DGC: released CoBit-S checkpoint, only the σ grid changed;\n"
         "seed 42, 1024 samples, S_noise 1.003, full-band churn, γ = S_churn/(NFE−1); grey: every churn value tried. A10G, bf16.\n"
         f"Teal front: per NFE, the churn point closest to the real-data point, counting 0.01 nats of entropy as {ENT_RATE * 0.01:g} GenPPL points.",
         fontsize=6.3, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
    print("wrote", f"{out}.{ext}")
