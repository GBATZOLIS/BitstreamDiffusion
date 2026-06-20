#!/usr/bin/env python3
"""Aggregate the 250K CFG NFE x gamma x guidance grid into readable tables."""
import json, glob, os, sys

OUT_DIR = "runs/tasks/tinygsm/cobit_raw_binary_bits_cfg/gsm8k_eval_grid250k"

rows = []
for f in glob.glob(os.path.join(OUT_DIR, "gsm8k_results_*.json")):
    d = json.load(open(f))
    rows.append((int(d["steps"]), float(d["gamma"]), float(d["guidance_scale"]),
                 100.0 * d["accuracy"], int(d["num_correct"]), int(d["num_examples"])))

if not rows:
    print("no result files yet in", OUT_DIR); sys.exit(0)

nfes   = sorted({r[0] for r in rows})
gammas = sorted({r[1] for r in rows})
ws     = sorted({r[2] for r in rows})
acc = {(n, g, w): a for n, g, w, a, c, ne in rows}
nex = {(n, g, w): ne for n, g, w, a, c, ne in rows}

print(f"\n250K CFG grid  ({len(rows)} configs, n_examples={sorted(set(nex.values()))})")
print("Accuracy % by (NFE, gamma) x guidance w.  [d] = delta vs w=0 control.\n")
for n in nfes:
    print(f"=== NFE {n} ===")
    header = "  gamma |" + "".join(f"  w={w:<4}" for w in ws)
    print(header); print("  " + "-" * (len(header) - 2))
    for g in gammas:
        cells = []
        base = acc.get((n, g, 0.0))
        for w in ws:
            a = acc.get((n, g, w))
            if a is None:
                cells.append("   --  ")
            elif w == 0.0 or base is None:
                cells.append(f" {a:5.2f} ")
            else:
                cells.append(f"{a:5.2f}{a-base:+4.1f}")
        print(f"  {g:<5} |" + "".join(f" {c}" for c in cells))
    print()

best = max(rows, key=lambda r: r[3])
print(f"BEST: {best[3]:.2f}%  (NFE {best[0]}, gamma {best[1]}, w {best[2]}, {best[4]}/{best[5]})")
print("Reference: unguided 250K BASE run best = 23.43% (NFE512, gamma0.34).")
