"""Collect dgc_ab results into one table: mean ± sd over seeds per (arm, spec).

Reads every <root>/**/dgc_ab/<arm>/seed<k>/results.jsonl. The eval writes long-format rows
({"tag": ..., "metric": ..., "value": ...}):
  * evaluation.run_eval (external_ppl driver): gen_full_external_ppl / gen_full_external_bpt with
    the spec tag, and real_full_external_* with tag "REAL";
  * evaluation.compute_entropy_from_caches: gen_full_token_unigram_entropy and
    real_full_token_unigram_entropy, both with the spec tag.
Rows whose metric starts with "real_" go to one "real[<arm>]" line per arm (the real reference
should be identical across arms for the same seed); the rest are keyed by (arm, spec).
Also prints paired per-seed differences vs `cobit`.
"""
import glob
import json
import math
import re
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
files = sorted(glob.glob(f"{root}/**/dgc_ab/*/seed*/results.jsonl", recursive=True))
if not files:
    sys.exit(f"no results.jsonl under {root}/**/dgc_ab/")

KEEP = re.compile(r"ppl|bpt|entropy", re.I)
# Run tags come from evaluation/sampling_specs.py::_tag_for_spec, e.g.
# ddim_entropic_scr-carry_tnfe-256_anfe-256_steps-255_gs0.0_term-1.10[_<churn suffix>].
# Tags are identical across arms (the arm is the directory), so the tag tail is the spec key.
SPEC = re.compile(r"tnfe-\d+.*")

# (arm, spec) -> seed -> {metric: value}
data = defaultdict(lambda: defaultdict(dict))
for f in files:
    arm = f.split("/dgc_ab/")[1].split("/")[0]
    seed = re.search(r"seed(\d+)", f).group(1)
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        m, v = row.get("metric"), row.get("value")
        if not isinstance(m, str) or not isinstance(v, (int, float)) or not KEEP.search(m):
            continue
        if m.startswith("real_"):
            data[(f"real[{arm}]", "data")][seed][m] = float(v)
            continue
        tag = row.get("tag")
        sp = SPEC.search(tag).group(0) if isinstance(tag, str) and SPEC.search(tag) else None
        if sp is None:
            continue
        data[(arm, sp)][seed][m] = float(v)
        wt = row.get("generation_wall_time_sec")
        if "external_ppl" in m and isinstance(wt, (int, float)):
            data[(arm, sp)][seed]["generation_wall_time_sec"] = float(wt)

if not data:
    sys.exit(f"no ppl/entropy metric rows in {len(files)} results.jsonl files under {root}")

metrics = sorted({m for d in data.values() for s in d.values() for m in s})


def ms(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "—"
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return f"{mu:.3f} ± {sd:.3f} (n={len(vals)})"


print("\n## Mean ± sd over seeds\n")
print("| arm | spec | " + " | ".join(metrics) + " |")
print("|" + "---|" * (2 + len(metrics)))
for (arm, sp) in sorted(data):
    seeds = data[(arm, sp)]
    print(f"| {arm} | {sp} | " + " | ".join(ms([seeds[s].get(m) for s in seeds]) for m in metrics) + " |")

print("\n## Per seed\n")
for (arm, sp) in sorted(data):
    for s in sorted(data[(arm, sp)]):
        vals = data[(arm, sp)][s]
        print(f"{arm:14s} {sp:60s} seed{s} " + " ".join(f"{m}={vals[m]:.4f}" for m in metrics if m in vals))

print("\n## Paired difference vs cobit (same seed): arm − cobit\n")
for (arm, sp) in sorted(data):
    if arm == "cobit" or arm.startswith("real[") or ("cobit", sp) not in data:
        continue
    base = data[("cobit", sp)]
    for m in metrics:
        if m == "generation_wall_time_sec":
            continue
        diffs = [data[(arm, sp)][s][m] - base[s][m] for s in data[(arm, sp)]
                 if s in base and m in base[s] and m in data[(arm, sp)][s]]
        if diffs:
            print(f"{arm:8s} {sp:60s} {m:32s} {ms(diffs)}  per-seed: {[round(d, 3) for d in diffs]}")
