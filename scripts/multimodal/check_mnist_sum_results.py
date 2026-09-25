#!/usr/bin/env python
"""
Check a MNIST-Sum results.json against the numbers reported in the paper.

Usage:
    python scripts/multimodal/check_mnist_sum_results.py results/<out>/results.json

Exits 0 if every reported cell reproduces, 1 otherwise. Reproduction here means
"inside the published 95% Wilson interval", not "bit-identical": sampling is
stochastic (EDM churn at gamma=0.175 for image->text) and cuDNN/​GPU differences
move the last decimal. A run that lands outside the interval is a real
discrepancy and worth reporting as an issue.
"""
from __future__ import annotations

import argparse
import json
import sys

# §4.2 and Appendix B.2 of the paper. Each entry:
#   (path in results.json, reported value, published 95% CI)
PAPER = [
    ("image->text, exact (val)",
     ("i2t", "val_gs2", "end_to_end"), 97.56, (97.04, 97.99)),
    ("image->text, exact (held-out)",
     ("i2t", "val_holdout_gs2", "end_to_end"), 97.85, (97.36, 98.25)),
    ("image->text, perception (val)",
     ("i2t", "val_gs2", "perception"), 97.66, (97.15, 98.08)),
    ("image->text, arithmetic (val)",
     ("i2t", "val_gs2", "arithmetic"), 99.90, (99.75, 99.96)),
    ("image->text, malformed (val)",
     ("i2t", "val_gs2", "malformed"), 0.00, (0.00, 0.09)),
    ("text->image, all four (val)",
     ("t2i", "val_gs2", "all_four"), 99.95, (99.82, 99.99)),
    ("text->image, all four (held-out)",
     ("t2i", "val_holdout_gs2", "all_four"), 99.93, (99.78, 99.98)),
    ("text->image, per quadrant (val)",
     ("t2i", "val_gs2", "per_quadrant"), 99.99, (99.96, 100.00)),
    ("joint, mutually consistent",
     ("joint", "uncond", "consistent"), 99.46, (99.19, 99.65)),
    ("joint, text well-formed",
     ("joint", "uncond", "text_wellformed"), 100.00, (99.91, 100.00)),
    ("joint, addends match image",
     ("joint", "uncond", "addends_match_image"), 99.71, (99.49, 99.83)),
    ("joint, arithmetic valid",
     ("joint", "uncond", "arithmetic_valid"), 99.76, (99.55, 99.87)),
]


def dig(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results.json")
    ap.add_argument("--strict-settings", action="store_true",
                    help="also fail if n, steps, EMA, compile or the entropy "
                         "schedule differ from the published protocol")
    args = ap.parse_args()

    with open(args.results) as fh:
        R = json.load(fh)

    print(f"checking {args.results}\n")

    # --- protocol first: a number is only comparable if it was produced the same way
    settings_bad = []
    if R.get("karras_fallback") is not False:
        settings_bad.append(
            f"karras_fallback={R.get('karras_fallback')} (must be false -- the "
            f"entropy tables were not found, so the sampler silently used a "
            f"Karras grid and NONE of the numbers below are comparable)")
    for key, want in (("n", 4096), ("steps_i2t", 256), ("steps_t2i", 256),
                      ("ema", True), ("compiled", True)):
        if R.get(key) != want:
            settings_bad.append(f"{key}={R.get(key)!r} (published: {want!r})")
    if R.get("global_step") != 500000:
        settings_bad.append(
            f"global_step={R.get('global_step')} (published: 500000 -- note that "
            f"best.pt is NOT the right checkpoint for this run; see the README)")

    if settings_bad:
        print("PROTOCOL DIFFERS FROM THE PUBLISHED RUN:")
        for s in settings_bad:
            print(f"  ! {s}")
        print()

    # --- the numbers
    width = max(len(n) for n, *_ in PAPER)
    fails, missing = [], []
    print(f"{'metric'.ljust(width)}  {'paper':>7}  {'yours':>7}  "
          f"{'published 95% CI':>18}   verdict")
    print("-" * (width + 52))
    for name, path, want, (lo, hi) in PAPER:
        cell = dig(R, path)
        if cell is None or "value" not in cell:
            missing.append(name)
            print(f"{name.ljust(width)}  {want:7.2f}  {'--':>7}  "
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18}   MISSING")
            continue
        got = 100.0 * cell["value"]
        ok = lo <= got <= hi
        if not ok:
            fails.append((name, want, got, lo, hi))
        print(f"{name.ljust(width)}  {want:7.2f}  {got:7.2f}  "
              f"{f'[{lo:.2f}, {hi:.2f}]':>18}   {'ok' if ok else 'OUT OF INTERVAL'}")

    print()
    if missing:
        print(f"{len(missing)} cell(s) missing from results.json -- was the run "
              f"invoked with --skip, or did it not finish?")
    if fails:
        print(f"{len(fails)} cell(s) outside the published interval:")
        for name, want, got, lo, hi in fails:
            print(f"  {name}: {got:.2f} vs paper {want:.2f}, interval [{lo:.2f}, {hi:.2f}]")
        print("\nIf the protocol block above is clean, this is a genuine "
              "discrepancy. Please open an issue and include results.json.")
    if not fails and not missing:
        print("All reported cells reproduce inside their published intervals.")

    bad = bool(fails or missing) or (args.strict_settings and bool(settings_bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
