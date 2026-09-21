#!/usr/bin/env python
"""
Offline, paper-grade evaluation of a MNIST-Sum checkpoint.

Produces the exact numbers for all three directions, at a sample size that
makes them quotable, with confidence intervals and -- for anything read through
the quadrant classifier -- the instrument's own ceiling beside it.

    python scripts/multimodal/eval_mnist_sum_offline.py \
        --config configs/multimodal/mnist_sum_p14_small.py \
        --ckpt   runs/multimodal/mnist_sum_p14_12x512/checkpoints/step=000500000.pt \
        --n 2048 --out results/mnist_sum_p14_500k

Deliberate choices, all of which change the numbers:

* EMA weights by default. The training callback evaluated under EMA, so raw
  weights would not be comparable with the curve we have been reading.
* torch.compile ON by default. Under eager + bf16 this long-sequence model
  loses image detail (eager+fp32 and eager+fp16 are fine, so it is bf16
  mantissa precision that compilation happens to avoid). A standalone script
  that skips compile silently evaluates a worse model -- use --no-compile only
  together with --fp32.
* Never select a checkpoint by validation loss. The entropy schedule's warmup
  ends at 40k and shifts the sigma distribution, so the loss is not comparable
  across that boundary and `best.pt` sits at its artifactual minimum, ~10
  accuracy points below the final model. Select on exact accuracy instead.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from data import mnist_sum_codec as C                                  # noqa: E402
from data.mnist_sum_bits import MNISTSumBitsDataset                    # noqa: E402
from evaluation import mnist_sum_classifier as K                       # noqa: E402
from evaluation.mnist_sum import (ms_sample, score_i2t, score_t2i,     # noqa: E402
                                  score_joint, wilson)


def load_cfg(path: str):
    s = importlib.util.spec_from_file_location("cfg_mod", path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.get_config()


def load_model(cfg, ckpt_path: str, device, *, use_ema: bool, compile_model: bool):
    from models import create_model
    model = create_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = None
    if use_ema and isinstance(ck.get("ema"), dict) and "shadow" in ck["ema"]:
        state = ck["ema"]["shadow"]
        print("[eval] loaded EMA shadow weights")
    if state is None:
        state = ck["model"]
        print("[eval] loaded raw model weights")
    cleaned = {}
    for k, v in state.items():
        for pfx in ("_orig_mod.", "module."):
            if k.startswith(pfx):
                k = k[len(pfx):]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:    print(f"[eval] WARNING missing keys: {len(missing)} e.g. {missing[:3]}")
    if unexpected: print(f"[eval] WARNING unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}")
    model.eval()
    if compile_model and hasattr(torch, "compile"):
        mode = getattr(cfg.train, "compile_mode", "default")
        print(f"[eval] torch.compile(mode={mode!r}) -- required for bf16 fidelity")
        model = torch.compile(model, mode=mode, fullgraph=False)
    return model, int(ck.get("global_step", -1))


def ci(p, n):
    lo, hi = wilson(p, n)
    return {"value": p, "ci_lo": lo, "ci_hi": hi, "n": float(n)}


def gen(cfg, model, proc, ds, device, *, task, n, micro, steps, gs, gamma,
        s_noise, win, sigma_decode, entropy_run_dir, seed=0):
    """Sample `n` rows for `task`, returning decoded text + quadrants + truth."""
    torch.manual_seed(seed)
    n = min(n, len(ds))
    idx = np.arange(n)
    true_d, true_s = ds.labels(idx)
    texts, quads = [], []
    t0 = time.time()
    for s0 in range(0, n, micro):
        chunk = torch.stack([ds[int(i)] for i in idx[s0:s0 + micro]]).float().to(device)
        bits = ms_sample(cfg, model, proc, chunk, layout=ds.layout, task=task,
                         sampler_name=str(getattr(cfg.train.mnist_sum_eval, "sampler", "ddim_entropic")),
                         num_steps=steps, guidance_scale=gs, device=device,
                         gamma=gamma, s_noise=s_noise, churn_window_mode=win,
                         sigma_decode=sigma_decode, entropy_run_dir=entropy_run_dir)
        out = ds.reconstruct_batch_from_bits(bits)
        texts.extend(out["text"]); quads.append(out["quadrants"].cpu())
        done = min(s0 + micro, n)
        print(f"    {task} gs={gs:g} {done}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return texts, torch.cat(quads, 0), true_d, true_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048, help="samples per direction per split")
    ap.add_argument("--n_joint", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--steps_i2t", type=int, default=256)
    ap.add_argument("--steps_t2i", type=int, default=256)
    ap.add_argument("--guidance", type=float, nargs="+", default=[2.0, 1.0])
    ap.add_argument("--gamma_i2t", type=float, default=0.175)
    ap.add_argument("--gamma_t2i", type=float, default=0.0)
    ap.add_argument("--clf_epochs", type=int, default=4)
    ap.add_argument("--clf_path", default="runs/mnist_sum_quadrant_cnn.pt")
    ap.add_argument("--mnist_raw", default=None,
                    help="Directory holding MNIST's raw IDX files. Resolved "
                         "automatically ($MNIST_RAW, then known locations) when "
                         "omitted; both arms must use the same one.")
    ap.add_argument("--ceiling_n", type=int, default=4096)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--allow-karras", action="store_true",
                    help="proceed even if the entropy tables are absent")
    ap.add_argument("--skip", nargs="*", default=[], choices=["i2t", "t2i", "joint"])
    args = ap.parse_args()

    if args.no_compile and not args.fp32:
        print("[eval] REFUSING: --no-compile without --fp32 evaluates a degraded model "
              "(eager+bf16 loses image detail on this sequence length). "
              "Pass --fp32 as well, or drop --no-compile.")
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    ec = cfg.train.mnist_sum_eval
    sigma_decode = float(getattr(ec, "sigma_decode", 0.1))
    s_noise = float(getattr(ec, "s_noise", 1.003))
    win = str(getattr(ec, "churn_window_mode", "full"))

    # Entropy tables decide the sampling schedule. Resolved from the CHECKPOINT,
    # not from cfg.evaluation.checkpoint_path, which is where the sampler looks by
    # default -- evaluating a checkpoint that lives anywhere else would silently
    # pick up another run's tables, or none, and fall back to a Karras grid behind
    # one WARN line. Two arms compared across two machines, one on entropic and one
    # on Karras, is a wrong result that looks like a real one, so this is fatal
    # rather than a warning.
    entropy_run_dir = Path(args.ckpt).expanduser().resolve().parent.parent
    need = ["entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt"]
    absent = [f for f in need if not (entropy_run_dir / f).exists()]
    if absent and not args.allow_karras:
        print(f"[eval] REFUSING: entropy tables {absent} missing in {entropy_run_dir}.\n"
              f"       The sampler would fall back to a Karras grid behind a single WARN "
              f"line, which is NOT comparable with the entropic numbers from training.\n"
              f"       Copy the run's entropy_*.pt next to the checkpoint dir, or pass "
              f"--allow-karras to accept a Karras schedule deliberately.")
        sys.exit(3)
    print(f"[eval] entropy tables: {entropy_run_dir} "
          f"({'OK' if not absent else 'MISSING -> Karras (explicitly allowed)'})")

    from diffusion.continuous.processes import ContinuousForwardProcess
    proc = ContinuousForwardProcess(cfg)
    model, step = load_model(cfg, args.ckpt, device,
                             use_ema=not args.no_ema, compile_model=not args.no_compile)

    splits = {s: MNISTSumBitsDataset(cfg, split=s) for s in ("val", "val_holdout")}

    # ---- the measuring instrument, and its ceiling ------------------------
    if Path(args.clf_path).exists():
        clf = K.load(args.clf_path, device); print(f"[eval] loaded classifier {args.clf_path}")
        clf_stats = {}
    else:
        print("[eval] training quadrant classifier ...")
        clf, clf_stats = K.train_classifier(device, raw_dir=args.mnist_raw,
                                            epochs=args.clf_epochs)
        clf_stats["mnist_raw"] = K.resolve_mnist_raw(args.mnist_raw)
        K.save(clf, args.clf_path)
    ceilings = {s: K.measure_ceiling(clf, ds, device, n=args.ceiling_n)
                for s, ds in splits.items()}
    for s, c in ceilings.items():
        print(f"[eval] CEILING {s}: all_four={100*c['ceiling_all_four']:.2f}% "
              f"per_quadrant={100*c['ceiling_per_quadrant']:.2f}% (n={int(c['n'])})")

    R = {"entropy_run_dir": str(entropy_run_dir), "karras_fallback": bool(absent),
     "checkpoint": args.ckpt, "global_step": step, "config": args.config,
         "ema": not args.no_ema, "compiled": not args.no_compile,
         "n": args.n, "steps_i2t": args.steps_i2t, "steps_t2i": args.steps_t2i,
         "classifier": {"path": args.clf_path, **clf_stats},
         "ceiling": ceilings, "i2t": {}, "t2i": {}, "joint": {}}

    for gs in args.guidance:
        for sname, ds in splits.items():
            base = ds.modal_sum_baseline
            # ---------- image -> text ----------
            if "i2t" not in args.skip:
                texts, _, td, ts = gen(cfg, model, proc, ds, device, task="i2t",
                                       n=args.n, micro=args.micro, steps=args.steps_i2t,
                                       gs=gs, gamma=args.gamma_i2t, s_noise=s_noise,
                                       win=win, sigma_decode=sigma_decode,
                                       entropy_run_dir=entropy_run_dir)
                m = score_i2t(texts, td[:len(texts)], ts[:len(texts)])
                nn = int(m.pop("n"))
                R["i2t"][f"{sname}_gs{gs:g}"] = {
                    **{k: ci(v, nn) for k, v in m.items()},
                    "modal_sum_baseline": base}
                print(f"  [i2t {sname} gs={gs:g}] e2e={100*m['end_to_end']:.2f}% "
                      f"percep={100*m['perception']:.2f}% arith={100*m['arithmetic']:.2f}% "
                      f"(baseline {100*base:.2f}%)", flush=True)
            # ---------- text -> image ----------
            if "t2i" not in args.skip:
                _, quads, td, _ = gen(cfg, model, proc, ds, device, task="t2i",
                                      n=args.n, micro=args.micro, steps=args.steps_t2i,
                                      gs=gs, gamma=args.gamma_t2i, s_noise=s_noise,
                                      win=win, sigma_decode=sigma_decode,
                                       entropy_run_dir=entropy_run_dir)
                pred = K.classify_quadrants(clf, quads, device)
                r = score_t2i(pred, td[:pred.shape[0]], ceilings[sname])
                R["t2i"][f"{sname}_gs{gs:g}"] = r
                print(f"  [t2i {sname} gs={gs:g}] all4={100*r['all_four']['value']:.2f}% "
                      f"(ceiling {100*r['ceiling_all_four']:.2f}%) "
                      f"per_quad={100*r['per_quadrant']['value']:.2f}%", flush=True)

    # ---------- joint : both halves from noise ----------
    if "joint" not in args.skip:
        ds = splits["val"]
        texts, quads, _, _ = gen(cfg, model, proc, ds, device, task="joint",
                                 n=args.n_joint, micro=args.micro, steps=args.steps_i2t,
                                 gs=1.0, gamma=args.gamma_i2t, s_noise=s_noise,
                                 win=win, sigma_decode=sigma_decode,
                                       entropy_run_dir=entropy_run_dir)
        pred = K.classify_quadrants(clf, quads, device)
        r = score_joint(texts, pred, ceilings["val"])
        R["joint"]["uncond"] = r
        print(f"  [joint] consistent={100*r['consistent']['value']:.2f}% "
              f"wellformed={100*r['text_wellformed']['value']:.2f}% "
              f"addends_match={100*r['addends_match_image']['value']:.2f}%", flush=True)

    json.dump(R, open(out_dir / "results.json", "w"), indent=1)
    (out_dir / "results.md").write_text(markdown(R))
    print(f"\n[eval] wrote {out_dir}/results.json and results.md")


def markdown(R) -> str:
    def pc(d): return f"{100*d['value']:.2f} <sub>[{100*d['ci_lo']:.2f}, {100*d['ci_hi']:.2f}]</sub>"
    L = [f"# MNIST-Sum offline evaluation", "",
         f"- checkpoint: `{R['checkpoint']}` (step {R['global_step']:,})",
         f"- weights: {'EMA' if R['ema'] else 'raw'}, compiled: {R['compiled']}",
         f"- n per direction per split: {R['n']:,}; steps i2t/t2i: {R['steps_i2t']}/{R['steps_t2i']}", ""]
    if R["classifier"].get("mnist_test_digit_accuracy"):
        L.append(f"Quadrant classifier: {100*R['classifier']['mnist_test_digit_accuracy']:.2f}% "
                 f"on binarised MNIST test digits.")
    L += ["", "## Classifier ceiling on REAL composites", "",
          "| split | all four | per quadrant |", "|---|---|---|"]
    for s, c in R["ceiling"].items():
        L.append(f"| {s} | {100*c['ceiling_all_four']:.2f} | {100*c['ceiling_per_quadrant']:.2f} |")
    if R["i2t"]:
        L += ["", "## image → text (exact, no instrument)", "",
              "| split · guidance | perception | arithmetic | end-to-end | e2e \\| percep | malformed | baseline |",
              "|---|---|---|---|---|---|---|"]
        for k, v in R["i2t"].items():
            L.append(f"| {k} | {pc(v['perception'])} | {pc(v['arithmetic'])} | **{pc(v['end_to_end'])}** | "
                     f"{pc(v['end_to_end_given_perception']) if 'end_to_end_given_perception' in v else '—'} | "
                     f"{pc(v['malformed'])} | {100*v['modal_sum_baseline']:.2f} |")
    if R["t2i"]:
        L += ["", "## text → image (via quadrant classifier)", "",
              "| split · guidance | all four | per quadrant | ceiling (all four) | % of ceiling |",
              "|---|---|---|---|---|"]
        for k, v in R["t2i"].items():
            L.append(f"| {k} | **{pc(v['all_four'])}** | {pc(v['per_quadrant'])} | "
                     f"{100*v.get('ceiling_all_four', 0):.2f} | {100*v.get('all_four_vs_ceiling', 0):.1f} |")
    if R["joint"]:
        v = R["joint"]["uncond"]
        L += ["", "## joint (both halves from noise) — mutual consistency", "",
              "| metric | value |", "|---|---|",
              f"| text well-formed | {pc(v['text_wellformed'])} |",
              f"| addends match the image | {pc(v['addends_match_image'])} |",
              f"| arithmetic valid | {pc(v['arithmetic_valid'])} |",
              f"| **consistent (equation correct for its image)** | **{pc(v['consistent'])}** |",
              f"| classifier ceiling | {100*v.get('ceiling_all_four', 0):.2f} |"]
    L += ["", "Intervals are 95% Wilson. Anything read through the classifier is bounded by "
          "its ceiling on real composites, which is reported beside it."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
