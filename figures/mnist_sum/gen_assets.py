"""Generate figure assets for the MNIST-Sum paper section (P=28 arm)."""
import pathlib
import importlib.util as ilu, json, sys, os
import numpy as np, torch
REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
os.chdir(REPO)

OUT = sys.argv[1]
spec = ilu.spec_from_file_location("c", "configs/multimodal/mnist_sum_p28_small.py")
m = ilu.module_from_spec(spec); spec.loader.exec_module(m); cfg = m.get_config()

from data.mnist_sum_bits import MNISTSumBitsDataset
from data import mnist_sum_codec as C
from evaluation.mnist_sum import ms_sample, parse_equation
from models import create_model
from pathlib import Path

dev = torch.device("cuda")
# step=000500000.pt, never best.pt: best.pt is selected on validation loss, which
# the entropy-schedule warmup at 40k makes incomparable across that boundary, so
# its minimum is an artefact near step 39k -- about 10 accuracy points below the
# final model. Overridable so an earlier checkpoint can be inspected.
CKPT = os.environ.get(
    "MS_CKPT", "runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = create_model(cfg)
state = ck.get("ema_shadow") or ck.get("ema") or ck["model"]
if isinstance(state, dict) and "shadow" in state: state = state["shadow"]
cleaned = {}
for k, v in state.items():
    for pfx in ("_orig_mod.", "module."):
        if k.startswith(pfx): k = k[len(pfx):]
    cleaned[k] = v
# strict=True on purpose. This used to be strict=False, which would silently
# accept a checkpoint whose parameter set does not match the model the config
# builds -- exactly the failure mode that let a config ask for a component the
# code did not have. A mismatch here means the config and the checkpoint
# disagree and the figures would be generated from a half-loaded model.
# Exactly one key may be absent: time_fn._freq is a deterministic sinusoidal
# table rebuilt identically at construction, and the EMA shadow tracks
# parameters rather than buffers, so it is never in the shadow. The superseded
# run's own evaluation logged the same missing key. Everything else is strict:
# a real parameter mismatch means the config and the checkpoint disagree and the
# figure would be drawn from a half-loaded model.
BENIGN_MISSING = {"time_fn._freq"}
_missing, _unexpected = model.load_state_dict(cleaned, strict=False)
_missing = set(_missing) - BENIGN_MISSING
if _missing or _unexpected:
    raise RuntimeError(
        f"checkpoint does not match the model this config builds.\n"
        f"  missing from checkpoint : {sorted(_missing)}\n"
        f"  unexpected in checkpoint: {sorted(_unexpected)}\n"
        f"checkpoint={CKPT}")
model = model.to(dev).eval()
model = torch.compile(model, fullgraph=False)
proc = None
ERD = Path(CKPT).parent.parent
ds = MNISTSumBitsDataset(cfg, split="val")
E = cfg.train.mnist_sum_eval

def run(task, n, gs, gamma, seed):
    torch.manual_seed(seed)
    idx = np.arange(n)
    refs = torch.stack([ds[int(i)] for i in idx]).float().to(dev)
    d, s = ds.labels(idx)
    bits = ms_sample(cfg, model, proc, refs, layout=ds.layout, task=task,
                     sampler_name="ddim_entropic", num_steps=256, guidance_scale=gs,
                     device=dev, gamma=gamma, s_noise=1.003, churn_window_mode="full",
                     sigma_decode=0.1, entropy_run_dir=ERD)
    out = ds.reconstruct_batch_from_bits(bits)
    return (out["image"][:, 0].cpu().numpy(), out["text"], d, s,
            out["quadrants"].cpu())

N = 32
res = {}
print("[assets] joint co-generation ...", flush=True)
img_j, txt_j, _, _, quad_j = run("joint", N, 1.0, 0.0, seed=11)
print("[assets] text -> image ...", flush=True)
img_t, txt_t, d_t, s_t, quad_t = run("t2i", N, 2.0, 0.0, seed=7)
print("[assets] image -> text ...", flush=True)
_, txt_i, d_i, s_i, _ = run("i2t", N, 2.0, 0.175, seed=7)
real = np.stack([ds.reconstruct_from_bits(ds[int(i)].float())["image"][0].numpy() for i in range(N)])

def eq_words(dig): return " ".join(C.equation_words(list(map(int, dig))))

# ---- per-sample correctness flags -------------------------------------------
# fig_paper_compact.py asserts that the displayed examples are scored correct, so
# the assets must carry the same verdicts the reported metrics are built from.
# Definitions copied from evaluation/mnist_sum.py:
#   i2t   ok         -> end_to_end: parses AND stated sum == TRUE sum
#   t2i   ok         -> all_four:   classifier's read of the generated quadrants
#                       equals the digits the prompt named
#   joint consistent -> addends match the classifier's read AND the arithmetic
#                       is valid (score_joint's `consistent`)
from evaluation import mnist_sum_classifier as K
clf = K.load("runs/mnist_sum_quadrant_cnn.pt", dev)
print("[assets] loaded quadrant classifier (the cached one, not retrained)", flush=True)

pred_t = K.classify_quadrants(clf, quad_t, dev).cpu()
pred_j = K.classify_quadrants(clf, quad_j, dev).cpu()

i2t_scored, t2i_scored, joint_scored = [], [], []
for k in range(N):
    p = parse_equation(txt_i[k])
    i2t_scored.append({"ok": bool(p is not None and p[1] == int(s_i[k])),
                       "wellformed": p is not None})
    t2i_scored.append({"ok": bool((pred_t[k] == d_t[k].cpu()).all())})
    pj = parse_equation(txt_j[k])
    if pj is None:
        joint_scored.append({"consistent": False, "wellformed": False,
                             "addends_match": False, "arithmetic_valid": False})
    else:
        add, stated = pj
        a_ok = (add == pred_j[k].tolist())
        s_ok = (stated == sum(add))
        joint_scored.append({"consistent": bool(a_ok and s_ok), "wellformed": True,
                             "addends_match": bool(a_ok), "arithmetic_valid": bool(s_ok)})
print(f"[assets] scored: i2t ok={sum(r['ok'] for r in i2t_scored)}/{N}  "
      f"t2i ok={sum(r['ok'] for r in t2i_scored)}/{N}  "
      f"joint consistent={sum(r['consistent'] for r in joint_scored)}/{N}", flush=True)
# joint: is the co-generated equation exactly right for the co-generated image?
joint_rows = []
for k in range(N):
    p = parse_equation(txt_j[k])
    joint_rows.append({"text": txt_j[k], "parsed": None if p is None else [p[0], p[1]],
                       "wellformed": p is not None})
np.savez_compressed(OUT + "/assets.npz",
                    joint_img=img_j, t2i_img=img_t, real_img=real)
json.dump({"joint": joint_rows,
           "i2t_scored": i2t_scored,
           "t2i_scored": t2i_scored,
           "joint_scored": joint_scored,
           "t2i_pred_digits": [list(map(int, pred_t[k])) for k in range(N)],
           "joint_pred_digits": [list(map(int, pred_j[k])) for k in range(N)],
           "t2i_prompt": [eq_words(d_t[k]) for k in range(N)],
           "t2i_true_digits": [list(map(int, d_t[k])) for k in range(N)],
           "i2t_pred": txt_i, "i2t_true": [eq_words(d_i[k]) for k in range(N)],
           "real_true": [eq_words(d_i[k]) for k in range(N)],
           "layout": {"P": 28, "text_pos": 3, "image_pos": 112, "markers": 6,
                      "positions": 121, "bits": 3388}},
          open(OUT + "/assets.json", "w"), indent=1)
print("[assets] done", flush=True)
