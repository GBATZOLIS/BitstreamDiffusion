"""
evaluation/mnist_sum.py

Sampling + EXACT scoring for the MNIST-Sum corpus.

Deliberately parallel to `evaluation/multimodal.py` rather than a generalisation
of it: that module resolves its masks through `data.mm_codec`, whose 19-bit
geometry backs published CC3M/CC12M checkpoints. Nothing here touches it.

The point of this corpus is that scoring needs no proxy metric and no learned
judge. `score_i2t` decomposes the image->text error three ways, because a single
accuracy number cannot distinguish "misread a digit" from "cannot add":

    perception  -- are the four NAMED addends the true digits? (also per slot)
    arithmetic  -- does the STATED sum equal the sum of the STATED addends?
                   scored even when perception is wrong, which isolates
                   arithmetic from reading.
    end_to_end  -- does the stated sum equal the TRUE sum?

plus `malformed` (output does not parse as the template) and the
`modal_sum_baseline` an image-blind guesser would score.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from contextlib import contextmanager

from data import mnist_sum_codec as C


# --- masks -----------------------------------------------------------------
def task_masks(layout: C.Layout, device, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (cond_mask, protect_mask) as BIT masks over the flat sequence.
    cond  = held at their clean values (the conditioning segments + markers)
    protect = never denoised (markers), per the method's marker rule.
    """
    text = layout.text_bit_mask(device)
    image = layout.image_bit_mask(device)
    marker = layout.marker_bit_mask(device)
    if task == "t2i":
        cond = text | marker
    elif task in ("i2t", "caption"):
        cond = image | marker
    elif task == "joint":
        cond = marker.clone()
    else:
        raise ValueError(f"unknown task {task!r}")
    return cond, marker


@torch.no_grad()
@contextmanager
def ms_churn(cfg, *, gamma: float, num_steps: int, s_noise: float = 1.003,
             window_mode: str = "full"):
    """Turn on EDM churn at a per-step rate `gamma` for the duration of the block.

    Churn is parameterised by gamma, not by s_churn, because s_churn is a
    per-*schedule* quantity: s_churn = gamma * (num_steps - 1). CC3M's config
    warns about exactly this -- its static s_churn=22.0 is the 128-step viz
    constant and reusing it at 256 steps would silently halve the rate to
    gamma = 22/255 ~= 0.086. Deriving s_churn from gamma at the actual step
    count keeps the rate fixed when the step count is retuned.

    gamma <= 0 leaves cfg untouched, so the sampler stays a deterministic ODE.
    """
    st = getattr(getattr(cfg, "evaluation", object()), "stochastic", None)
    if st is None or float(gamma) <= 0.0:
        yield
        return
    keys = ("enabled", "s_churn", "s_noise", "window_mode")
    prev = {k: getattr(st, k, None) for k in keys}
    st.enabled = True
    st.s_churn = float(gamma) * float(max(1, int(num_steps) - 1))
    st.s_noise = float(s_noise)
    st.window_mode = str(window_mode)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is not None:
                setattr(st, k, v)


def ms_sample(cfg, model, proc, ref_bits: torch.Tensor, *, layout: C.Layout,
              task: str, sampler_name: str = "ddim_entropic", num_steps: int = 64,
              guidance_scale: float = 1.0, device, sigma_decode: Optional[float] = None,
              sc_refresh_mode: str = "carry", entropy_run_dir=None,
              gamma: float = 0.0, s_noise: float = 1.003,
              churn_window_mode: str = "full",
              amp: bool = True, amp_dtype: torch.dtype = torch.bfloat16,
              progress: bool = False) -> torch.Tensor:
    """Generate the target modality; the conditioned modality is re-imposed exactly."""
    from evaluation.generation_driver import create_sampler

    ref_bits = ref_bits.to(device=device, dtype=torch.float32)
    B, S = ref_bits.shape
    cond_mask, protect = task_masks(layout, device, task)
    cond_b = cond_mask.view(1, S).expand(B, S).contiguous()

    bundle = create_sampler(cfg, model, proc, sampler_name, device=device,
                            num_steps=num_steps)
    sigma_min = float(getattr(cfg.diffusion.continuous, "sigma_min", 0.002))
    sd = float(sigma_decode if sigma_decode is not None else sigma_min)

    with ms_churn(cfg, gamma=gamma, num_steps=int(num_steps), s_noise=s_noise,
                  window_mode=churn_window_mode), \
            torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
        _, probs = bundle.sampler.sample(
            B, S, schedule=bundle.schedule, num_steps=int(num_steps),
            sigma_min_override=sd, entropy_run_dir=entropy_run_dir,
            conditioning_prefix_full=ref_bits, cond_prefix_mask=cond_b,
            protect_mask=protect.view(1, S).expand(B, S).contiguous(),
            guidance_scale=float(guidance_scale), sc_refresh_mode=sc_refresh_mode,
            return_probs=True, progress=progress)

    bits = (probs > 0.5).to(torch.uint8)
    ref_u8 = (ref_bits > 0.5).to(torch.uint8)
    bits[cond_b] = ref_u8[cond_b]
    return bits


# --- parsing + exact scoring ----------------------------------------------
def parse_equation(text: str) -> Optional[Tuple[List[int], int]]:
    """
    "seven + six + three + two = eighteen" -> ([7,6,3,2], 18)
    Returns None if the string does not match the template exactly.
    """
    toks = text.split()
    if len(toks) != 2 * C.N_DIGITS + 1:          # d + d + d + d = s  -> 9
        return None
    word_to_num = {w: i for i, w in enumerate(C.NUMBER_WORDS)}
    addends: List[int] = []
    for i in range(C.N_DIGITS):
        w = toks[2 * i]
        if w not in word_to_num:
            return None
        v = word_to_num[w]
        if v > 9:                                 # an addend must be a digit
            return None
        addends.append(v)
        if i < C.N_DIGITS - 1 and toks[2 * i + 1] != "+":
            return None
    if toks[2 * C.N_DIGITS - 1] != "=":
        return None
    tail = toks[-1]
    if tail not in word_to_num:
        return None
    return addends, word_to_num[tail]


def score_i2t(texts: Sequence[str], true_digits: torch.Tensor,
              true_sums: torch.Tensor) -> Dict[str, float]:
    """Exact three-way decomposition. true_digits [N,4], true_sums [N]."""
    n = len(texts)
    if n == 0:
        return {}
    td = true_digits.cpu().tolist()
    ts = true_sums.cpu().tolist()

    malformed = 0
    slot_ok = [0] * C.N_DIGITS
    perception_ok = 0       # all four addends correct
    arithmetic_ok = 0       # stated sum == sum(stated addends)
    end_to_end_ok = 0       # stated sum == true sum
    for i, t in enumerate(texts):
        p = parse_equation(t)
        if p is None:
            malformed += 1
            continue
        add, stated = p
        for q in range(C.N_DIGITS):
            if add[q] == td[i][q]:
                slot_ok[q] += 1
        if add == list(td[i]):
            perception_ok += 1
        if stated == sum(add):
            arithmetic_ok += 1
        if stated == ts[i]:
            end_to_end_ok += 1

    out = {
        "n": float(n),
        "malformed": malformed / n,
        "perception": perception_ok / n,
        "arithmetic": arithmetic_ok / n,
        "end_to_end": end_to_end_ok / n,
    }
    for q, name in enumerate(C.DIGIT_SLOT_NAMES):
        out[f"perception_{name}"] = slot_ok[q] / n
    # Arithmetic conditioned on having read the image correctly -- the cleanest
    # read of "can it add", free of perception errors.
    if perception_ok:
        both = sum(1 for i, t in enumerate(texts)
                   if (p := parse_equation(t)) is not None
                   and p[0] == list(td[i]) and p[1] == ts[i])
        out["end_to_end_given_perception"] = both / perception_ok
    return out


# --- Wilson interval -------------------------------------------------------
def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% CI for a proportion. Wilson rather than normal: at p near 1 and the
    n we can afford, the normal interval runs past 1.0 and understates the
    uncertainty that matters for a 98%-accuracy claim."""
    if n <= 0:
        return (0.0, 0.0)
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _ci(p: float, n: int) -> Dict[str, float]:
    lo, hi = wilson(p, n)
    return {"value": p, "ci_lo": lo, "ci_hi": hi, "n": float(n)}


# --- text -> image : exact, against the classifier ceiling -----------------
def score_t2i(pred_digits: torch.Tensor, true_digits: torch.Tensor,
              ceiling: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    `pred_digits` [N,4] are the classifier's read of the GENERATED quadrants;
    `true_digits` [N,4] are the digits the prompt named.

    The task is one-to-many -- any rendering of those digits is correct -- so
    this scores semantic correctness, not pixel fidelity. Every number is
    reported beside the classifier's own ceiling on real composites, because
    the instrument, not the model, sets the maximum.
    """
    n = int(pred_digits.shape[0])
    if n == 0:
        return {}
    hit = (pred_digits == true_digits)
    all4 = float(hit.all(1).float().mean())
    per = hit.float().mean(0)
    out: Dict[str, Any] = {
        "all_four": _ci(all4, n),
        "per_quadrant": _ci(float(per.mean()), 4 * n),
        **{k: _ci(float(per[i]), n) for i, k in enumerate(C.DIGIT_SLOT_NAMES)},
    }
    if ceiling:
        c4 = ceiling.get("ceiling_all_four")
        cq = ceiling.get("ceiling_per_quadrant")
        out["ceiling_all_four"] = c4
        out["ceiling_per_quadrant"] = cq
        # Fraction of the achievable score actually reached: 1.0 means the
        # generated images are as legible to the instrument as real ones.
        if c4: out["all_four_vs_ceiling"] = all4 / c4
        if cq: out["per_quadrant_vs_ceiling"] = float(per.mean()) / cq
    return out


# --- joint : mutual consistency -------------------------------------------
def score_joint(texts: Sequence[str], pred_digits: torch.Tensor,
                ceiling: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Both halves are generated together from noise; nothing is conditioned.

    The question is not whether either half is good in isolation but whether
    they AGREE: does the equation describe the image it was generated with?
    The paper's joint direction is currently defended with pair-CLIP coherence,
    a soft similarity with no notion of correct. This is a rate.

    Reported separately so a low consistency score can be attributed:
      text_wellformed  -- the text parses as the template at all
      addends_match    -- the named addends equal the classifier's read of the image
      arithmetic_valid -- the stated sum equals the sum of the stated addends
      consistent       -- addends_match AND arithmetic_valid: the equation is
                          exactly correct FOR THE IMAGE GENERATED WITH IT
    """
    n = len(texts)
    if n == 0:
        return {}
    wf = am = av = cons = 0
    for i, t in enumerate(texts):
        p = parse_equation(t)
        if p is None:
            continue
        wf += 1
        add, stated = p
        a_ok = (add == pred_digits[i].tolist())
        s_ok = (stated == sum(add))
        am += int(a_ok); av += int(s_ok); cons += int(a_ok and s_ok)
    out: Dict[str, Any] = {
        "text_wellformed": _ci(wf / n, n),
        "addends_match_image": _ci(am / n, n),
        "arithmetic_valid": _ci(av / n, n),
        "consistent": _ci(cons / n, n),
    }
    if ceiling and ceiling.get("ceiling_all_four"):
        # Agreement is read through the classifier, so it inherits its ceiling.
        out["ceiling_all_four"] = ceiling["ceiling_all_four"]
        out["consistent_vs_ceiling"] = (cons / n) / ceiling["ceiling_all_four"]
    return out
