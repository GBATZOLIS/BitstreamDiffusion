"""Tests for posterior-temperature decoding (continuous analogue of MDLM/Duo low-T).

Covers:
  * T == 1.0 is a bit-identical no-op vs the untempered postprocessing (the
    backward-compat guarantee that the headline 28.96% config is unchanged).
  * "learned" target sharpens only the network logit, leaving the matched-filter
    data-consistency term intact.
  * "full" target sharpens the whole postprocessed logit.
  * The sigma_ramp schedule is T=1 above sigma_hi, T below sigma_lo, monotone in
    between, and a global no-op when T==1.
"""
import math

import torch

from diffusion.continuous.logit_postprocess import apply_continuous_logit_postprocessing
from diffusion.continuous.samplers import _posterior_temp_at


def _mf(x_t, sigma, *, center=0.5, scale=1.0, clip=30.0):
    sigma2 = float(sigma) ** 2
    mf = scale * (x_t.float() - center) / sigma2
    return mf.clamp(-clip, clip)


def test_temp_one_is_noop_matched_filter():
    torch.manual_seed(0)
    B, S = 4, 32
    logits = torch.randn(B, S)
    x_t = torch.rand(B, S)
    sigma = torch.full((B,), 0.7)
    kw = dict(
        mode="matched_filter_residual", x_t=x_t, data_center=0.5,
        matched_filter_center=0.5, matched_filter_scale=1.0, matched_filter_clip=30.0,
    )
    base = apply_continuous_logit_postprocessing(logits.clone(), sigma, **kw)
    for target in ("learned", "full"):
        out = apply_continuous_logit_postprocessing(
            logits.clone(), sigma, posterior_temp=1.0, posterior_temp_target=target, **kw)
        assert torch.equal(base, out), f"T=1 changed output for target={target}"


def test_learned_target_leaves_matched_filter_untouched():
    B, S = 2, 8
    logits = torch.randn(B, S)
    x_t = torch.rand(B, S)
    sigma = torch.full((B,), 0.5)
    mf = _mf(x_t, 0.5)
    T = 0.4
    kw = dict(mode="matched_filter_residual", x_t=x_t, matched_filter_center=0.5,
              matched_filter_scale=1.0, matched_filter_clip=30.0)
    out = apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=T, posterior_temp_target="learned", **kw)
    expected = logits / T + mf
    assert torch.allclose(out, expected, atol=1e-5)


def test_full_target_sharpens_everything():
    B, S = 2, 8
    logits = torch.randn(B, S)
    x_t = torch.rand(B, S)
    sigma = torch.full((B,), 0.5)
    mf = _mf(x_t, 0.5)
    T = 0.4
    kw = dict(mode="matched_filter_residual", x_t=x_t, matched_filter_center=0.5,
              matched_filter_scale=1.0, matched_filter_clip=30.0)
    out = apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=T, posterior_temp_target="full", **kw)
    expected = (logits + mf) / T
    assert torch.allclose(out, expected, atol=1e-5)


def test_learned_vs_full_differ_when_mf_present():
    B, S = 2, 8
    logits = torch.randn(B, S)
    x_t = torch.rand(B, S)
    sigma = torch.full((B,), 0.5)
    kw = dict(mode="matched_filter_residual", x_t=x_t, matched_filter_center=0.5,
              matched_filter_scale=1.0, matched_filter_clip=30.0)
    learned = apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=0.5, posterior_temp_target="learned", **kw)
    full = apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=0.5, posterior_temp_target="full", **kw)
    assert not torch.allclose(learned, full)


def test_sharpening_pushes_probs_to_rails():
    # sigmoid(logit/T) is more extreme than sigmoid(logit) for T<1.
    logits = torch.tensor([[2.0, -2.0, 0.5, -0.5]])
    x_t = torch.full_like(logits, 0.5)  # mf == 0 at center, isolate the learned term
    sigma = torch.tensor([1.0])
    kw = dict(mode="matched_filter_residual", x_t=x_t, matched_filter_center=0.5,
              matched_filter_scale=1.0, matched_filter_clip=30.0)
    p1 = torch.sigmoid(apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=1.0, posterior_temp_target="learned", **kw))
    pT = torch.sigmoid(apply_continuous_logit_postprocessing(
        logits.clone(), sigma, posterior_temp=0.3, posterior_temp_target="learned", **kw))
    # Bits with logit>0 get pushed up; logit<0 pushed down.
    assert (pT[logits > 0] > p1[logits > 0]).all()
    assert (pT[logits < 0] < p1[logits < 0]).all()


def test_sigma_ramp_schedule():
    T = 0.3
    lo, hi = 0.1, 4.0
    # T==1 -> global no-op regardless of schedule/sigma
    assert _posterior_temp_at(1.0, temp=1.0, schedule="sigma_ramp", sigma_lo=lo, sigma_hi=hi) == 1.0
    # above hi -> untempered
    assert _posterior_temp_at(10.0, temp=T, schedule="sigma_ramp", sigma_lo=lo, sigma_hi=hi) == 1.0
    # at/below lo -> full temperature
    assert math.isclose(_posterior_temp_at(0.05, temp=T, schedule="sigma_ramp", sigma_lo=lo, sigma_hi=hi), T)
    # const applies everywhere
    assert math.isclose(_posterior_temp_at(50.0, temp=T, schedule="const", sigma_lo=lo, sigma_hi=hi), T)
    # monotone increase in T as sigma grows across the band (1.0 at hi, T at lo)
    mids = [_posterior_temp_at(s, temp=T, schedule="sigma_ramp", sigma_lo=lo, sigma_hi=hi)
            for s in (0.2, 0.5, 1.0, 2.0)]
    assert all(T <= m <= 1.0 for m in mids)
    assert mids == sorted(mids)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all posterior-temp tests passed")
