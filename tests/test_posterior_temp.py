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


def _full_codebook(m):
    # all 2^m codes as rows, MSB-first to match token_ids_to_bits
    V = 2 ** m
    ids = torch.arange(V)
    bits = ((ids.unsqueeze(1) >> torch.arange(m - 1, -1, -1)) & 1).float()
    return bits  # [V, m]


def test_codeword_T1_full_codebook_recovers_sigmoid():
    # With the FULL 2^m codebook and T=1, the valid-codeword expectation E[c_b]
    # must equal the per-bit marginal sigmoid(ell_b) (independent Bernoullis).
    from diffusion.continuous.logit_postprocess import _codeword_sharpen
    m = 4
    C = _full_codebook(m)
    raw = torch.randn(2, 3 * m)  # B=2, P=3 tokens
    for target in ("learned", "full"):
        D = _codeword_sharpen(raw, None, temp=1.0, target=target, codebook=C)
        assert torch.allclose(D, torch.sigmoid(raw), atol=1e-5), target


def test_codeword_T1_with_mf_full_codebook():
    from diffusion.continuous.logit_postprocess import _codeword_sharpen
    m = 4
    C = _full_codebook(m)
    raw = torch.randn(2, 2 * m)
    mf = torch.randn(2, 2 * m)
    # At T=1 learned and full coincide and equal sigmoid(raw+mf).
    for target in ("learned", "full"):
        D = _codeword_sharpen(raw, mf, temp=1.0, target=target, codebook=C)
        assert torch.allclose(D, torch.sigmoid(raw + mf), atol=1e-5), target


def test_codeword_lowT_full_codebook_is_perbit_map():
    # Full codebook => joint MAP == per-bit MAP, so D -> 1[ell>0].
    from diffusion.continuous.logit_postprocess import _codeword_sharpen
    m = 4
    C = _full_codebook(m)
    raw = torch.randn(2, 2 * m)
    D = _codeword_sharpen(raw, None, temp=1e-3, target="learned", codebook=C)
    assert torch.allclose(D, (raw > 0).float(), atol=1e-3)


def test_codeword_restricted_avoids_invalid_corner():
    # m=2, valid codes {00,01,10} (11 dropped). raw favors both bits=1 (per-bit
    # MAP = invalid 11). Joint MAP over valid must NOT be [1,1]; the tie 01/10
    # gives D ~ [0.5, 0.5]. Proves invalid corner is unreachable.
    from diffusion.continuous.logit_postprocess import _codeword_sharpen
    C = torch.tensor([[0., 0.], [0., 1.], [1., 0.]])  # drop [1,1]
    raw = torch.tensor([[8.0, 8.0]])  # both bits strongly want 1
    D = _codeword_sharpen(raw, None, temp=1e-2, target="learned", codebook=C)
    assert (D < 0.9).all(), D  # never commits to the invalid 11 corner
    assert torch.allclose(D, torch.tensor([[0.5, 0.5]]), atol=1e-2)


def test_codeword_topk_in_hull():
    from diffusion.continuous.logit_postprocess import _codeword_sharpen
    m = 4
    C = _full_codebook(m)
    raw = torch.randn(2, 2 * m)
    D = _codeword_sharpen(raw, None, temp=0.5, target="learned", codebook=C, topk=3)
    assert bool((D >= -1e-5).all()) and bool((D <= 1 + 1e-5).all())


def test_matched_filter_binary_matches_postproc():
    import types
    from diffusion.continuous.logit_postprocess import _matched_filter_binary
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(continuous_logit_scaling="matched_filter_residual",
                                    matched_filter_center=0.5, matched_filter_scale=1.0,
                                    matched_filter_clip=30.0),
        diffusion=types.SimpleNamespace(continuous=types.SimpleNamespace(data_center=0.5)),
    )
    x_t = torch.rand(3, 8)
    sigma = torch.full((3,), 0.4)
    got = _matched_filter_binary(cfg, x_t, sigma)
    exp = ((x_t - 0.5) / 0.16).clamp(-30, 30)
    assert torch.allclose(got, exp, atol=1e-4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all posterior-temp tests passed")
