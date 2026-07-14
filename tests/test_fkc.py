"""Correctness gates for the Feynman-Kac Euler-Maruyama sampler (FKC).

These MUST pass before any Sudoku/GSM8K FKC sweep (they guard the tempering
math and the particle bookkeeping):

  Gate 1  beta=1, K=1 is bit-identical to EulerMaruyamaSampler (lambda0=0 and >0)
  Gate 1c beta>1 edm_churn: FKC weight uses the sigma_cur->sigma_next interval
          (consecutive targets), NOT the churned sigma_hat  [regression]
  Gate 2  beta=1: every weight increment is 0, ESS==K, no in-loop resampling
  Gate 3  beta>1, K=1: weights/resampling cannot change the trajectory (control)
  Gate 4  prompt invariance: clamped coords never move; drift/noise zero there
  Gate 5  ancestry: x / sc / score gather with identical indices
  Gate 6  duplicated ancestors branch: same state post-resample, diverge via noise
  Gate 7  systematic resampling: fixed weights + fixed u0 -> exact indices
  Gate 8  analytic 1-D Gaussian: weighted/resampled Var -> (v+sigma^2)/beta
          for lambda in {0, 1, 1/beta, lambda(sigma)}  [the key math test]
"""
import math

import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import (
    DDIMSampler, EulerMaruyamaSampler, FeynmanKacEulerMaruyamaSampler,
)
from diffusion.continuous.smc import (
    effective_sample_size, systematic_resample_indices, gather_particles,
)
from tests._sampler_harness import TinyBinaryDenoiser, make_cpu_cfg, make_conditioning

B, S, NP, STEPS = 2, 16, 4, 10


def _em(cfg, model, **kw):
    return EulerMaruyamaSampler(
        model, ContinuousForwardProcess(cfg), cfg,
        lambda_profile_name="flat", lambda_profile_normalize="as_saved",
        em_step_gamma_cap=1.0, **kw,
    )


def _fkc(cfg, model, **kw):
    return FeynmanKacEulerMaruyamaSampler(
        model, ContinuousForwardProcess(cfg), cfg,
        lambda_profile_name="flat", lambda_profile_normalize="as_saved",
        em_step_gamma_cap=1.0, **kw,
    )


def _run_em(sampler, pf, pm, seed):
    torch.manual_seed(seed)
    x, probs = sampler.sample(
        num_samples=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", sc_refresh_mode="carry", ati_eta=0.0,
        return_probs=True, progress=False,
    )
    return x, (probs >= 0.5).long()


def _run_fkc(sampler, pf, pm, seed, K=1):
    torch.manual_seed(seed)          # global RNG -> prior + Langevin noise (matches EM)
    out = sampler.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=seed, progress=False,
    )
    return out


def _gate1(lambda_zero):
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    em = _em(cfg, model, lambda_zero=lambda_zero)
    fkc = _fkc(cfg, model, beta=1.0, num_particles=1, lambda_zero=lambda_zero)
    x_em, bits_em = _run_em(em, pf, pm, seed=123)
    out = _run_fkc(fkc, pf, pm, seed=123, K=1)
    x_fkc = out.x.squeeze(1)
    bits_fkc = out.bits.squeeze(1)
    assert torch.equal(x_em, x_fkc), (
        f"lambda0={lambda_zero}: x max|diff|={float((x_em-x_fkc).abs().max()):.3e}"
    )
    assert torch.equal(bits_em, bits_fkc)


def test_gate1_deterministic_bit_identical_to_em():
    _gate1(0.0)


def test_gate1_stochastic_bit_identical_to_em():
    _gate1(0.5)


def _ddim_with_churn(cfg, model, gamma, pf, pm, seed, steps=STEPS):
    from ml_collections import config_dict
    st = config_dict.ConfigDict()
    st.enabled = True
    st.s_churn = gamma * (steps - 1)      # -> per-step gamma_i = gamma (full band)
    st.s_noise = 1.0
    st.window_mode = "full"
    st.entropy_quantile_lo = 0.0
    st.entropy_quantile_hi = 1.0
    st.entropy_fallback = "deterministic"
    st.s_tmin = None
    st.s_tmax = None
    cfg.evaluation.stochastic = st
    s = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    torch.manual_seed(seed)
    x, probs = s.sample(
        num_samples=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=steps, schedule="karras", sc_refresh_mode="carry", ati_eta=0.0,
        return_probs=True, progress=False,
    )
    return x, (probs >= 0.5).long()


def test_gate1b_churn_proposal_bit_identical_to_ddim_churn():
    # FKC-churn(beta=1, K=1) must reproduce the production DDIM+EDM-churn sampler
    # bit-for-bit at a matched constant gamma (churn noise uses the global RNG in
    # the same order; resampling uses an isolated generator).
    g = 0.3
    cfg_d = make_cpu_cfg(num_steps=STEPS)     # DDIM: churn enabled below
    cfg_f = make_cpu_cfg(num_steps=STEPS)     # FKC: cfg churn stays disabled
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    x_d, bits_d = _ddim_with_churn(cfg_d, model, g, pf, pm, seed=123)
    fkc = FeynmanKacEulerMaruyamaSampler(
        model, ContinuousForwardProcess(cfg_f), cfg_f,
        beta=1.0, num_particles=1, proposal="edm_churn", churn_gamma=g,
    )
    torch.manual_seed(123)
    out = fkc.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=123, progress=False,
    )
    assert torch.equal(x_d, out.x.squeeze(1)), \
        f"churn x max|diff|={float((x_d-out.x.squeeze(1)).abs().max()):.3e}"
    assert torch.equal(bits_d, out.bits.squeeze(1))


def test_gate1c_churn_fkc_weight_uses_consecutive_target_interval():
    # REGRESSION for the load-bearing FKC bug. For proposal='edm_churn' at beta>1 the
    # FKC log-weight increment must be integrated over the CONSECUTIVE-TARGET interval
    # sigma_cur^2 - sigma_next^2, NOT the churned-up sigma_hat^2 - sigma_next^2. Using
    # sigma_hat double-counts the up-churn excursion [(1+gamma)^2-1] sigma_cur^2 (~0.96
    # sigma_cur^2 at gamma=0.4) and is what collapsed ESS. Unlike Gate 8 (a hand loop),
    # this drives the REAL sample_particles edm_churn path: we spy on s_weight at every
    # step and check log_weights_final matches the sigma_cur formula to fp precision --
    # and differs materially from the sigma_hat formula. Fails on the pre-fix code.
    beta, gamma, K = 1.5, 0.4, 3
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = FeynmanKacEulerMaruyamaSampler(
        model, ContinuousForwardProcess(cfg), cfg,
        beta=beta, num_particles=K, proposal="edm_churn", churn_gamma=gamma,
        resampling_policy="never", final_resample=False,   # keep logw un-reset
    )

    # Spy on every denoiser call: capture (sigma_state, s_weight). The in-loop calls use
    # the churned sigma_hat; the trailing final-decode calls use the un-churned final
    # sigma. diagnostics.sigmas has exactly one float(sigma_cur) per real step, so it
    # gives us both the step count N and the exact sigma_cur grid (no un-churn needed).
    captured = []
    orig = fkc._guided_posterior_and_score

    def _spy(x, sigma_state, sc, sc_u, **kw):
        out = orig(x, sigma_state, sc, sc_u, **kw)      # (..., s_weight, beta_w)
        captured.append((float(sigma_state), out[4].detach().clone()))
        return out

    fkc._guided_posterior_and_score = _spy

    torch.manual_seed(123)
    out = fkc.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=123, progress=False,
    )
    logw = out.log_weights_final                        # [B, K], accumulated, never reset

    g_eff = min(gamma, math.sqrt(2.0) - 1.0)
    sig_cur = [float(s) for s in out.diagnostics.sigmas]   # exact sigma_cur per step
    N = len(sig_cur)
    sigma_final = captured[N][0]                         # first final-decode call, un-churned
    sig_next = sig_cur[1:] + [sigma_final]
    sig_hat = [c * (1.0 + g_eff) for c in sig_cur]       # what the buggy code used
    s_weights = [w for _, w in captured[:N]]

    def _accumulate(intervals):
        acc = torch.zeros_like(logw)
        for dsig2, s in zip(intervals, s_weights):
            snorm2 = s.to(torch.float64).square().sum(dim=-1)      # [B, K]
            acc = acc + 0.5 * beta * (beta - 1.0) * dsig2 * snorm2
        return acc

    want_cur = _accumulate([c * c - n * n for c, n in zip(sig_cur, sig_next)])
    want_hat = _accumulate([h * h - n * n for h, n in zip(sig_hat, sig_next)])

    assert torch.allclose(logw, want_cur, atol=1e-6, rtol=1e-6), (
        "edm_churn FKC weight must use the sigma_cur->sigma_next interval; "
        f"max|logw - want_cur|={float((logw - want_cur).abs().max()):.3e}"
    )
    # Guard that the test actually discriminates: the buggy sigma_hat interval is far off.
    denom = want_cur.abs().max().clamp_min(1e-12)
    rel = float((want_hat - want_cur).abs().max() / denom)
    assert rel > 0.5, f"non-discriminating: sigma_hat vs sigma_cur differ by only {rel:.2%}"


def test_gate2_beta_one_weights_are_noop():
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.0, num_particles=8, lambda_zero=0.5)
    out = _run_fkc(fkc, pf, pm, seed=7, K=8)
    assert torch.allclose(out.log_weights_final, torch.zeros_like(out.log_weights_final))
    ess = torch.stack(out.diagnostics.ess)                 # [T, B]
    assert torch.allclose(ess, torch.full_like(ess, 8.0), atol=1e-6)
    assert not any(out.diagnostics.resampled)              # no in-loop resampling


def test_gate3_beta_gt1_K1_trajectory_independent_of_resampling():
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    # K=1: resampling of one particle is identity, so final_resample on/off must match.
    a = _fkc(cfg, model, beta=1.6, num_particles=1, lambda_zero=0.5, final_resample=True)
    b = _fkc(cfg, model, beta=1.6, num_particles=1, lambda_zero=0.5, final_resample=False)
    xa = _run_fkc(a, pf, pm, seed=9).x
    xb = _run_fkc(b, pf, pm, seed=9).x
    assert torch.equal(xa, xb) and torch.isfinite(xa).all()


def test_gate4_prompt_invariance():
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.5, num_particles=6, lambda_zero=0.6)
    out = _run_fkc(fkc, pf, pm, seed=11, K=6)
    pm_bk = pm.unsqueeze(1).expand(B, 6, S)
    pf_bk = pf.unsqueeze(1).expand(B, 6, S)
    # clamped coordinates never moved from the clean prefix
    assert torch.equal(out.x[pm_bk], pf_bk[pm_bk])
    # decoded prompt bits equal the prompt
    assert torch.equal(out.bits[pm_bk], pf_bk[pm_bk].long())


def test_gate5_ancestry_gather_consistency():
    # x, sc, score tagged by particle id must gather with identical ancestor idx.
    Bk, K = 3, 5
    idx = torch.tensor([[4, 4, 0, 1, 2], [0, 1, 2, 3, 4], [2, 2, 2, 2, 2]])
    tag = torch.arange(K).view(1, K, 1).expand(Bk, K, 7).float()
    x = tag.clone(); sc = tag.clone() + 100.0; score = tag.clone() + 1000.0
    gx, gsc, gsco = (gather_particles(t, idx) for t in (x, sc, score))
    for b in range(Bk):
        for k in range(K):
            a = int(idx[b, k])
            assert int(gx[b, k, 0]) == a
            assert int(gsc[b, k, 0]) == a + 100
            assert int(gsco[b, k, 0]) == a + 1000


def test_gate6_duplicated_ancestors_branch_via_noise():
    # After a forced resample onto few ancestors, particles sharing an ancestor
    # start identical then diverge on FREE coords through independent EM noise,
    # while prompt coords stay clamped. Drive it with a large beta so ESS
    # collapses and resampling fires, then check post-run diversity.
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=2.0, num_particles=8, lambda_zero=0.7,
               ess_threshold_fraction=0.9)
    out = _run_fkc(fkc, pf, pm, seed=13, K=8)
    assert any(out.diagnostics.resampled), "expected at least one resample event"
    # free coords differ across particles (branching), prompt coords identical.
    free = ~pm[0].bool()
    xf = out.x[0][:, free]                     # [K, n_free]
    assert xf.unique(dim=0).shape[0] > 1, "particles did not branch on free coords"
    pm_bk = pm.unsqueeze(1).expand(B, 8, S)
    pf_bk = pf.unsqueeze(1).expand(B, 8, S)
    assert torch.equal(out.x[pm_bk], pf_bk[pm_bk])


def _run_fkc_cfg(sampler, pf, pm, seed, w):
    torch.manual_seed(seed)
    return sampler.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=seed, guidance_scale=w, progress=False,
    )


def test_gate9_cfg_w1_reduces_to_plain_conditional():
    # CFG+FKC at guidance w=1 targets q_u^0 q_c^1 = q_c with zero FKC weight, so it
    # must be bit-identical to the plain conditional FKC run (guidance off).
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.0, num_particles=4, lambda_zero=0.5)
    out_plain = _run_fkc(fkc, pf, pm, seed=21, K=4)
    out_w1 = _run_fkc_cfg(fkc, pf, pm, seed=21, w=1.0)
    assert torch.equal(out_plain.x, out_w1.x), \
        f"w=1 x max|diff|={float((out_plain.x-out_w1.x).abs().max()):.3e}"
    assert torch.equal(out_plain.bits, out_w1.bits)
    # w=1 leaves FKC weights at zero (beta_w(beta_w-1)=0)
    assert torch.allclose(out_w1.log_weights_final, torch.zeros_like(out_w1.log_weights_final))


def test_gate9b_cfg_w_gt1_runs_and_reweights():
    # CFG+FKC at w>1 must run, stay finite, respect prompt invariance, and produce
    # non-uniform FKC weights (||s_c - s_u||^2 potential is active).
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.0, num_particles=6, lambda_zero=0.6)
    out = _run_fkc_cfg(fkc, pf, pm, seed=23, w=2.0)
    assert torch.isfinite(out.x).all()
    pm_bk = pm.unsqueeze(1).expand(B, 6, S)
    pf_bk = pf.unsqueeze(1).expand(B, 6, S)
    assert torch.equal(out.x[pm_bk], pf_bk[pm_bk])                 # prompt invariance
    lw = torch.stack([w for w in [out.log_weights_final]])         # [1,B,K]
    assert float(lw.abs().max()) > 0.0, "w>1 produced all-zero FKC weights"


def test_gate9c_cfg_beta_gt1_rejected():
    # Annealing beta>1 combined with guidance is out of scope in v1 -> must raise.
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.5, num_particles=2, lambda_zero=0.5)
    try:
        _run_fkc_cfg(fkc, pf, pm, seed=1, w=2.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for beta>1 with guidance_scale>0")


def test_gate7_systematic_resample_fixed_u0():
    # w = [0.7, 0.1, 0.1, 0.1], cdf = [0.7, 0.8, 0.9, 1.0], u0 = 0.1
    # positions = [0.10, 0.35, 0.60, 0.85] (all off the CDF knots) -> [0, 0, 0, 2]
    logw = torch.log(torch.tensor([[0.7, 0.1, 0.1, 0.1]], dtype=torch.float64))
    idx = systematic_resample_indices(logw, u0=torch.tensor([[0.1]]))
    assert idx.reshape(-1).tolist() == [0, 0, 0, 2]


# ------------------------------- Gate 8 -----------------------------------
# Analytic 1-D Gaussian FKC, validating the tempering math via the shared smc.py
# weight formula (the sigmoid-bound sampler cannot represent a real-valued
# Gaussian). q_sigma = N(0, v + sigma^2), s(x,sigma) = -x/(v+sigma^2); target
# q^beta has Var = (v + sigma^2)/beta. Proposal (codebase convention,
# h = s_next - s_cur, d = -sigma*s):
#   x    <- x + h*beta*(1+lam)*d + sqrt(2*lam*sigma*Delta)*z
#   dlogw += 0.5*beta*(beta-1)*(sigma_cur^2 - sigma_next^2)*s^2
#
# The FK log-weight is lambda-INDEPENDENT, so the *importance-weighted* terminal
# variance must equal (v+sigma^2)/beta for every lambda in {0, 1, 1/beta,
# lambda(sigma)} -- this is the theorem under test. We use the weighted
# estimator (unbiased for all lambda) rather than a resampled one: lambda=0 has
# no mixing, so its particle diversity can only decay and a resampled variance
# would be meaningless. A modest beta and a short sigma-range keep the
# finite-step discretization bias and the weight spread small.

def _weighted_var(x, logw):
    w = torch.softmax(logw, dim=1)
    mean = (w * x).sum(dim=1, keepdim=True)
    return float((w * (x - mean) ** 2).sum())


def _analytic_fkc_weighted_var(beta, lam_mode, *, v=1.0, K=16384, N=200, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    sig_max, sig_min = 3.0, 0.5
    i = torch.linspace(0, 1, N + 1, dtype=torch.float64)
    sigmas = sig_max + i * (sig_min - sig_max)          # linear grid, fine steps

    def lam_of(sig):
        if lam_mode == "zero":
            return 0.0
        if lam_mode == "one":
            return 1.0
        if lam_mode == "inv_beta":
            return 1.0 / beta
        return float(0.5 + 0.6 * math.exp(-((math.log(float(sig))) ** 2) / 2.0))

    # Prior = tempered high-sigma marginal q_{sigma_max}^beta.
    x = torch.randn(1, K, dtype=torch.float64) * math.sqrt((v + sig_max ** 2) / beta)
    logw = torch.zeros(1, K, dtype=torch.float64)
    for k in range(N):
        sc, sn = sigmas[k], sigmas[k + 1]
        s = -x / (v + sc ** 2)
        d = -sc * s
        logw = logw + 0.5 * beta * (beta - 1.0) * (sc ** 2 - sn ** 2) * (s ** 2)
        lam = lam_of(sc)
        h = sn - sc
        x = x + h * beta * (1.0 + lam) * d
        if lam > 0.0:
            delta = (sc - sn).clamp_min(0.0)
            z = torch.randn(1, K, dtype=torch.float64, generator=gen)
            x = x + (2.0 * lam * sc * delta).clamp_min(0.0).sqrt() * z
    target_var = (v + float(sigmas[-1]) ** 2) / beta
    return _weighted_var(x, logw), target_var


def _check_gate8(lam_mode):
    beta = 1.5
    got, want = _analytic_fkc_weighted_var(beta, lam_mode)
    rel = abs(got - want) / want
    assert rel < 0.12, f"lam={lam_mode}: weighted Var={got:.4f} want {want:.4f} (rel {rel:.2%})"


def test_gate8_gaussian_lambda_zero():
    _check_gate8("zero")


def test_gate8_gaussian_lambda_one():
    _check_gate8("one")


def test_gate8_gaussian_lambda_inv_beta():
    _check_gate8("inv_beta")


def test_gate8_gaussian_lambda_sigma():
    _check_gate8("sigma")


# ------------------------------- Gate 10 ----------------------------------
# Analytic 1-D Gaussian CFG-FKC (Prop 3.1), the two-model analogue of Gate 8.
# It validates the three CFG-specific pieces Gate 8 does NOT cover: the guided
# DRIFT s_u + w(s_c - s_u), the score-DIFFERENCE weight ||s_c - s_u||^2, and the
# beta_w = w mapping.
#
# Unconditional q_u = N(0, v_u + sigma^2), conditional q_c = N(0, v_c + sigma^2),
#   s_u = -x/(v_u+sigma^2),  s_c = -x/(v_c+sigma^2).
# Geometric average q_u^{1-w} q_c^{w} is Gaussian with
#   1/Var* = (1-w)/(v_u+sigma^2) + w/(v_c+sigma^2).
# Proposal (codebase convention, annealing beta=1 for pure CFG; h=s_next-s_cur):
#   s_geo = (1-w) s_u + w s_c = s_u + w (s_c - s_u);  d = -sigma * s_geo
#   x     <- x + h*(1+lam)*d + sqrt(2*lam*sigma*Delta)*z
#   dlogw += 0.5*w*(w-1)*(sigma_cur^2 - sigma_next^2)*||s_c - s_u||^2
# The FK weight is lambda-independent, so the weighted terminal variance must hit
# Var*(sigma_min) for every lambda -- the theorem under test. A weight using
# ||s_c||^2 or ||s_u||^2 (instead of the difference), or the wrong coefficient,
# misses the target; lambda-independence would also break.

def _analytic_cfg_weighted_var(w, lam_mode, *, v_u=1.0, v_c=0.4, K=16384, N=200, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    sig_max, sig_min = 3.0, 0.5
    i = torch.linspace(0, 1, N + 1, dtype=torch.float64)
    sigmas = sig_max + i * (sig_min - sig_max)

    def target_var(sig2):
        inv = (1.0 - w) / (v_u + sig2) + w / (v_c + sig2)
        return 1.0 / inv

    def lam_of(sig):
        if lam_mode == "zero":
            return 0.0
        if lam_mode == "one":
            return 1.0
        if lam_mode == "inv_w":
            return 1.0 / w
        return float(0.5 + 0.6 * math.exp(-((math.log(float(sig))) ** 2) / 2.0))

    # Prior = geometric-average high-sigma marginal.
    x = torch.randn(1, K, dtype=torch.float64) * math.sqrt(target_var(sig_max ** 2))
    logw = torch.zeros(1, K, dtype=torch.float64)
    for k in range(N):
        sc, sn = sigmas[k], sigmas[k + 1]
        s_u = -x / (v_u + sc ** 2)
        s_c = -x / (v_c + sc ** 2)
        s_geo = (1.0 - w) * s_u + w * s_c        # guided drift score = s_u + w(s_c - s_u)
        s_wt = s_c - s_u                          # weight uses the score DIFFERENCE
        logw = logw + 0.5 * w * (w - 1.0) * (sc ** 2 - sn ** 2) * (s_wt ** 2)
        lam = lam_of(sc)
        h = sn - sc
        d = -sc * s_geo
        x = x + h * (1.0 + lam) * d               # annealing beta = 1 for pure CFG
        if lam > 0.0:
            delta = (sc - sn).clamp_min(0.0)
            z = torch.randn(1, K, dtype=torch.float64, generator=gen)
            x = x + (2.0 * lam * sc * delta).clamp_min(0.0).sqrt() * z
    return _weighted_var(x, logw), target_var(float(sigmas[-1]) ** 2)


def _check_gate10(lam_mode):
    w = 1.5
    got, want = _analytic_cfg_weighted_var(w, lam_mode)
    rel = abs(got - want) / want
    assert rel < 0.12, f"lam={lam_mode}: weighted Var={got:.4f} want {want:.4f} (rel {rel:.2%})"


def test_gate10_cfg_gaussian_lambda_zero():
    _check_gate10("zero")


def test_gate10_cfg_gaussian_lambda_one():
    _check_gate10("one")


def test_gate10_cfg_gaussian_lambda_inv_w():
    _check_gate10("inv_w")


def test_gate10_cfg_gaussian_lambda_sigma():
    _check_gate10("sigma")


# ------------------------------- Gate 11 ----------------------------------
# CODE-PATH validation of CFG-FKC: Gate 10 checks the weight MATH; Gate 11 drives
# the REAL sample_particles CFG path (annealing beta=1, guidance w>1, em proposal
# so sigma_state == sigma_cur) and confirms the sampler wires the right quantities
# into the shared weight kernel:
#   (1) s_weight IS the score difference s_c - s_u  (independently == (Dc-Du)/sigma^2
#       in the free/non-prompt region, from the two posteriors the method returns),
#   (2) beta_w == w (the guidance weight, NOT the annealing beta=1),
#   (3) log_weights_final == 1/2 * w(w-1) * sum (sigma_cur^2 - sigma_next^2) * ||s_weight||^2.

def test_gate11_cfg_sampler_wires_score_diff_and_betaw():
    w, K = 2.0, 3
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = _fkc(cfg, model, beta=1.0, num_particles=K, lambda_zero=0.5)  # em proposal

    captured = []
    orig = fkc._guided_posterior_and_score

    def _spy(x, sigma_state, sc, sc_u, **kw):
        out = orig(x, sigma_state, sc, sc_u, **kw)   # (D_geo, s_drift, Dc, Du, s_weight, beta_w)
        Dc, Du, s_weight, beta_w = out[2], out[3], out[4], float(out[5])
        sc2 = float(sigma_state) ** 2
        pmb = kw["pm"].bool()
        free = (~(pmb if pmb.dim() == x.dim() else pmb.unsqueeze(1)))  # [B,1,S] broadcast over K
        s_diff = (Dc - Du) / sc2                                      # == s_c - s_u in free coords
        # atol/rtol absorb the float-cancellation gap between the code's
        # (Dc-x)/s^2 - (Du-x)/s^2 and this recomputed (Dc-Du)/s^2 (~2e-6 at low
        # sigma); a WRONG s_weight (e.g. s_c, or the negated diff) would be off
        # by O(0.1-1), so this still discriminates strongly.
        diff_ok = torch.allclose((s_weight * free), (s_diff * free), atol=1e-4, rtol=1e-2)
        captured.append((float(sigma_state), s_weight.detach().clone(), beta_w, bool(diff_ok)))
        return out

    fkc._guided_posterior_and_score = _spy
    torch.manual_seed(7)
    out = fkc.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=7, guidance_scale=w, progress=False,
    )
    logw = out.log_weights_final

    sig_cur = [float(s) for s in out.diagnostics.sigmas]
    N = len(sig_cur)
    sigma_final = captured[N][0]
    sig_next = sig_cur[1:] + [sigma_final]
    steps = captured[:N]

    # (1) s_weight is the score DIFFERENCE (Dc-Du)/sigma^2 at every in-loop step
    assert all(ok for _, _, _, ok in steps), "s_weight is not s_c - s_u in the free region"
    # em proposal => the spied sigma_state is exactly sigma_cur
    assert all(abs(s0 - c) < 1e-9 for (s0, _, _, _), c in zip(steps, sig_cur))
    # (2) the weight coefficient is the GUIDANCE weight w, not the annealing beta (=1)
    assert all(abs(b - w) < 1e-9 for _, _, b, _ in steps), "beta_w != guidance weight w"

    # (3) accumulated log-weights match the CFG formula with the sampler's own s_weight
    acc = torch.zeros_like(logw)
    for (c, n, (_, s, _, _)) in zip(sig_cur, sig_next, steps):
        snorm2 = s.to(torch.float64).square().sum(dim=-1)
        acc = acc + 0.5 * w * (w - 1.0) * (c * c - n * n) * snorm2
    assert torch.allclose(logw, acc, atol=1e-6, rtol=1e-6), \
        f"sampler CFG weight != formula; max|logw-acc|={float((logw - acc).abs().max()):.3e}"
    assert float(logw.abs().max()) > 0.0, "w>1 produced all-zero CFG weights"


# ------------------------------- Gate 12 ----------------------------------
# CFG analogue of Gate 1c: the load-bearing interval fix must ALSO hold for the
# CFG-FKC edm_churn path (annealing beta=1, guidance w>1). Drives the real
# sample_particles CFG+churn path, spies on s_weight (evaluated at sigma_hat --
# the documented O(gamma) score-location approximation) and beta_w, and asserts
# log_weights_final uses the CONSECUTIVE-TARGET interval sigma_cur^2 - sigma_next^2
# (NOT the churned sigma_hat interval), with beta_w == w and s_weight == s_c - s_u.

def test_gate12_cfg_churn_weight_uses_consecutive_target_interval():
    w, gamma, K = 2.0, 0.4, 3
    cfg = make_cpu_cfg(num_steps=STEPS)
    model = TinyBinaryDenoiser(S, seed=3)
    pf, pm = make_conditioning(B, S, NP, seed=5)
    fkc = FeynmanKacEulerMaruyamaSampler(
        model, ContinuousForwardProcess(cfg), cfg,
        beta=1.0, num_particles=K, proposal="edm_churn", churn_gamma=gamma,
        resampling_policy="never", final_resample=False,   # keep logw un-reset
    )

    captured = []
    orig = fkc._guided_posterior_and_score

    def _spy(x, sigma_state, sc, sc_u, **kw):
        out = orig(x, sigma_state, sc, sc_u, **kw)
        Dc, Du, s_weight, beta_w = out[2], out[3], out[4], float(out[5])
        sc2 = float(sigma_state) ** 2                       # sigma_hat^2 under churn
        pmb = kw["pm"].bool()
        free = ~(pmb if pmb.dim() == x.dim() else pmb.unsqueeze(1))
        diff_ok = torch.allclose((s_weight * free), (((Dc - Du) / sc2) * free), atol=1e-4, rtol=1e-2)
        captured.append((float(sigma_state), s_weight.detach().clone(), beta_w, bool(diff_ok)))
        return out

    fkc._guided_posterior_and_score = _spy
    torch.manual_seed(123)
    out = fkc.sample_particles(
        num_prompts=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=STEPS, schedule="karras", seed=123, guidance_scale=w, progress=False,
    )
    logw = out.log_weights_final

    g_eff = min(gamma, math.sqrt(2.0) - 1.0)
    sig_cur = [float(s) for s in out.diagnostics.sigmas]
    N = len(sig_cur)
    sigma_final = captured[N][0]                            # first final-decode call, un-churned
    sig_next = sig_cur[1:] + [sigma_final]
    sig_hat = [c * (1.0 + g_eff) for c in sig_cur]          # what the buggy interval would use
    steps = captured[:N]
    s_weights = [s for _, s, _, _ in steps]

    assert all(ok for _, _, _, ok in steps), "s_weight is not s_c - s_u (free) under churn"
    assert all(abs(b - w) < 1e-9 for _, _, b, _ in steps), "beta_w != guidance weight w"

    def _accumulate(intervals):
        acc = torch.zeros_like(logw)
        for dsig2, s in zip(intervals, s_weights):
            snorm2 = s.to(torch.float64).square().sum(dim=-1)
            acc = acc + 0.5 * w * (w - 1.0) * dsig2 * snorm2
        return acc

    want_cur = _accumulate([c * c - n * n for c, n in zip(sig_cur, sig_next)])
    want_hat = _accumulate([h * h - n * n for h, n in zip(sig_hat, sig_next)])
    assert torch.allclose(logw, want_cur, atol=1e-6, rtol=1e-6), (
        "CFG+churn FKC weight must use the sigma_cur->sigma_next interval; "
        f"max|logw - want_cur|={float((logw - want_cur).abs().max()):.3e}"
    )
    denom = want_cur.abs().max().clamp_min(1e-12)
    rel = float((want_hat - want_cur).abs().max() / denom)
    assert rel > 0.5, f"non-discriminating: sigma_hat vs sigma_cur differ by only {rel:.2%}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all FKC gates passed")
