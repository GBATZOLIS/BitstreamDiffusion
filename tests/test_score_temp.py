"""Tests for Track A1 local score-level temperature (particle-free, 0 extra NFE).

kappa(sigma) = (v + sigma^2) / (tau*v + sigma^2) rescales the PF-ODE score/drift:
  * tau == 1.0 is a bit-identical no-op (both at the kappa-function level and end
    to end through DDIMSampler.sample);
  * kappa -> 1 at high sigma (mode allocation untouched);
  * kappa -> 1/tau as sigma -> 0 (late sharpening only);
  * kappa is monotone in sigma and bounded in [1, 1/tau] for tau < 1;
  * the knob is actually wired: tau < 1 changes the sampled trajectory.
"""
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler, _score_temp_kappa
from tests._sampler_harness import TinyBinaryDenoiser, make_cpu_cfg, make_conditioning

V = 0.25


def test_kappa_tau_one_is_identity():
    sig = torch.tensor([1e-3, 0.1, 1.0, 10.0, 80.0])
    k = _score_temp_kappa(sig, 1.0, V, ndim=2)
    assert torch.allclose(k.reshape(-1), torch.ones_like(sig), atol=1e-7)


def test_kappa_high_sigma_limit():
    # sigma^2 dominates v -> kappa ~ 1.
    k = _score_temp_kappa(torch.tensor(80.0), 0.3, V, ndim=2)
    assert abs(float(k) - 1.0) < 1e-3


def test_kappa_low_sigma_limit():
    # sigma -> 0 -> kappa -> 1/tau.
    tau = 0.3
    k = _score_temp_kappa(torch.tensor(1e-4), tau, V, ndim=2)
    assert abs(float(k) - 1.0 / tau) < 1e-2


def test_kappa_monotone_and_bounded():
    tau = 0.5
    sig = torch.logspace(-3, 2, 200)
    k = _score_temp_kappa(sig, tau, V, ndim=2).reshape(-1)
    # decreasing in sigma (sharpening concentrates at low sigma)
    assert torch.all(k[1:] <= k[:-1] + 1e-6)
    assert float(k.min()) >= 1.0 - 1e-6
    assert float(k.max()) <= 1.0 / tau + 1e-6


def test_kappa_broadcast_shapes():
    sig = torch.tensor([0.1, 0.5, 2.0])          # [B]
    assert _score_temp_kappa(sig, 0.5, V, ndim=2).shape == (3, 1)
    assert _score_temp_kappa(sig, 0.5, V, ndim=3).shape == (3, 1, 1)
    assert _score_temp_kappa(torch.tensor(0.5), 0.5, V, ndim=2).dim() == 0


def _run(tau, *, seed=0, steps=12):
    B, S, n_prompt = 2, 16, 4
    cfg = make_cpu_cfg(num_steps=steps)
    model = TinyBinaryDenoiser(S, seed=1)
    proc = ContinuousForwardProcess(cfg)
    sampler = DDIMSampler(model, proc, cfg)
    pf, pm = make_conditioning(B, S, n_prompt, seed=7)
    torch.manual_seed(seed)
    x, probs = sampler.sample(
        num_samples=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=steps, schedule="karras", sc_refresh_mode="carry", ati_eta=0.0,
        return_probs=True, progress=False, score_temp_tau=tau,
    )
    return x, probs, pf, pm


def test_end_to_end_tau_one_is_bit_identical():
    x0, p0, _, _ = _run(1.0, seed=0)
    x1, p1, _, _ = _run(1.0, seed=0)  # determinism sanity
    assert torch.equal(x0, x1) and torch.equal(p0, p1)
    # A default-arg run (no score_temp_tau passed) must match tau=1.0 exactly.
    B, S, n_prompt = 2, 16, 4
    cfg = make_cpu_cfg(num_steps=12)
    model = TinyBinaryDenoiser(S, seed=1)
    sampler = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    pf, pm = make_conditioning(B, S, n_prompt, seed=7)
    torch.manual_seed(0)
    xb, pb = sampler.sample(
        num_samples=B, seq_len=S, conditioning_prefix_full=pf, cond_prefix_mask=pm,
        num_steps=12, schedule="karras", sc_refresh_mode="carry", ati_eta=0.0,
        return_probs=True, progress=False,
    )
    assert torch.equal(xb, x0) and torch.equal(pb, p0)


def test_end_to_end_tau_changes_trajectory():
    x1, p1, pf, pm = _run(1.0, seed=0)
    xt, pt, _, _ = _run(0.3, seed=0)
    # The knob is active: sharpening changes the sampled state...
    assert not torch.equal(x1, xt)
    # ...but never on the clamped prompt coordinates.
    assert torch.equal(x1[pm], xt[pm])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all score-temp tests passed")
