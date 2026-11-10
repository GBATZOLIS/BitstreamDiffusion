"""CPU tests: global score temperature in DDIMSampler and its protein wiring.

The protein generate() reads cfg.evaluation.score_temp (default 1.0) and passes
it to DDIMSampler for all directions (forward folding, inverse folding, joint).
tau = 1 must be bit-identical to the untouched sampler.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from test_protein_multimodal_e2e import M0FixtureMixin, _fixture_cfg

REPO = Path(__file__).resolve().parents[1]
L = 16


def _observed(task):
    rng = np.random.default_rng(0)
    if task == "inverse_folding":
        return {"struct_index": rng.integers(0, 8192, size=L)}
    if task == "forward_folding":
        return {"seq_ids": rng.integers(0, 20, size=L)}
    return None


class ScoreTemperatureTests(M0FixtureMixin, unittest.TestCase):
    def _run(self, task, **ev):
        from evaluation.evaluation_drivers.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        cfg.evaluation.solver = "ddim_entropic"
        cfg.evaluation.guidance_scale = 1.5
        cfg.evaluation.stochastic.enabled = True
        cfg.evaluation.stochastic.s_churn = 2.0
        cfg.evaluation.stochastic.window_mode = "fixed"
        cfg.evaluation.stochastic.s_tmin = 0.05
        cfg.evaluation.stochastic.s_tmax = 50.0
        for k, v in ev.items():
            setattr(cfg.evaluation, k, v)
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()
        calls = {"n": 0}
        real = model.forward

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        model.forward = counting  # type: ignore[assignment]
        out = generate(model, cfg, task, L, num_samples=2, device=torch.device("cpu"),
                       observed=_observed(task), seed=7)
        return out, calls["n"]

    def test_tau_one_is_bit_identical_for_every_direction(self):
        for task in ("forward_folding", "inverse_folding", "joint"):
            with self.subTest(task=task):
                base, n0 = self._run(task)
                one, n1 = self._run(task, score_temp=1.0, score_temp_final="sharpened")
                self.assertTrue(np.array_equal(base["bits"], one["bits"]))
                self.assertEqual(n0, n1)
                self.assertEqual(base["sampling"]["score_temp"], 1.0)

    def _final_state(self, task, **ev):
        # The tiny untrained fixture's readout barely depends on x, so compare the
        # sampler's final state (it moves under tempering) rather than rounded bits.
        from diffusion.continuous import samplers

        seen = {}
        real = samplers.DDIMSampler.sample

        def spy(self, *a, **k):
            x, probs = real(self, *a, **k)
            seen["x"] = x.detach().clone()
            return x, probs

        with mock.patch.object(samplers.DDIMSampler, "sample", spy):
            out, n = self._run(task, **ev)
        return out, n, seen["x"]

    def test_lower_tau_changes_trajectory_at_equal_cost_and_is_recorded(self):
        for task in ("forward_folding", "inverse_folding", "joint"):
            with self.subTest(task=task):
                _, n0, x0 = self._final_state(task)
                cold, n1, x1 = self._final_state(task, score_temp=0.3)
                self.assertEqual(n0, n1)
                self.assertGreater(float((x0 - x1).abs().max()), 1e-4)
                self.assertTrue(bool(torch.isfinite(x1).all()))
                self.assertEqual(cold["sampling"]["score_temp"], 0.3)
                self.assertEqual(cold["sampling"]["score_temp_final"], "unsharpened")

    def test_temperature_reaches_the_ddim_sampler(self):
        from diffusion.continuous import samplers

        seen = {}
        real = samplers.DDIMSampler.sample

        def spy(self, *a, **k):
            seen.update(score_temp=k.get("score_temp"), score_temp_final=k.get("score_temp_final"))
            return real(self, *a, **k)

        with mock.patch.object(samplers.DDIMSampler, "sample", spy):
            self._run("joint", score_temp=0.5, score_temp_final="sharpened")
        self.assertEqual(seen, {"score_temp": 0.5, "score_temp_final": "sharpened"})

    def test_heun_refuses_temperature(self):
        with self.assertRaisesRegex(ValueError, "DDIM"):
            self._run("joint", solver="heun", score_temp=0.5)

    def test_invalid_temperature_rejected(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._run("joint", score_temp=bad)


class _XSensitiveModel(torch.nn.Module):
    """Deterministic toy denoiser whose logits depend on x, sigma and the SC input."""

    def forward(self, x, sigma, cond):
        s = sigma.reshape(-1, 1).to(x.dtype)
        return 3.0 * torch.roll(x, 1, dims=1) / (1 + s) + 1.5 * (cond - 0.5)


class SamplerTemperatureTests(unittest.TestCase):
    def _sample(self, **kw):
        from configs.proteins.proteins_400m import get_config
        from diffusion.continuous.samplers import DDIMSampler

        cfg = get_config()
        cfg.device = "cpu"
        cfg.data.sequence_len = 64
        cfg.evaluation.stochastic.enabled = False
        cfg.model.continuous_logit_scaling = "none"
        sampler = DDIMSampler(_XSensitiveModel(), None, cfg)
        torch.manual_seed(0)
        return sampler.sample(4, 64, guidance_scale=0.0, schedule="karras", num_steps=12,
                              sc_refresh_mode="carry", return_probs=True, progress=False, **kw)

    def test_tau_one_identical_and_lower_tau_changes_readout(self):
        x0, p0 = self._sample()
        x1, p1 = self._sample(score_temp=1.0, score_temp_final="sharpened")
        self.assertTrue(torch.equal(x0, x1) and torch.equal(p0, p1))
        _, pc = self._sample(score_temp=0.3)
        self.assertFalse(torch.equal(p0, pc))
        self.assertTrue(bool(torch.isfinite(pc).all()))


class SweepConfigTests(unittest.TestCase):
    def _load(self, env):
        spec = importlib.util.spec_from_file_location(
            "st_cfg", REPO / "configs/proteins/proteins_400m_score_temp.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("SCORE_TEMP", "SCORE_TEMP_FINAL"):
                if k not in env:
                    os.environ.pop(k, None)
            return mod.get_config()

    def test_sweep_config_changes_only_temperature_and_output_dir(self):
        from configs.proteins.proteins_400m import get_config

        base = get_config()
        cfg = self._load({"SCORE_TEMP": "0.5"})
        self.assertEqual(cfg.evaluation.score_temp, 0.5)
        self.assertEqual(cfg.evaluation.score_temp_final, "unsharpened")
        self.assertTrue(cfg.evaluation.out_dir.endswith("evaluation/score_temp_0.5"))
        for k in ("solver", "schedule", "num_sampling_steps", "guidance_scale", "checkpoint_path"):
            self.assertEqual(cfg.evaluation[k], base.evaluation[k])
        self.assertEqual(cfg.evaluation.stochastic.to_dict(), base.evaluation.stochastic.to_dict())
        self.assertEqual(base.evaluation.score_temp, 1.0)

    def test_sweep_config_requires_explicit_valid_temperature(self):
        for env in ({}, {"SCORE_TEMP": "0"}, {"SCORE_TEMP": "1.5"},
                    {"SCORE_TEMP": "0.5", "SCORE_TEMP_FINAL": "last"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self._load(env)


if __name__ == "__main__":
    unittest.main()
