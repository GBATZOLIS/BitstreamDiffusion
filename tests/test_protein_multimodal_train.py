"""CPU tests for the multimodal training step and warm-start column surgery.

Run with:
    python -m unittest tests.test_protein_multimodal_train
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from ml_collections import config_dict

from data import protein_tasks as T
from data.protein_structure_codec import PATCH_BITS_PER_RESIDUE
from diffusion.continuous.processes import ContinuousForwardProcess
from models.sdt import SequenceVDTContinuousModel
from trainers.multimodal_step import multimodal_training_step
from utils.protein_warmstart import warm_start_multimodal_from_sequence


def _base_cfg(multimodal, patch_size, content_dim=8, self_condition=False):
    cfg = config_dict.ConfigDict()
    cfg.framework = "continuous_score"
    cfg.device = "cpu"
    cfg.model = config_dict.ConfigDict()
    cfg.model.patch_size = patch_size
    cfg.model.embed_dim = 32
    cfg.model.n_blocks = 2
    cfg.model.n_heads = 2
    cfg.model.out_dim = 1
    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.head_hidden = 24
    cfg.model.content_dim_continuous = content_dim
    cfg.model.self_condition = self_condition
    cfg.model.multimodal = multimodal
    cfg.model.independent_modality_noise = True
    cfg.model.lambda_seq = 1.0
    cfg.model.lambda_struct = 1.0
    cfg.data = config_dict.ConfigDict()
    cfg.data.representation = "binary"
    cfg.data.vocab_size = 2
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.train = config_dict.ConfigDict()
    cfg.train.loss_type = "binary_ce"
    # Unit weighting keeps the tiny-sigma EDM weight from dominating a short
    # overfit; the EDM weighting itself is exercised in the loss unit tests.
    cfg.train.loss_weighting = "none"
    cfg.train.sigma_sampling_strategy = "log-normal"
    return cfg


def _batch(B, L, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=(B, L * PATCH_BITS_PER_RESIDUE)).astype(
        np.float32
    )
    states = np.zeros((B, L, 2), dtype=np.int64)
    for b in range(B):
        states[b] = T.build_example_states(
            "joint" if b % 2 else "forward_folding", L
        )
    residue = np.ones((B, L), dtype=bool)
    seq_full, struct_full = T.full_modality_target_masks_np(states, residue)
    return {
        "x0": torch.from_numpy(x0),
        "states": torch.from_numpy(states),
        "seq_target_full": torch.from_numpy(seq_full),
        "struct_target_full": torch.from_numpy(struct_full),
    }


class MultimodalStepTests(unittest.TestCase):
    def test_step_runs_and_overfits(self):
        torch.manual_seed(0)
        cfg = _base_cfg(multimodal=True, patch_size=18)
        model = SequenceVDTContinuousModel(cfg)
        proc = ContinuousForwardProcess(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        batch = _batch(4, 5, seed=1)
        torch.manual_seed(0)  # fix the per-step noise draw sequence
        losses = []
        for _ in range(80):
            opt.zero_grad()
            loss, comps = multimodal_training_step(
                model, batch, proc, cfg, device=torch.device("cpu")
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertTrue(np.isfinite(losses).all())
        self.assertIn("seq_bit_acc", comps)
        self.assertIn("struct_bit_acc", comps)
        # trend down over a short window (noise makes it non-monotone)
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]))


class WarmStartTests(unittest.TestCase):
    def test_column_surgery_copies_sequence_slots(self):
        torch.manual_seed(0)
        src = SequenceVDTContinuousModel(
            _base_cfg(multimodal=False, patch_size=5)
        )
        tgt = SequenceVDTContinuousModel(
            _base_cfg(multimodal=True, patch_size=18)
        )
        # perturb source so copies are detectable
        for p in src.parameters():
            p.data.add_(0.1)
        report = warm_start_multimodal_from_sequence(
            tgt, src.state_dict(), verbose=False
        )
        self.assertGreater(report["exact"], 0)
        self.assertGreaterEqual(report["slot_copied"], 1)

        sd_src = dict(src.named_parameters())
        sd_tgt = dict(tgt.named_parameters())
        # trunk block copied exactly
        self.assertTrue(
            torch.allclose(
                sd_tgt["blocks.0.attn.qkv_proj.weight"].data,
                sd_src["blocks.0.attn.qkv_proj.weight"].data,
            )
        )
        # patch_proj: first 5 sequence slot-blocks copied
        d = src.D_trunk_in
        cols = 5 * d
        self.assertTrue(
            torch.allclose(
                sd_tgt["patch_proj.weight"].data[:, :cols],
                sd_src["patch_proj.weight"].data[:, :cols],
            )
        )
        # target has 18 slots -> struct columns exist beyond the copied region
        self.assertEqual(sd_tgt["patch_proj.weight"].shape[1], 18 * d)
        self.assertEqual(sd_src["patch_proj.weight"].shape[1], 5 * d)


class SamplerClampTests(unittest.TestCase):
    def test_observed_bits_held_through_reverse_diffusion(self):
        # Inverse folding: clamp the 13 structure bits and denoise sequence bits.
        # The observed bits must be held at their clean values every solver step.
        from configs.proteins.multimodal_lfq18_smoke import get_config
        from data.protein_structure_codec import SEQ_BITS_PER_RESIDUE
        from diffusion.continuous.processes import ContinuousForwardProcess
        from diffusion.continuous.samplers import HeunSampler

        cfg = get_config()
        cfg.device = "cpu"
        cfg.model.use_flash_attn = False
        cfg.model.embed_dim = 32
        cfg.model.n_blocks = 2
        cfg.model.n_heads = 2
        cfg.model.content_dim_continuous = 8
        cfg.model.head_hidden = 16
        cfg.model.self_condition = False
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()
        sampler = HeunSampler(model, ContinuousForwardProcess(cfg), cfg)

        B, L = 2, 4
        S = L * PATCH_BITS_PER_RESIDUE
        per_res = torch.zeros(L, PATCH_BITS_PER_RESIDUE, dtype=torch.bool)
        per_res[:, SEQ_BITS_PER_RESIDUE:] = True  # structure observed
        clamp_mask = per_res.reshape(S)
        known = (torch.rand(S) > 0.5).float()
        clamp_full = torch.where(clamp_mask, known, torch.zeros(S))

        x, probs = sampler.sample(
            B,
            S,
            conditioning_prefix_full=clamp_full,
            cond_prefix_mask=clamp_mask,
            guidance_scale=0,
            num_steps=8,
            return_probs=True,
            progress=False,
        )
        self.assertEqual(tuple(x.shape), (B, S))
        target = clamp_full[clamp_mask][None, :].expand(B, -1)
        max_err = (x[:, clamp_mask] - target).abs().max().item()
        self.assertLess(max_err, 1e-4)


def _batch_tasks(tasks, L, seed=0):
    """Build a paired batch, one example per named whole-example task."""
    rng = np.random.default_rng(seed)
    B = len(tasks)
    x0 = rng.integers(0, 2, size=(B, L * PATCH_BITS_PER_RESIDUE)).astype(np.float32)
    states = np.zeros((B, L, 2), dtype=np.int64)
    for b, t in enumerate(tasks):
        states[b] = T.build_example_states(t, L)
    residue = np.ones((B, L), dtype=bool)
    seq_full, struct_full = T.full_modality_target_masks_np(states, residue)
    return {
        "x0": torch.from_numpy(x0),
        "states": torch.from_numpy(states),
        "seq_target_full": torch.from_numpy(seq_full),
        "struct_target_full": torch.from_numpy(struct_full),
    }


class EntropySinkTests(unittest.TestCase):
    """The multimodal step must report the per-modality (sigma, denoising-MSE)
    samples the online entropy schedule consumes, and skip modalities a task does
    not supervise (the single-modality marginals)."""

    def test_entropy_sink_reports_valid_per_modality_pairs(self):
        torch.manual_seed(0)
        cfg = _base_cfg(multimodal=True, patch_size=18)
        model = SequenceVDTContinuousModel(cfg)
        proc = ContinuousForwardProcess(cfg)
        tasks = ["joint", "sequence_marginal", "structure_marginal"]
        batch = _batch_tasks(tasks, L=6, seed=2)

        sink = {}
        loss, comps = multimodal_training_step(
            model, batch, proc, cfg, device=torch.device("cpu"),
            is_train=True, entropy_sink=sink,
        )
        self.assertTrue(np.isfinite(loss.item()))

        B = len(tasks)
        for k in (
            "sigma_seq", "sigma_struct", "metric_seq",
            "metric_struct", "valid_seq", "valid_struct",
        ):
            self.assertIn(k, sink)
            self.assertEqual(tuple(sink[k].shape), (B,))

        # valid flags follow the task modes: joint supervises both modalities,
        # sequence_marginal only sequence, structure_marginal only structure.
        self.assertEqual(sink["valid_seq"].tolist(), [True, True, False])
        self.assertEqual(sink["valid_struct"].tolist(), [True, False, True])

        # metrics are probability-space MSE in [0, 1], finite on the valid entries.
        for mkey, vkey in (("metric_seq", "valid_seq"), ("metric_struct", "valid_struct")):
            m = sink[mkey][sink[vkey]]
            self.assertTrue(torch.isfinite(m).all())
            self.assertTrue(bool((m >= 0).all()) and bool((m <= 1).all()))

        # What the trainer pushes: valid pairs from both modalities. joint->2,
        # sequence_marginal->1 (seq), structure_marginal->1 (struct) => 4 total.
        sig = torch.cat([
            sink["sigma_seq"][sink["valid_seq"]],
            sink["sigma_struct"][sink["valid_struct"]],
        ])
        met = torch.cat([
            sink["metric_seq"][sink["valid_seq"]],
            sink["metric_struct"][sink["valid_struct"]],
        ])
        self.assertEqual(sig.numel(), 4)
        self.assertEqual(met.numel(), 4)

    def test_sigma_draw_fn_is_honored(self):
        # The trainer passes its entropy-schedule sampler as sigma_draw_fn; the
        # step must draw both modalities from it (independent noise -> two calls).
        torch.manual_seed(0)
        cfg = _base_cfg(multimodal=True, patch_size=18)
        self.assertTrue(bool(cfg.model.independent_modality_noise))
        model = SequenceVDTContinuousModel(cfg)
        proc = ContinuousForwardProcess(cfg)
        batch = _batch_tasks(["joint", "joint"], L=5, seed=3)

        calls = {"n": 0}

        def draw_fn(bsz):
            calls["n"] += 1
            return torch.full((bsz,), 3.0)

        sink = {}
        multimodal_training_step(
            model, batch, proc, cfg, device=torch.device("cpu"),
            is_train=True, sigma_draw_fn=draw_fn, entropy_sink=sink,
        )
        self.assertEqual(calls["n"], 2)
        self.assertTrue(torch.allclose(sink["sigma_seq"], torch.full((2,), 3.0)))
        self.assertTrue(torch.allclose(sink["sigma_struct"], torch.full((2,), 3.0)))

    def test_sink_absent_is_unchanged(self):
        # Without an entropy_sink the step returns exactly (loss, components).
        torch.manual_seed(0)
        cfg = _base_cfg(multimodal=True, patch_size=18)
        model = SequenceVDTContinuousModel(cfg)
        proc = ContinuousForwardProcess(cfg)
        batch = _batch_tasks(["joint", "forward_folding"], L=5, seed=4)
        out = multimodal_training_step(
            model, batch, proc, cfg, device=torch.device("cpu"), is_train=True,
        )
        self.assertEqual(len(out), 2)
        loss, comps = out
        self.assertTrue(np.isfinite(loss.item()))
        self.assertIn("seq_bit_acc", comps)


if __name__ == "__main__":
    unittest.main()
