"""End-to-end CPU tests for the multimodal path against a real tiny M0 fixture.

These exercise the executable contract the plumbing smoke also covers, but fast
and deterministically for CI:

  * the real sharded loader + source/length sampler + task collator feed the
    real model and the loss trends down (data path learns);
  * the batch sampler and the task collator resume to the EXACT next batch /
    task draw from their saved state (gate 5);
  * production generation drives the model with the trained multimodal contract
    (per-residue states + two modality sigmas), not the sequence-only path
    (gate 4).

Run with:
    python -m unittest tests.test_protein_multimodal_e2e
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from data import protein_tasks as T
from data.protein_multimodal import (
    MultimodalTaskCollator,
    ProteinMultimodalDataset,
    get_multimodal_loader,
)
from data.protein_multimodal_sampler import SourceLengthBatchSampler
from data.protein_structure_codec import PATCH_BITS_PER_RESIDUE


def _fixture_cfg(shard_dir):
    from configs.proteins.multimodal_lfq18_smoke import get_config

    cfg = get_config()
    cfg.device = "cpu"
    cfg.data.shard_dir = shard_dir
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.min_len = 40
    cfg.data.max_len = 256
    cfg.model.use_flash_attn = False
    cfg.model.embed_dim = 32
    cfg.model.dim_ff = 64
    cfg.model.n_blocks = 2
    cfg.model.n_heads = 2
    cfg.model.content_dim_continuous = 8
    cfg.model.head_hidden = 16
    cfg.model.self_condition = False
    cfg.train.batch_size = 8
    cfg.train.steps_per_epoch = 20
    cfg.evaluation.num_sampling_steps = 8
    return cfg


class M0FixtureMixin:
    @classmethod
    def setUpClass(cls):
        from scripts.proteins.setup.build_m0_fixture import build_fixture

        cls._dir = tempfile.mkdtemp(prefix="m0e2e_")
        build_fixture(cls._dir, seed=0)


class DataPathLearnsTests(M0FixtureMixin, unittest.TestCase):
    def test_real_loader_step_trends_down(self):
        from diffusion.continuous.processes import ContinuousForwardProcess
        from models.sdt import SequenceVDTContinuousModel
        from trainers.multimodal_step import multimodal_training_step

        cfg = _fixture_cfg(self._dir)
        torch.manual_seed(0)
        loader = get_multimodal_loader(
            cfg, split="train", batch_size=8, shuffle=True, seed=1
        )
        # Overfit a single fixed batch from the real collator so the trend is
        # unambiguous (task noise across batches would otherwise mask it).
        batch = next(iter(loader))
        self.assertIn("x0", batch)
        self.assertEqual(batch["x0"].shape[1] % PATCH_BITS_PER_RESIDUE, 0)

        model = SequenceVDTContinuousModel(cfg)
        proc = ContinuousForwardProcess(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        torch.manual_seed(0)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss, comps = multimodal_training_step(
                model, batch, proc, cfg, device=torch.device("cpu")
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]))


class SamplerResumeTests(M0FixtureMixin, unittest.TestCase):
    def _sampler(self):
        ds = ProteinMultimodalDataset(self._dir, split="train")
        return SourceLengthBatchSampler(
            ds.lengths,
            ds.sources,
            batch_size=4,
            num_batches=20,
            seed=7,
        )

    def test_batch_sampler_resumes_to_exact_next_batch(self):
        bs = self._sampler()
        bs.set_epoch(0)
        full = list(iter(bs))
        self.assertEqual(len(full), 20)

        bs2 = self._sampler()
        bs2.set_epoch(0)
        # Resume at cursor 8 via the saved state_dict.
        bs2.load_state_dict({"seed": 7, "epoch": 0, "next_batch": 8})
        resumed = list(iter(bs2))
        self.assertEqual(resumed, full[8:])

    def test_collator_tasks_are_content_derived_and_worker_safe(self):
        # The collator draws tasks from a seed derived from the batch's row ids,
        # not a mutable counter, so it is correct under DataLoader workers and
        # resumes exactly (the sampler reproduces the batch -> same tasks).
        ds = ProteinMultimodalDataset(self._dir, split="train")
        length = int(ds.lengths[0])
        idxs = [i for i in range(len(ds)) if int(ds.lengths[i]) == length]
        self.assertGreaterEqual(len(idxs), 8)
        b0 = [ds[i] for i in idxs[:4]]

        coll = MultimodalTaskCollator(seed=5)
        coll.set_epoch(0)
        t0 = coll(b0)["task_names"]

        # A fresh collator at a DIFFERENT counter (as a worker fork would be) must
        # produce the SAME tasks for the SAME batch -> counter-independent.
        coll2 = MultimodalTaskCollator(seed=5)
        coll2.set_epoch(0)
        for _ in range(7):
            coll2._counter += 1
        self.assertEqual(coll2(b0)["task_names"], t0)

        # Distinct batches produce varied task assignments (decorrelated).
        coll3 = MultimodalTaskCollator(seed=5)
        coll3.set_epoch(0)
        batches = [
            [ds[i] for i in idxs[k : k + 2]]
            for k in range(0, min(len(idxs), 16), 2)
            if len(idxs[k : k + 2]) == 2
        ]
        vecs = {tuple(coll3(b)["task_names"]) for b in batches}
        self.assertGreater(len(vecs), 1)


class GenerationContractTests(M0FixtureMixin, unittest.TestCase):
    def test_generation_passes_multimodal_contract(self):
        from evaluation.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()

        seen = {}
        real_forward = model.forward

        def spy(x_t, sigma, x0_hat=None, *, slot_state=None, modality_sigmas=None):
            seen["slot_state"] = slot_state
            seen["modality_sigmas"] = modality_sigmas
            seen["sigma_is_per_bit"] = sigma.dim() == 2
            return real_forward(
                x_t,
                sigma,
                x0_hat,
                slot_state=slot_state,
                modality_sigmas=modality_sigmas,
            )

        model.forward = spy  # type: ignore[assignment]

        L = 24
        struct_index = np.random.default_rng(0).integers(0, 8192, size=L)
        out = generate(
            model,
            cfg,
            "inverse_folding",
            L,
            num_samples=2,
            device=torch.device("cpu"),
            observed={"struct_index": struct_index},
        )
        # The trained contract must reach the model at sampling time.
        self.assertIsNotNone(seen.get("slot_state"))
        self.assertIsNotNone(seen.get("modality_sigmas"))
        self.assertTrue(seen.get("sigma_is_per_bit"))
        self.assertEqual(tuple(seen["modality_sigmas"].shape), (2, 2))
        # inverse folding observes structure: every residue's struct state OBSERVED.
        states = seen["slot_state"]  # [B, L, 2]
        self.assertTrue(
            bool((states[:, :, T.STRUCT] == T.OBSERVED).all())
        )
        self.assertTrue(
            bool((states[:, :, T.SEQ] == T.NOISY).all())
        )
        # Decoded outputs have the right shapes and valid structure ids.
        self.assertEqual(len(out["seq_strings"]), 2)
        self.assertTrue(all(len(s) == L for s in out["seq_strings"]))
        self.assertEqual(out["struct_index"].shape, (2, L))
        self.assertLess(int(out["struct_index"].max()), 8192)

    def test_generation_ignores_inherited_matched_filter_scaling(self):
        # The multimodal model is trained on raw logits, so generation must force
        # identity logit scaling regardless of the config's inherited value. Two
        # runs from the same seed -- one with the sequence path's
        # matched_filter_residual, one with 'none' -- must produce identical
        # samples, and generate() must restore the caller's config value.
        from evaluation.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()

        cfg.model.continuous_logit_scaling = "matched_filter_residual"
        torch.manual_seed(123)
        out_a = generate(model, cfg, "joint", 30, num_samples=2, device=torch.device("cpu"))
        self.assertEqual(
            cfg.model.continuous_logit_scaling, "matched_filter_residual"
        )  # restored

        cfg.model.continuous_logit_scaling = "none"
        torch.manual_seed(123)
        out_b = generate(model, cfg, "joint", 30, num_samples=2, device=torch.device("cpu"))

        self.assertTrue(np.array_equal(out_a["struct_index"], out_b["struct_index"]))
        self.assertEqual(out_a["seq_strings"], out_b["seq_strings"])


if __name__ == "__main__":
    unittest.main()
