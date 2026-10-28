"""
End-to-end CPU tests for the multimodal path against a real tiny M0
fixture.
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
    from configs.proteins.proteins_400m import get_config

    cfg = get_config()
    cfg.data.sequence_replay_root = ""
    cfg.data.sequence_replay_fraction = 0.0
    cfg.data.use_mixture_schedule = False
    cfg.train.loss_type = "binary_ce"
    cfg.train.loss_weighting = "none"
    cfg.train.coupled_sigma_prob = 0.0
    cfg.cond.p_uncond = 0.0
    cfg.train.grad_accum_steps = 1
    cfg.evaluation.solver = "heun"
    cfg.evaluation.schedule = "karras"
    cfg.evaluation.guidance_scale = 0.0
    cfg.evaluation.stochastic.enabled = False
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
        from protein_fixture import build_fixture

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
        # Overfit a single fixed batch from the real collator so the
        # trend is unambiguous (task noise across batches would
        # otherwise mask it).
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
        # The collator draws tasks from a seed derived from the batch's
        # row ids, not a mutable counter, so it is correct under
        # DataLoader workers and resumes exactly (the sampler reproduces
        # the batch -> same tasks).
        ds = ProteinMultimodalDataset(self._dir, split="train")
        length = int(ds.lengths[0])
        idxs = [i for i in range(len(ds)) if int(ds.lengths[i]) == length]
        self.assertGreaterEqual(len(idxs), 8)
        b0 = [ds[i] for i in idxs[:4]]

        coll = MultimodalTaskCollator(seed=5)
        coll.set_epoch(0)
        t0 = coll(b0)["task_names"]

        # A fresh collator at a DIFFERENT counter (as a worker fork
        # would be) must produce the SAME tasks for the SAME batch ->
        # counter-independent.
        coll2 = MultimodalTaskCollator(seed=5)
        coll2.set_epoch(0)
        for _ in range(7):
            coll2._counter += 1
        self.assertEqual(coll2(b0)["task_names"], t0)

        # Distinct batches produce varied task assignments
        # (decorrelated).
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
        from evaluation.evaluation_drivers.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        cfg.evaluation.solver = "ddim_entropic"
        cfg.evaluation.guidance_scale = 1.5
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()

        seen = {}
        real_forward = model.forward

        def spy(
            x_t,
            sigma,
            x0_hat=None,
            *,
            slot_state=None,
            bit_state_ids=None,
            modality_sigmas=None,
            residue_mask=None,
        ):
            seen["slot_state"] = slot_state
            seen["bit_state_ids"] = bit_state_ids
            seen["modality_sigmas"] = modality_sigmas
            seen["sigma_is_per_bit"] = sigma.dim() == 2
            # residue_mask joined the contract with the length and
            # terminus conditioning: without it the model would take the
            # chain length from the padded token count rather than the
            # real one.
            seen["residue_mask"] = residue_mask
            return real_forward(
                x_t,
                sigma,
                x0_hat,
                slot_state=slot_state,
                bit_state_ids=bit_state_ids,
                modality_sigmas=modality_sigmas,
                residue_mask=residue_mask,
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
        self.assertEqual(tuple(seen["modality_sigmas"].shape), (4, 2))
        self.assertIsNotNone(seen.get("residue_mask"))
        self.assertEqual(tuple(seen["residue_mask"].shape), (4, L))
        # CFG batches conditional rows first and dropped-condition rows
        # second.
        states = seen["slot_state"]  # [B, L, 2]
        self.assertTrue(bool((states[:2, :, T.STRUCT] == T.OBSERVED).all()))
        self.assertTrue(bool((states[2:, :, T.STRUCT] == T.DROPPED).all()))
        self.assertTrue(bool((states[:, :, T.SEQ] == T.NOISY).all()))
        # Decoded outputs have the right shapes and valid structure ids.
        self.assertEqual(len(out["seq_strings"]), 2)
        self.assertTrue(all(len(s) == L for s in out["seq_strings"]))
        self.assertEqual(out["struct_index"].shape, (2, L))
        self.assertLess(int(out["struct_index"].max()), 8192)
        self.assertTrue(out["sampling"]["cfg_active"])
        self.assertEqual(out["sampling"]["solver"], "ddim_entropic")
        self.assertEqual(out["sampling"]["effective_guidance_scale"], 1.5)

    def test_ddim_preserves_sdedit_refinement_interface(self):
        from evaluation.evaluation_drivers.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        cfg.evaluation.solver = "ddim_entropic"
        torch.manual_seed(2)
        model = SequenceVDTContinuousModel(cfg).eval()
        length = 8
        observed = {
            "seq_ids": np.arange(length, dtype=np.int64) % 20,
            "struct_index": np.arange(length, dtype=np.int64),
        }
        out = generate(
            model,
            cfg,
            "joint",
            length,
            num_samples=2,
            device=torch.device("cpu"),
            observed=observed,
            sigma_start=0.5,
        )
        self.assertEqual(out["sampling"]["solver"], "ddim_entropic")
        self.assertEqual(out["bits"].shape, (2, length * 18))

    def test_generation_ignores_inherited_matched_filter_scaling(self):
        # The multimodal model is trained on raw logits, so generation
        # must force identity logit scaling regardless of the config's
        # inherited value. Two runs from the same seed -- one with the
        # sequence path's matched_filter_residual, one with 'none' --
        # must produce identical samples, and generate() must restore
        # the caller's config value.
        from evaluation.evaluation_drivers.proteins.generate_multimodal import generate
        from models.sdt import SequenceVDTContinuousModel

        cfg = _fixture_cfg(self._dir)
        cfg.evaluation.guidance_scale = 1.5
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(cfg).eval()

        cfg.model.continuous_logit_scaling = "matched_filter_residual"
        torch.manual_seed(123)
        out_a = generate(
            model, cfg, "joint", 30, num_samples=2, device=torch.device("cpu")
        )
        self.assertEqual(
            cfg.model.continuous_logit_scaling, "matched_filter_residual"
        )  # restored

        cfg.model.continuous_logit_scaling = "none"
        torch.manual_seed(123)
        out_b = generate(
            model, cfg, "joint", 30, num_samples=2, device=torch.device("cpu")
        )

        self.assertTrue(
            np.array_equal(out_a["struct_index"], out_b["struct_index"])
        )
        self.assertEqual(out_a["seq_strings"], out_b["seq_strings"])
        self.assertFalse(out_a["sampling"]["cfg_active"])
        self.assertEqual(out_a["sampling"]["effective_guidance_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
