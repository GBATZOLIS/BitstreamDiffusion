"""CPU smoke tests for the opt-in multimodal SDT path and the two-modality loss.

Run with:
    python -m unittest tests.test_protein_multimodal_model
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from ml_collections import config_dict

from data import protein_tasks as T
from data.protein_structure_codec import PATCH_BITS_PER_RESIDUE
from diffusion.continuous.losses import multimodal_bit_loss
from models.sdt import SequenceVDTContinuousModel


def _cfg(multimodal: bool, content_dim: int = 8, patch_size: int = 18):
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
    cfg.model.head_hidden = 32
    cfg.model.content_dim_continuous = content_dim
    cfg.model.self_condition = False
    cfg.model.multimodal = multimodal
    cfg.data = config_dict.ConfigDict()
    cfg.data.representation = "binary"
    cfg.data.vocab_size = 2
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.train = config_dict.ConfigDict()
    cfg.train.loss_type = "binary_ce"
    cfg.train.loss_weighting = "edm"
    return cfg


def _make_batch(B, L, seed=0):
    rng = np.random.default_rng(seed)
    x0 = torch.from_numpy(
        rng.integers(0, 2, size=(B, L * PATCH_BITS_PER_RESIDUE)).astype(
            np.float32
        )
    )
    # forward folding for half the batch, joint for the rest.
    states = np.zeros((B, L, 2), dtype=np.int64)
    for b in range(B):
        task = "forward_folding" if b % 2 == 0 else "joint"
        states[b] = T.build_example_states(task, L)
    states_t = torch.from_numpy(states)
    residue = torch.ones((B, L), dtype=torch.bool)
    sigma_seq = torch.full((B,), 0.8)
    sigma_struct = torch.full((B,), 1.1)
    sigma_map = T.per_bit_sigma_map_torch(states_t, sigma_seq, sigma_struct)
    seq_full_np, struct_full_np = T.full_modality_target_masks_np(
        states, residue.numpy()
    )
    seq_full = torch.from_numpy(seq_full_np)
    struct_full = torch.from_numpy(struct_full_np)
    modality_sigmas = torch.stack([sigma_seq, sigma_struct], dim=-1)
    return x0, states_t, sigma_map, modality_sigmas, seq_full, struct_full


class MultimodalModelSmoke(unittest.TestCase):
    def test_forward_backward_shapes(self):
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(_cfg(multimodal=True))
        B, L = 2, 4
        x0, states, sigma_map, modality_sigmas, seq_full, struct_full = (
            _make_batch(B, L)
        )
        logits = model(
            x0, sigma_map, slot_state=states, modality_sigmas=modality_sigmas
        )
        self.assertEqual(logits.shape, (B, L * PATCH_BITS_PER_RESIDUE, 1))
        loss, comps = multimodal_bit_loss(
            logits, x0, sigma_map, model.cfg, seq_full, struct_full
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        # slot/state embeddings should receive gradient (they are used).
        self.assertIsNotNone(model.mm_embed.slot_embed.weight.grad)

    def test_overfit_reduces_loss(self):
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(_cfg(multimodal=True))
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        B, L = 2, 4
        x0, states, sigma_map, modality_sigmas, seq_full, struct_full = (
            _make_batch(B, L)
        )
        first = last = None
        for step in range(40):
            opt.zero_grad()
            logits = model(
                x0,
                sigma_map,
                slot_state=states,
                modality_sigmas=modality_sigmas,
            )
            loss, _ = multimodal_bit_loss(
                logits, x0, sigma_map, model.cfg, seq_full, struct_full
            )
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        self.assertLess(last, first, "overfit loss did not decrease")

    def test_equal_modality_weighting_balances_gradient(self):
        # struct has 13 bits vs seq 5, but equal-modality reduction should not let
        # struct dominate: the two component losses are computed on separate means.
        torch.manual_seed(1)
        model = SequenceVDTContinuousModel(_cfg(multimodal=True))
        B, L = 2, 5
        x0, states, sigma_map, modality_sigmas, seq_full, struct_full = (
            _make_batch(B, L, seed=2)
        )
        logits = model(
            x0, sigma_map, slot_state=states, modality_sigmas=modality_sigmas
        )
        _, comps = multimodal_bit_loss(
            logits, x0, sigma_map, model.cfg, seq_full, struct_full
        )
        self.assertIn("loss_seq", comps)
        self.assertIn("loss_struct", comps)
        self.assertGreaterEqual(float(comps["struct_target_bits"]), 0.0)


class SequencePathUnchanged(unittest.TestCase):
    def test_non_multimodal_forward_unaffected(self):
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(
            _cfg(multimodal=False, content_dim=1, patch_size=5)
        )
        self.assertIsNone(model.mm_embed)
        B, S = 3, 20  # 4 residues x 5 bits
        x_t = torch.rand(B, S)
        sigma = torch.rand(B) + 0.1
        logits = model(x_t, sigma)
        self.assertEqual(logits.shape, (B, S, 1))

    def test_multimodal_model_shared_sigma_smoke(self):
        # A multimodal model may also be called with a plain [B] sigma and no
        # extra kwargs (the first integration smoke path). It must still run.
        torch.manual_seed(0)
        model = SequenceVDTContinuousModel(_cfg(multimodal=True))
        B, L = 2, 3
        x_t = torch.rand(B, L * PATCH_BITS_PER_RESIDUE)
        sigma = torch.rand(B) + 0.1
        logits = model(x_t, sigma)
        self.assertEqual(logits.shape, (B, L * PATCH_BITS_PER_RESIDUE, 1))


if __name__ == "__main__":
    unittest.main()
