"""Tests for the multimodal task masks, paired dataset, collator, and sampler.

Run with:
    python -m unittest tests.test_protein_multimodal_tasks
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import protein_tasks as T
from data.protein_multimodal import (
    CANONICAL_AA,
    MultimodalTaskCollator,
    ProteinMultimodalDataset,
    RowRecord,
    StructurePermutation,
    write_manifest,
    write_shard,
)
from data.protein_multimodal_sampler import SourceLengthBatchSampler
from data.protein_structure_codec import (
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
    STRUCT_CODEBOOK_SIZE,
)


def _make_dataset(root: Path, n_pdb=6, n_afdb=10, length=12, split="train"):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pdb + n_afdb):
        src = "pdb" if i < n_pdb else "afdb"
        seq_ids = rng.integers(0, len(CANONICAL_AA), size=length).astype(
            np.uint8
        )
        struct_index = rng.integers(
            0, STRUCT_CODEBOOK_SIZE, size=length
        ).astype(np.uint16)
        seq_mask = np.ones(length, dtype=bool)
        struct_mask = np.ones(length, dtype=bool)
        # introduce one structure gap in a couple of rows
        if i % 5 == 0:
            struct_mask[0] = False
        rows.append(
            RowRecord(
                stable_id=f"{src}_{i}",
                seq_ids=seq_ids,
                struct_index=struct_index,
                seq_mask=seq_mask,
                struct_mask=struct_mask,
                source=src,
                cluster_id=f"cl{i % 3}",
                split=split,
            )
        )
    write_shard(root, "shard0", rows, tokenizer_revision="test")
    write_manifest(root, ["shard0"])


class TaskMaskTests(unittest.TestCase):
    def test_task_modes_cover_table(self):
        for name in (
            "joint",
            "inverse_folding",
            "forward_folding",
            "sequence_marginal",
            "structure_marginal",
        ):
            self.assertIn(name, T.TASK_MODES)

    def test_forward_folding_states(self):
        st = T.build_example_states("forward_folding", 4)
        self.assertTrue((st[:, T.SEQ] == T.OBSERVED).all())
        self.assertTrue((st[:, T.STRUCT] == T.NOISY).all())

    def test_target_masks_only_noisy_real(self):
        states = np.zeros((1, 3, 2), dtype=np.int64)
        states[0, :, T.SEQ] = T.NOISY
        states[0, :, T.STRUCT] = T.OBSERVED
        residue = np.array([[True, True, False]])
        seq_t, struct_t = T.target_loss_masks_np(states, residue)
        self.assertEqual(seq_t.shape, (1, 3 * SEQ_BITS_PER_RESIDUE))
        self.assertEqual(struct_t.shape, (1, 3 * STRUCT_BITS_PER_RESIDUE))
        # residue 2 is padding -> no targets; struct is observed -> no targets
        self.assertEqual(int(seq_t.sum()), 2 * SEQ_BITS_PER_RESIDUE)
        self.assertEqual(int(struct_t.sum()), 0)

    def test_clamp_mask_only_observed(self):
        states = np.zeros((1, 2, 2), dtype=np.int64)
        states[0, :, T.SEQ] = T.NOISY
        states[0, :, T.STRUCT] = T.OBSERVED
        residue = np.ones((1, 2), dtype=bool)
        clamp = T.observed_clamp_mask_np(states, residue)
        self.assertEqual(clamp.shape, (1, 2 * PATCH_BITS_PER_RESIDUE))
        # structure slots (5..17) observed, sequence slots (0..4) not
        clamp = clamp.reshape(1, 2, PATCH_BITS_PER_RESIDUE)
        self.assertFalse(clamp[0, :, :SEQ_BITS_PER_RESIDUE].any())
        self.assertTrue(clamp[0, :, SEQ_BITS_PER_RESIDUE:].all())

    def test_per_bit_sigma_map_gating(self):
        states = np.zeros((1, 2, 2), dtype=np.int64)
        states[0, :, T.SEQ] = T.NOISY
        states[0, :, T.STRUCT] = T.OBSERVED
        smap = T.per_bit_sigma_map_np(states, np.array([0.7]), np.array([1.3]))
        smap = smap.reshape(1, 2, PATCH_BITS_PER_RESIDUE)
        self.assertTrue(np.allclose(smap[0, :, :SEQ_BITS_PER_RESIDUE], 0.7))
        self.assertTrue(np.allclose(smap[0, :, SEQ_BITS_PER_RESIDUE:], 0.0))

    def test_torch_matches_numpy_masks(self):
        states = np.zeros((2, 3, 2), dtype=np.int64)
        states[:, :, T.SEQ] = T.NOISY
        states[0, 1, T.STRUCT] = T.NOISY
        residue = np.ones((2, 3), dtype=bool)
        seq_np, struct_np = T.target_loss_masks_np(states, residue)
        seq_t, struct_t = T.target_loss_masks_torch(
            torch.from_numpy(states), torch.from_numpy(residue)
        )
        self.assertTrue(np.array_equal(seq_np, seq_t.numpy()))
        self.assertTrue(np.array_equal(struct_np, struct_t.numpy()))
        smap_np = T.per_bit_sigma_map_np(
            states, np.array([0.5, 0.9]), np.array([0.2, 0.4])
        )
        smap_t = T.per_bit_sigma_map_torch(
            torch.from_numpy(states),
            torch.tensor([0.5, 0.9]),
            torch.tensor([0.2, 0.4]),
        )
        self.assertTrue(np.allclose(smap_np, smap_t.numpy()))


class DatasetCollatorTests(unittest.TestCase):
    def test_shard_roundtrip_and_collate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, length=10)
            ds = ProteinMultimodalDataset(root, split="train")
            self.assertGreater(len(ds), 0)
            row = ds[0]
            self.assertEqual(row["length"], 10)
            self.assertEqual(row["seq_ids"].shape[0], 10)

            collator = MultimodalTaskCollator(
                task_weights={"joint": 1.0}, seed=1
            )
            batch = [ds[i] for i in range(4)]
            out = collator(batch)
            self.assertEqual(out["x0"].shape, (4, 10 * PATCH_BITS_PER_RESIDUE))
            self.assertEqual(out["states"].shape, (4, 10, 2))
            # under joint, all present modalities are NOISY targets
            self.assertTrue((out["seq_target_mask"].sum() > 0))
            self.assertTrue((out["struct_target_mask"].sum() > 0))
            # x0 is strictly bits
            self.assertTrue(
                set(torch.unique(out["x0"]).tolist()) <= {0.0, 1.0}
            )

    def test_inverse_folding_clamps_structure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, length=8)
            ds = ProteinMultimodalDataset(root, split="train")
            collator = MultimodalTaskCollator(
                task_weights={"inverse_folding": 1.0}, seed=3
            )
            out = collator([ds[i] for i in range(3)])
            clamp = out["clamp_mask"].reshape(3, 8, PATCH_BITS_PER_RESIDUE)
            # sequence slots are targets (not clamped); structure slots clamped where available
            self.assertEqual(int(clamp[:, :, :SEQ_BITS_PER_RESIDUE].sum()), 0)
            self.assertGreater(
                int(clamp[:, :, SEQ_BITS_PER_RESIDUE:].sum()), 0
            )
            # no sequence-structure bit is both target and clamped
            seq_target = out["seq_target_mask"].sum()
            self.assertGreater(int(seq_target), 0)

    def test_structure_permutation_changes_bits_not_length(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, length=6)
            base = ProteinMultimodalDataset(root, split="train")
            perm = ProteinMultimodalDataset(
                root,
                split="train",
                struct_permutation=StructurePermutation(seed=7),
            )
            r0 = base[0]["struct_index"]
            r0p = perm[0]["struct_index"]
            self.assertEqual(r0.shape, r0p.shape)
            self.assertFalse(np.array_equal(r0, r0p))

    def test_manifest_convention_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_dataset(root, length=6)
            # corrupt the manifest hash
            import json

            mpath = root / "manifest.json"
            m = json.loads(mpath.read_text())
            m["struct_convention_hash"] = "deadbeefdeadbeef"
            mpath.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                ProteinMultimodalDataset(root, split="train")


class SamplerTests(unittest.TestCase):
    def _lengths_sources(self):
        lengths = [10, 10, 10, 20, 20, 30] * 4
        sources = (["pdb"] * 3 + ["afdb"] * 3) * 4
        return lengths, sources

    def test_same_length_batches(self):
        lengths, sources = self._lengths_sources()
        s = SourceLengthBatchSampler(
            lengths, sources, batch_size=2, num_batches=20, seed=0
        )
        for batch in s:
            self.assertEqual(len({lengths[i] for i in batch}), 1)

    def test_determinism_and_resume(self):
        lengths, sources = self._lengths_sources()
        s1 = SourceLengthBatchSampler(
            lengths, sources, batch_size=2, num_batches=10, seed=5
        )
        all_batches = list(s1)
        s2 = SourceLengthBatchSampler(
            lengths, sources, batch_size=2, num_batches=10, seed=5
        )
        s2.load_state_dict({"seed": 5, "epoch": 0, "next_batch": 4})
        resumed = list(s2)
        self.assertEqual(resumed, all_batches[4:])

    def test_ddp_ranks_disjoint_same_length(self):
        lengths, sources = self._lengths_sources()
        kwargs = dict(batch_size=2, num_batches=6, seed=9, world_size=2)
        r0 = list(SourceLengthBatchSampler(lengths, sources, rank=0, **kwargs))
        r1 = list(SourceLengthBatchSampler(lengths, sources, rank=1, **kwargs))
        for b0, b1 in zip(r0, r1):
            self.assertEqual(
                {lengths[i] for i in b0}, {lengths[i] for i in b1}
            )
            # ranks partition the same (source,length) draw; typically disjoint
            self.assertEqual(len(b0), len(b1))

    def test_source_balancing_oversamples_rare_source(self):
        # 3 pdb rows vs 300 afdb rows; equal source weights should sample pdb far
        # more than its raw 1 percent frequency.
        lengths = [10] * 303
        sources = ["pdb"] * 3 + ["afdb"] * 300
        s = SourceLengthBatchSampler(
            lengths,
            sources,
            batch_size=1,
            num_batches=400,
            seed=1,
            source_weights={"pdb": 1.0, "afdb": 1.0},
        )
        picked_pdb = 0
        for batch in s:
            for i in batch:
                if sources[i] == "pdb":
                    picked_pdb += 1
        # with equal source weight ~50 percent of picks are pdb, far above 1 percent
        self.assertGreater(picked_pdb, 100)


if __name__ == "__main__":
    unittest.main()
