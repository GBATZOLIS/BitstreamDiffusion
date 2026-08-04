"""Round-trip and bit-order tests for the DPLM-2 LFQ structure codec.

These tests prove the codec's exact bit convention before any structure data
flows into the model. The default codec is ``msb_first`` and must match the
released DPLM-2 tokenizer exactly: ``LFQ.forward`` (which produced the released
token ids) and ``LFQ.get_codebook_entry`` (which detokenizes an id) both use the
descending mask ``2 ** arange(12, -1, -1)``, so latent dimension 0 is the MOST
significant bit of the released decimal index. ``test_matches_dplm_descending_``
``convention`` pins that; getting it wrong silently bit-reverses every released
M0 structure token.

Run with:
    python -m unittest tests.test_protein_structure_codec
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    NUM_LFQ_DIMS,
    PATCH_BITS_PER_RESIDUE,
    STRUCT_CODEBOOK_SIZE,
    LFQBitCodec,
    assemble_patch_np,
    build_full_roundtrip_fixture,
    split_patch_np,
)
from data.proteins import build_token_to_bits_table

# Hand-computed reference bit patterns in the default convention
# (msb_first: dimension 0 is the MOST significant bit; bit d is the coefficient
# of 2**(12 - d), matching the released DPLM decimal index).
HAND_FIXTURES = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    4096: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    8191: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # 4793 = 4096 + 512 + 128 + 32 + 16 + 8 + 1; MSB-first over 13 dims.
    4793: [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
}


class LFQConventionTests(unittest.TestCase):
    def test_hand_computed_bit_patterns(self):
        for idx, expected in HAND_FIXTURES.items():
            bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
                np.array(idx)
            ).tolist()
            self.assertEqual(
                bits, expected, f"index {idx} decoded to wrong bits"
            )

    def test_exhaustive_index_bits_roundtrip(self):
        ids = np.arange(STRUCT_CODEBOOK_SIZE, dtype=np.int64)
        bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(ids)
        self.assertEqual(bits.shape, (STRUCT_CODEBOOK_SIZE, NUM_LFQ_DIMS))
        self.assertTrue(set(np.unique(bits).tolist()) <= {0, 1})
        recovered = DEFAULT_STRUCT_CODEC.bits_to_index_np(bits)
        self.assertTrue(np.array_equal(ids, recovered))

    def test_exhaustive_index_signs_roundtrip(self):
        ids = np.arange(STRUCT_CODEBOOK_SIZE, dtype=np.int64)
        signs = DEFAULT_STRUCT_CODEC.index_to_signs_np(ids)
        self.assertTrue(set(np.unique(signs).tolist()) <= {-1.0, 1.0})
        recovered = DEFAULT_STRUCT_CODEC.signs_to_index_np(signs)
        self.assertTrue(np.array_equal(ids, recovered))

    def test_sign_mapping_matches_bit(self):
        # bit==1 -> +1, bit==0 -> -1 in the default convention.
        bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(np.array(4793))
        signs = DEFAULT_STRUCT_CODEC.bits_to_signs_np(bits)
        for b, s in zip(bits.tolist(), signs.tolist()):
            self.assertEqual(s, 1.0 if b == 1 else -1.0)

    def test_matches_dplm_descending_convention(self):
        # Pin the released DPLM convention directly: LFQ.forward and
        # get_codebook_entry use mask = 2 ** arange(12, -1, -1), so for a token
        # id n the sign of latent dim d is +1 iff bit (12 - d) of n is set. The
        # codec's index_to_bits must reproduce exactly that for every id.
        ids = np.arange(STRUCT_CODEBOOK_SIZE, dtype=np.int64)
        bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(ids)  # [8192, 13]
        shifts = (NUM_LFQ_DIMS - 1) - np.arange(NUM_LFQ_DIMS)  # [12, 11, ..., 0]
        expected = (ids[:, None] >> shifts[None, :]) & 1
        self.assertTrue(np.array_equal(bits.astype(np.int64), expected))
        # Consequently the default codec now coincides numerically with the
        # big-endian amino-acid table (both MSB-first); the modules stay separate
        # only for provenance and semantics, guarded by the convention hash.
        big_endian = build_token_to_bits_table(
            STRUCT_CODEBOOK_SIZE, NUM_LFQ_DIMS
        )
        self.assertTrue(
            np.array_equal(big_endian.numpy(), bits.astype(np.int64))
        )

    def test_default_codec_is_msb_first(self):
        self.assertEqual(DEFAULT_STRUCT_CODEC.bit_order, "msb_first")
        self.assertEqual(
            DEFAULT_STRUCT_CODEC.convention_hash(), "71220cdbbb22b4ff"
        )

    def test_torch_matches_numpy(self):
        ids = torch.arange(STRUCT_CODEBOOK_SIZE, dtype=torch.int64)
        bits_torch = DEFAULT_STRUCT_CODEC.index_to_bits_torch(ids)
        bits_np = DEFAULT_STRUCT_CODEC.index_to_bits_np(ids.numpy())
        self.assertTrue(np.array_equal(bits_torch.numpy(), bits_np))
        recovered = DEFAULT_STRUCT_CODEC.bits_to_index_torch(bits_torch)
        self.assertTrue(torch.equal(recovered, ids))

    def test_msb_first_variant_reverses(self):
        lsb = LFQBitCodec(bit_order="lsb_first")
        msb = LFQBitCodec(bit_order="msb_first")
        bits_lsb = lsb.index_to_bits_np(np.array(4793))
        bits_msb = msb.index_to_bits_np(np.array(4793))
        self.assertTrue(np.array_equal(bits_msb, bits_lsb[::-1]))
        # Both still round-trip within their own convention.
        self.assertEqual(int(msb.bits_to_index_np(bits_msb)), 4793)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            DEFAULT_STRUCT_CODEC.index_to_bits_np(
                np.array(STRUCT_CODEBOOK_SIZE)
            )
        with self.assertRaises(ValueError):
            DEFAULT_STRUCT_CODEC.index_to_bits_np(np.array(-1))


class PatchAssemblyTests(unittest.TestCase):
    def test_assemble_split_roundtrip(self):
        rng = np.random.default_rng(0)
        seq_bits = rng.integers(0, 2, size=(7, 5)).astype(np.uint8)
        struct_bits = rng.integers(0, 2, size=(7, 13)).astype(np.uint8)
        patch = assemble_patch_np(seq_bits, struct_bits)
        self.assertEqual(patch.shape, (7, PATCH_BITS_PER_RESIDUE))
        s2, z2 = split_patch_np(patch)
        self.assertTrue(np.array_equal(s2, seq_bits))
        self.assertTrue(np.array_equal(z2, struct_bits))

    def test_struct_bits_land_in_slots_5_to_17(self):
        seq_bits = np.zeros((1, 5), dtype=np.uint8)
        struct_bits = np.ones((1, 13), dtype=np.uint8)
        patch = assemble_patch_np(seq_bits, struct_bits)
        self.assertTrue(np.array_equal(patch[0, :5], np.zeros(5)))
        self.assertTrue(np.array_equal(patch[0, 5:], np.ones(13)))


class FixtureTests(unittest.TestCase):
    def test_full_roundtrip_fixture_is_exact_and_stable(self):
        fixture = build_full_roundtrip_fixture()
        self.assertTrue(fixture["exact_roundtrip"])
        self.assertEqual(fixture["codebook_size"], STRUCT_CODEBOOK_SIZE)
        # The convention hash is a stable pin; if the bit order ever changes this
        # value changes and downstream manifests would flag it.
        self.assertEqual(len(fixture["convention_hash"]), 16)
        self.assertEqual(fixture, build_full_roundtrip_fixture())


if __name__ == "__main__":
    unittest.main()
