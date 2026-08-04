import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data.proteins import bitstreams_to_token_ids
from data.uniref50 import (
    EVODIFF_UNIREF50_ALPHABET,
    EVODIFF_UNIREF50_ARCHIVE_MD5,
    EvoDiffUniRef50Dataset,
)
from data.uniref50_sampler import DistributedRandomLengthBatchSampler
from evaluation.proteins.io import EvoDiffReferenceStore
from evaluation.proteins.metrics import basic_metrics


def _write_fixture(root: Path) -> None:
    data_dir = root / "uniref50"
    data_dir.mkdir(parents=True)
    sequences = [
        "ACDE",            # train
        "FGHIK",           # valid
        "LMNPQR",          # legacy test (must not be used)
        "STVWYBZX",        # rtest
        "ACDEFGHIKLMN",    # train, cropped in smoke tests
    ]
    offsets, position = [], 0
    with (data_dir / "consensus.fasta").open("wb") as handle:
        for sequence in sequences:
            offsets.append(position)
            row = sequence.encode("ascii") + b"\n"
            handle.write(row)
            position += len(row)
    np.savez(
        data_dir / "lengths_and_offsets.npz",
        seq_offsets=np.asarray(offsets, dtype=np.int64),
        ells=np.asarray([len(sequence) for sequence in sequences], dtype=np.int64),
    )
    with (data_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"train": [0, 4], "valid": [1], "test": [2], "rtest": [3]},
            handle,
        )
    manifest = {
        "source": {"archive_md5": EVODIFF_UNIREF50_ARCHIVE_MD5},
        "dataset": {"data_dir": "uniref50"},
    }
    with (root / "frozen_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def _config(root: Path, *, representation: str = "tokens", max_len: int = 20):
    return SimpleNamespace(
        data=SimpleNamespace(
            root=str(root),
            representation=representation,
            alphabet=EVODIFF_UNIREF50_ALPHABET,
            bits_per_token=5,
            min_len=1,
            max_len=max_len,
            limit_train=0,
            limit_eval=0,
        )
    )


class EvoDiffDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _write_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_public_splits_keep_validation_and_rtest_distinct(self):
        val = EvoDiffUniRef50Dataset(_config(self.root), split="val")
        test = EvoDiffUniRef50Dataset(_config(self.root), split="test")
        self.assertEqual(val.sequence(0), "FGHIK")
        self.assertEqual(test.sequence(0), "STVWYBZX")
        self.assertNotEqual(test.sequence(0), "LMNPQR")

    def test_binary_representation_uses_five_bits_for_26_symbols(self):
        tokens = EvoDiffUniRef50Dataset(
            _config(self.root, representation="tokens"), split="test"
        )
        binary = EvoDiffUniRef50Dataset(
            _config(self.root, representation="binary"), split="test"
        )
        self.assertEqual(binary[0].numel(), tokens[0].numel() * 5)
        np.testing.assert_array_equal(
            bitstreams_to_token_ids(binary[0], 5).squeeze(0).numpy(),
            tokens[0].numpy(),
        )

    def test_evaluation_crop_is_deterministic(self):
        dataset = EvoDiffUniRef50Dataset(
            _config(self.root, max_len=4), split="test"
        )
        self.assertEqual(dataset.sequence(0), "STVW")
        self.assertEqual(dataset.sequence(0), "STVW")

    def test_reference_store_matches_lengths_from_rtest(self):
        store = EvoDiffReferenceStore(self.root)
        references = store.length_matched([4, 8], seed=3)
        self.assertEqual(references, ["STVW", "STVWYBZX"])


class EvoDiffSamplerTests(unittest.TestCase):
    def test_ranks_receive_same_shape_and_disjoint_rows(self):
        lengths = np.asarray([100] * 32 + [200] * 32)
        samplers = [
            DistributedRandomLengthBatchSampler(
                lengths, batch_size=4, num_batches=3, seed=9,
                rank=rank, world_size=2,
            )
            for rank in range(2)
        ]
        batches = [list(sampler) for sampler in samplers]
        for step in range(3):
            left, right = batches[0][step], batches[1][step]
            self.assertEqual(len(left), len(right))
            self.assertEqual({lengths[i] for i in left}, {lengths[i] for i in right})
            self.assertTrue(set(left).isdisjoint(right))


class EvoDiffMetricTests(unittest.TestCase):
    def test_reports_extended_validity_separately_from_canonical(self):
        metrics = basic_metrics(["ACDE", "ACDX", "AC?"], ["ACDE", "ACDE", "ACDE"])
        self.assertAlmostEqual(metrics["valid_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["canonical_sequence_fraction"], 0.5)
        self.assertFalse(metrics["lengths_exactly_matched"])


if __name__ == "__main__":
    unittest.main()
