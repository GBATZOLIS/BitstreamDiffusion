"""Focused CPU tests for the M0 v4 scheduler and validation controls."""

from __future__ import annotations

import unittest

import torch

from configs.proteins.m0_v4_base000 import get_config as cfg000
from configs.proteins.m0_v4_base025 import get_config as cfg025
from configs.proteins.m0_v4_base050 import get_config as cfg050
from configs.proteins.m0_v4_base100 import get_config as cfg100
from trainers.trainer import (
    _ema_decay_for_step,
    _mm_accumulate_validation,
    _mm_finalize_validation,
    _mm_validation_accumulator,
)
from utils.schedule_controller import _blend_entropy_with_base


class EntropyBaseFloorTests(unittest.TestCase):
    def test_empty_bins_recover_base_mass(self):
        learned = torch.tensor([1.0, 0.0, 0.0])
        base = torch.tensor([0.2, 0.3, 0.5])
        count = torch.tensor([100.0, 0.0, 50.0])
        pdf = _blend_entropy_with_base(learned, base, count, 100, 0.0)
        self.assertAlmostEqual(float(pdf.sum()), 1.0, places=6)
        self.assertGreater(float(pdf[1]), 0.0)
        self.assertGreater(float(pdf[2]), 0.0)

    def test_pure_base_endpoint_and_invalid_fraction(self):
        learned = torch.tensor([0.9, 0.1])
        base = torch.tensor([0.25, 0.75])
        count = torch.tensor([100.0, 100.0])
        pdf = _blend_entropy_with_base(learned, base, count, 100, 1.0)
        self.assertTrue(torch.allclose(pdf, base))
        with self.assertRaises(ValueError):
            _blend_entropy_with_base(learned, base, count, 100, 1.01)

    def test_fraction_is_exact_final_mixture(self):
        learned = torch.tensor([0.8, 0.2])
        base = torch.tensor([0.2, 0.8])
        count = torch.tensor([100.0, 100.0])
        pdf = _blend_entropy_with_base(learned, base, count, 100, 0.25)
        expected = 0.75 * learned + 0.25 * base
        self.assertTrue(torch.allclose(pdf, expected))


class ValidationAggregationTests(unittest.TestCase):
    def test_balances_modalities_and_preserves_breakdowns(self):
        stats = _mm_validation_accumulator(
            torch.device("cpu"), ["joint", "forward_folding"]
        )
        diagnostics = {
            "sigma_seq": torch.tensor([0.005, 0.2]),
            "sigma_struct": torch.tensor([0.02, 2.0]),
            "seq_edm_sum": torch.tensor([2.0, 0.0]),
            "seq_mse_sum": torch.tensor([1.0, 0.0]),
            "seq_correct_sum": torch.tensor([1.0, 2.0]),
            "seq_count": torch.tensor([2.0, 2.0]),
            "struct_edm_sum": torch.tensor([4.0, 0.0]),
            "struct_mse_sum": torch.tensor([2.0, 0.0]),
            "struct_correct_sum": torch.tensor([3.0, 0.0]),
            "struct_count": torch.tensor([4.0, 0.0]),
        }
        _mm_accumulate_validation(
            stats, diagnostics, ["joint", "forward_folding"]
        )
        metrics = _mm_finalize_validation(stats)
        self.assertAlmostEqual(metrics["validation/modality/seq/mse"], 0.25)
        self.assertAlmostEqual(metrics["validation/modality/struct/mse"], 0.5)
        self.assertAlmostEqual(metrics["validation/balanced_mse"], 0.375)
        self.assertAlmostEqual(metrics["validation/modality/seq/bit_acc"], 0.75)
        self.assertAlmostEqual(metrics["validation/modality/struct/bit_acc"], 0.75)
        self.assertIn("validation/task/joint/seq/mse", metrics)
        self.assertIn("validation/sigma/seq/0p002_0p01/mse", metrics)
        self.assertNotIn(
            "validation/task/forward_folding/struct/mse", metrics
        )


class EmaRampTests(unittest.TestCase):
    def test_linear_decay_ramp(self):
        self.assertAlmostEqual(_ema_decay_for_step(0.99, 0.9999, 10_000, 0), 0.99)
        self.assertAlmostEqual(
            _ema_decay_for_step(0.99, 0.9999, 10_000, 5_000), 0.99495
        )
        self.assertAlmostEqual(
            _ema_decay_for_step(0.99, 0.9999, 10_000, 10_000), 0.9999
        )
        self.assertAlmostEqual(
            _ema_decay_for_step(0.99, 0.9999, 10_000, 20_000), 0.9999
        )


class M0V4ConfigTests(unittest.TestCase):
    def test_variants_are_unique_and_controlled(self):
        configs = [cfg000(), cfg025(), cfg050(), cfg100()]
        self.assertEqual(
            [float(c.train.entropy_base_fraction) for c in configs],
            [0.0, 0.25, 0.5, 1.0],
        )
        self.assertEqual(len({str(c.experiment) for c in configs}), 4)
        for cfg in configs:
            self.assertEqual(int(cfg.train.batch_size), 128)
            self.assertEqual(int(cfg.train.grad_accum_steps), 2)
            self.assertEqual(int(cfg.train.expected_world_size), 1)
            self.assertEqual(int(cfg.optim.total_steps), 250_000)
            self.assertAlmostEqual(float(cfg.optim.lr), 1.2e-4)
            self.assertEqual(str(cfg.train.checkpointing.metric), "balanced_mse")
            self.assertFalse(bool(cfg.train.early_stopping.enabled))


if __name__ == "__main__":
    unittest.main()
