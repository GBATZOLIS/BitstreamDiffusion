import math
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from data.proteins import (
    DistributedLengthBucketBatchSampler,
    FrozenDimaSwissProtDataset,
    bitstreams_to_token_ids,
    make_char_tokenizer,
)
from evaluation.protein_metrics import compute_protein_metrics, decode_token_ids
from evaluation.utils import _bits_from_probs
from evaluation.vlb import (
    _diffusion_loss_sum_binary,
    _recon_term_binary,
)
from utils.callbacks.vlb_bound import VLBBoundCallback


class _VariableLengthDataset(Dataset):
    def __init__(self):
        self.lengths = [2, 2, 3, 3, 3]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return torch.zeros(self.lengths[index])


class ProteinDecodingTests(unittest.TestCase):
    def setUp(self):
        self.tok = make_char_tokenizer()
        self.a = next(i for i, c in self.tok.id_to_char.items() if c == "A")
        self.c = next(i for i, c in self.tok.id_to_char.items() if c == "C")

    def test_strict_valid_framing(self):
        row = torch.tensor([[self.tok.bos_id, self.a, self.c, self.tok.eos_id, self.tok.pad_id]])
        record = decode_token_ids(row, self.tok, min_len=2)[0]
        self.assertTrue(record["strict_valid"])
        self.assertEqual(record["residues"], "AC")

    def test_invalid_code_and_tail_are_not_repaired(self):
        bad_code = self.tok.vocab_size
        rows = torch.tensor([
            [self.tok.bos_id, self.a, bad_code, self.tok.eos_id, self.tok.pad_id],
            [self.tok.bos_id, self.a, self.tok.eos_id, self.c, self.tok.pad_id],
        ])
        records = decode_token_ids(rows, self.tok)
        self.assertFalse(records[0]["strict_valid"])
        self.assertIn("invalid_code_before_eos", records[0]["failure_reasons"])
        self.assertFalse(records[1]["strict_valid"])
        self.assertIn("nonpad_after_eos", records[1]["failure_reasons"])

    def test_metrics_only_use_strict_valid_samples(self):
        rows = torch.tensor([
            [self.tok.bos_id, self.a, self.tok.eos_id, self.tok.pad_id],
            [self.tok.bos_id, self.a, self.tok.eos_id, self.c],
        ])
        metrics = compute_protein_metrics(rows, self.tok)
        self.assertEqual(metrics["num_valid_samples"], 1)
        self.assertEqual(metrics["valid_sample_fraction"], 0.5)
        self.assertEqual(metrics["unique_fraction_valid"], 1.0)

    def test_codebook_map_never_emits_unused_binary_codes(self):
        # Independent thresholding would produce 31, outside the 29-token codebook.
        probs = torch.full((3, 10), 0.999)
        bits = _bits_from_probs(
            probs,
            prefix_bits=None,
            cL_bits=0,
            decode_strategy="codebook_map",
            codebook_size=self.tok.vocab_size,
            bits_per_code=5,
        )
        ids = bitstreams_to_token_ids(bits, 5)
        self.assertTrue(torch.all(ids < self.tok.vocab_size))


    def test_grammar_map_emits_strict_sequence(self):
        probs = torch.rand((4, 40), generator=torch.Generator().manual_seed(7))
        bits = _bits_from_probs(
            probs,
            prefix_bits=None,
            cL_bits=0,
            decode_strategy="protein_grammar_map",
            codebook_size=self.tok.vocab_size,
            bits_per_code=5,
            valid_body_codes=sorted(self.tok.residue_ids),
            bos_code=self.tok.bos_id,
            eos_code=self.tok.eos_id,
            pad_code=self.tok.pad_id,
            min_body_codes=2,
        )
        ids = bitstreams_to_token_ids(bits, 5)
        records = decode_token_ids(ids, self.tok, min_len=2)
        self.assertTrue(all(record["strict_valid"] for record in records))


class FrozenDimaDatasetTests(unittest.TestCase):
    def test_binary_and_token_representations_are_aligned(self):
        from configs.proteins.dima35m_bitstream_smoke import get_config

        binary_cfg = get_config()
        binary_cfg.data.limit_train = 8
        binary = FrozenDimaSwissProtDataset(binary_cfg, split="train")
        token_cfg = get_config()
        token_cfg.data.limit_train = 8
        token_cfg.data.representation = "tokens"
        tokens = FrozenDimaSwissProtDataset(token_cfg, split="train")
        self.assertEqual(binary[0].numel(), tokens[0].numel() * 5)
        decoded = bitstreams_to_token_ids(binary[0], 5).squeeze(0)
        torch.testing.assert_close(decoded, tokens[0])

    def test_distributed_length_batches_match_across_ranks(self):
        lengths = [128] * 13 + [129] * 18
        samplers = [
            DistributedLengthBucketBatchSampler(
                lengths,
                batch_size=2,
                shuffle=True,
                seed=11,
                rank=rank,
                world_size=4,
            )
            for rank in range(4)
        ]
        batches = [list(sampler) for sampler in samplers]
        self.assertTrue(all(len(rows) == len(batches[0]) for rows in batches))
        for step in range(len(batches[0])):
            local_sizes = {len(rows[step]) for rows in batches}
            local_lengths = {
                tuple(lengths[index] for index in rows[step]) for rows in batches
            }
            self.assertEqual(len(local_sizes), 1)
            self.assertEqual(len(local_lengths), 1)


class VLBMaskTests(unittest.TestCase):
    def test_callback_preserves_same_length_batches(self):
        dataset = _VariableLengthDataset()
        base_sampler = DistributedLengthBucketBatchSampler(
            dataset.lengths,
            batch_size=2,
            shuffle=False,
            seed=7,
            rank=0,
            world_size=1,
        )
        base_loader = DataLoader(dataset, batch_sampler=base_sampler)
        trainer = SimpleNamespace(
            train_loader=base_loader,
            val_loader=base_loader,
            cfg=SimpleNamespace(
                train=SimpleNamespace(
                    seed=7,
                    vlb=SimpleNamespace(batch_size=2),
                ),
                data=SimpleNamespace(
                    num_workers=0,
                    pin_memory=False,
                    prefetch_factor=2,
                ),
            ),
        )
        loader = VLBBoundCallback()._get_split_loader(trainer, "val")
        shapes = [tuple(batch.shape) for batch in loader]
        self.assertEqual(shapes, [(2, 2), (2, 3), (1, 3)])

    def test_binary_reconstruction_mask_excludes_padding(self):
        logits = torch.zeros((1, 4))
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        loss = _recon_term_binary(logits, target, mask)
        self.assertAlmostEqual(loss.item(), 2.0 * math.log(2.0), places=6)

    def test_binary_diffusion_mask_excludes_padding(self):
        probs = torch.zeros((1, 2, 4))
        target = torch.ones((1, 4))
        mask = torch.tensor([[True, False, True, False]])
        loss = _diffusion_loss_sum_binary(probs, target, mask)
        torch.testing.assert_close(loss, torch.tensor([[2.0, 2.0]]))


if __name__ == "__main__":
    unittest.main()
