import torch
from data.task_codec import (
    bits_required, token_ids_to_bits, bits_to_token_ids, token_mask_to_bit_mask,
)
from utils.text_decode import bitstreams_to_token_ids_raw_binary


def test_bits_required():
    assert bits_required(12) == 4
    assert bits_required(49152) == 16
    assert bits_required(2) == 1


def test_roundtrip_sudoku_ids():
    ids = torch.arange(12).view(1, 12)
    bits = token_ids_to_bits(ids, 4)
    assert bits.shape == (1, 48)
    assert torch.equal(bits_to_token_ids(bits, 4), ids)


def test_roundtrip_random_16bit():
    ids = torch.randint(0, 49152, (8, 512))
    bits = token_ids_to_bits(ids, 16)
    assert torch.equal(bits_to_token_ids(bits, 16), ids)


def test_msb_first_ordering():
    # id 1 with width 4 -> 0001 (MSB-first), LSB at last index.
    bits = token_ids_to_bits(torch.tensor([[1]]), 4)
    assert bits.flatten().tolist() == [0, 0, 0, 1]
    # id 8 -> 1000
    bits8 = token_ids_to_bits(torch.tensor([[8]]), 4)
    assert bits8.flatten().tolist() == [1, 0, 0, 0]


def test_consistency_with_repo_decoder():
    # Our decode must agree with the repo's raw_binary decoder used at eval.
    ids = torch.randint(0, 49152, (4, 64))
    bits = token_ids_to_bits(ids, 16).float()
    repo = bitstreams_to_token_ids_raw_binary(bits, bits_per_token=16)
    assert torch.equal(repo.cpu(), ids.cpu())


def test_mask_expansion():
    m = torch.tensor([[True, False, True]])
    bm = token_mask_to_bit_mask(m, 4)
    assert bm.shape == (1, 12)
    assert bm.flatten().tolist() == [1,1,1,1, 0,0,0,0, 1,1,1,1]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("ALL CODEC TESTS PASSED")
