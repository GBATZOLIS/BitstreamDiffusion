"""
Regression test for the CFG marker carve-out (`protect_mask`).

This guards a failure that is SILENT. The multimodal masking scheme clamps the
marker bits clean in every regime and never drops them under CFG dropout, so at
sampling time the unconditional branch must see the markers at their true codes.
Before `protect_mask` existed in this sampler, `_make_null_full` nulled the
whole conditioned prefix -- markers included -- to 0.5, a value the model never
saw in training. Nothing raised; guided image->text and text->image just quietly
degraded, and the published MNIST-Sum numbers are reported at guidance 2.0.

Run:  pytest tests/test_protect_mask.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "multimodal" / "mnist_sum_p28_small.py"


def _load_cfg():
    os.environ.setdefault("MS_DATA_ROOT", "datasets/mnist_sum_p28")
    spec = importlib.util.spec_from_file_location("_ms_cfg", CONFIG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


@pytest.fixture(scope="module")
def setup():
    from data import mnist_sum_codec as C

    cfg = _load_cfg()
    layout = C.resolve_layout(cfg)
    return cfg, layout


def _build(cfg, ref, cond_b, protect, S, B):
    from diffusion.continuous.samplers import _build_mask_conditioning

    return _build_mask_conditioning(
        cfg=cfg, B=B, S=S, device=torch.device("cpu"),
        conditioning_prefix_full=ref, cond_prefix_mask=cond_b,
        conditioning_prefix=None, cond_len_bits=None,
        protect_mask=protect, is_cont_tokens=False, vocab_size=2,
    )


@pytest.mark.parametrize("task", ["i2t", "t2i", "joint"])
def test_markers_survive_the_null_branch(setup, task):
    from evaluation.mnist_sum import task_masks

    cfg, layout = setup
    dev = torch.device("cpu")
    S, B = layout.total_bits, 4
    torch.manual_seed(0)
    ref = torch.randint(0, 2, (B, S)).float()

    cond, protect = task_masks(layout, dev, task)
    cond_b = cond.view(1, S).expand(B, S).contiguous()
    prot_b = protect.view(1, S).expand(B, S).contiguous()

    assert prot_b.any(), "the layout reports no marker bits"

    _, _, _, null_with = _build(cfg, ref, cond_b, prot_b, S, B)

    # The protected positions carry their TRUE codes, not the null value.
    assert torch.equal(null_with[prot_b], ref[prot_b]), (
        "marker bits were not restored in the CFG unconditional branch")

    # Everything else in the conditioned prefix is still nulled.
    body = cond_b & (~prot_b)
    if body.any():
        assert (null_with[body] == 0.5).all(), (
            "non-marker conditioning was not nulled; protect_mask is too broad")


def test_omitting_protect_mask_reproduces_the_bug(setup):
    """Pin the behaviour the carve-out fixes, so the test is known to bite."""
    from evaluation.mnist_sum import task_masks

    cfg, layout = setup
    dev = torch.device("cpu")
    S, B = layout.total_bits, 2
    torch.manual_seed(0)
    ref = torch.randint(0, 2, (B, S)).float()

    cond, protect = task_masks(layout, dev, "i2t")
    cond_b = cond.view(1, S).expand(B, S).contiguous()
    prot_b = protect.view(1, S).expand(B, S).contiguous()

    _, _, _, null_without = _build(cfg, ref, cond_b, None, S, B)
    assert (null_without[prot_b] == 0.5).all(), (
        "expected the un-protected path to null markers to 0.5")

    _, _, _, null_with = _build(cfg, ref, cond_b, prot_b, S, B)
    assert not torch.equal(null_without, null_with), (
        "protect_mask made no difference -- it is not being threaded through")


def test_protect_mask_shape_validation(setup):
    cfg, layout = setup
    dev = torch.device("cpu")
    S, B = layout.total_bits, 2
    ref = torch.zeros(B, S)
    cond_b = torch.ones(B, S, dtype=torch.bool)

    with pytest.raises(ValueError, match="protect_mask"):
        _build(cfg, ref, cond_b, torch.ones(S + 1, dtype=torch.bool), S, B)
    with pytest.raises(ValueError, match="protect_mask"):
        _build(cfg, ref, cond_b, torch.ones(B, S, 2, dtype=torch.bool), S, B)


def test_all_three_sampler_classes_accept_protect_mask():
    """The monolith has one sample() per class; a port can miss one."""
    import inspect

    from diffusion.continuous.samplers import (
        DDIMSampler, EulerMaruyamaSampler, HeunSampler)

    for cls in (DDIMSampler, HeunSampler, EulerMaruyamaSampler):
        params = inspect.signature(cls.sample).parameters
        assert "protect_mask" in params, (
            f"{cls.__name__}.sample() does not accept protect_mask")
