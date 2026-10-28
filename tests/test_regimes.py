from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.regimes import (
    ConditioningState,
    apply_condition_dropout_states,
    plan_from_bit_states,
    sample_conditioning_plan,
)


def test_text_audio_adapter_preserves_fixed_tts_mask():
    cfg = SimpleNamespace(
        data=SimpleNamespace(text_seq_len=2, speaker_seq_len=1),
        cond=SimpleNamespace(
            continuation_prefix=1,
            unconditional_rate=0.0,
            texttospeech_rate=1.0,
            speechtotext_rate=0.0,
            continuation_rate=0.0,
        ),
    )
    plan = sample_conditioning_plan(
        "text_audio", cfg, 3, 10, torch.device("cpu"), bits_per_token=2
    )
    expected = torch.zeros(3, 10, dtype=torch.bool)
    expected[:, :6] = True
    assert torch.equal(plan.clamp_mask, expected)
    assert torch.equal(plan.loss_mask, ~expected)
    assert not plan.protect_mask.any()


def _mnist_cfg():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "mnist_sum" / "mnist_sum_p28_small.py"
    os.environ.setdefault("MS_DATA_ROOT", "datasets/mnist_sum_p28")
    spec = importlib.util.spec_from_file_location("_regime_ms_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def test_mnist_adapter_matches_compatibility_api():
    from data.mnist_sum import mnist_sum_codec
    from utils.mnist_sum.utils import _sample_regimes_and_cond_masks

    cfg = _mnist_cfg()
    S = mnist_sum_codec.resolve_layout(cfg).total_bits
    torch.manual_seed(17)
    plan = sample_conditioning_plan(
        "mnist_sum", cfg, 8, S, torch.device("cpu")
    )
    torch.manual_seed(17)
    ids, clamp, protect = _sample_regimes_and_cond_masks(
        cfg, 8, S, torch.device("cpu")
    )
    assert torch.equal(plan.regime_ids, ids)
    assert torch.equal(plan.clamp_mask, clamp)
    assert torch.equal(plan.protect_mask, protect)
    assert torch.equal(plan.loss_mask, ~clamp)
    assert protect.any()


def test_state_plan_covers_protein_state_semantics():
    states = torch.tensor(
        [[
            int(ConditioningState.OBSERVED),
            int(ConditioningState.NOISY),
            int(ConditioningState.ABSENT),
            int(ConditioningState.DROPPED),
        ]]
    )
    plan = plan_from_bit_states(states)
    assert plan.clamp_mask.tolist() == [[True, False, False, False]]
    assert plan.loss_mask.tolist() == [[False, True, False, False]]
    assert plan.null_mask.tolist() == [[False, False, True, True]]


def test_shared_condition_dropout_only_rewrites_observed_states():
    states = torch.tensor(
        [
            [int(ConditioningState.OBSERVED), int(ConditioningState.NOISY)],
            [int(ConditioningState.OBSERVED), int(ConditioningState.ABSENT)],
        ]
    )
    out = apply_condition_dropout_states(states, torch.tensor([True, False]))
    assert out.tolist() == [
        [int(ConditioningState.DROPPED), int(ConditioningState.NOISY)],
        [int(ConditioningState.OBSERVED), int(ConditioningState.ABSENT)],
    ]
    assert torch.equal(states, torch.tensor([[1, 2], [1, 0]]))


def test_trainer_has_no_experiment_specific_regime_imports():
    trainer = (
        Path(__file__).resolve().parents[1] / "trainers" / "trainer.py"
    ).read_text(encoding="utf-8")
    assert "utils.mnist_sum.utils" not in trainer
    assert "utils.text_audio.utils" not in trainer
    assert "cond_mode == \"multimodal_mask\"" not in trainer
