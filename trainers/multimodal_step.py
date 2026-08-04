"""Multimodal training step for the 18-bit paired model.

This is a self-contained step function the Trainer can call for the multimodal
protein path without disturbing the sequence-only ``_step_continuous``. It draws
independent sequence and structure noise levels, noises only the noisy target
bits (observed and absent bits stay clean because their per-bit sigma is 0),
runs the opt-in multimodal forward, and reduces the two modalities separately.

To wire it into ``trainers/trainer.py``: detect the collated dict batch (it has
an ``x0`` key) and route to ``multimodal_training_step`` instead of unpacking
``batch[0]``. The sequence-only path is unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from data.protein_tasks import per_bit_sigma_map_torch
from diffusion.continuous.losses import multimodal_bit_loss


def draw_modality_sigmas(
    proc,
    batch_size: int,
    strategy: str,
    *,
    independent: bool = True,
    draw_fn=None,
):
    """Draw ``(sigma_seq, sigma_struct)`` each ``[B]`` for the two modalities.

    When ``draw_fn`` is provided (the trainer's entropy-schedule sampler,
    ``EntropyScheduleController.draw_sigma``) it is used in place of the raw
    forward-process sampler, so training sigmas follow the online *entropic*
    schedule once it engages (before that ``draw_fn`` returns the same
    log-normal/EDM base, so behaviour is unchanged). Each modality is drawn with
    its own call so independent-noise decorrelation is exactly as before.
    """
    if draw_fn is not None:
        sigma_seq = draw_fn(batch_size)
        sigma_struct = draw_fn(batch_size) if independent else sigma_seq.clone()
        return sigma_seq, sigma_struct
    sigma_seq = proc.sample_sigma(batch_size, strategy=strategy)
    if independent:
        sigma_struct = proc.sample_sigma(batch_size, strategy=strategy)
    else:
        sigma_struct = sigma_seq.clone()
    return sigma_seq, sigma_struct


def multimodal_training_step(
    model,
    batch: Dict[str, object],
    proc,
    cfg,
    *,
    device=None,
    is_train: bool = True,
    sigma_draw_fn=None,
    entropy_sink: Dict[str, torch.Tensor] | None = None,
    diagnostics_sink: Dict[str, object] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run one multimodal denoising step; returns ``(loss, components)``.

    ``sigma_draw_fn`` (optional): callable ``bsz -> [bsz]`` used to draw the
    per-modality noise levels instead of ``proc.sample_sigma``. The trainer
    passes its entropy-schedule sampler here so the online entropic schedule
    drives the training-sigma distribution.

    ``entropy_sink`` (optional): a dict the caller owns; when given, this step
    fills it with the per-example, per-modality quantities the online entropy
    schedule consumes -- ``sigma_seq``/``sigma_struct`` (the modality noise
    levels ``[B]``), ``metric_seq``/``metric_struct`` (per-example unweighted
    denoising MSE over that modality's target bits ``[B]``) and
    ``valid_seq``/``valid_struct`` (``[B]`` bool, False when the example has no
    target bits for that modality, e.g. the single-modality marginal tasks). The
    trainer pushes the valid ``(sigma, metric)`` pairs into the entropy FIFO
    buffer. Purely additive: with ``entropy_sink=None`` the step is unchanged.

    ``diagnostics_sink`` (optional): receives per-example sufficient statistics
    for deterministic validation breakdowns by modality, task, and sigma bin.
    It never changes the optimized loss.
    """
    if device is None:
        device = next(model.parameters()).device
    x0 = batch["x0"].to(device).float()  # [B, S]
    states = batch["states"].to(device)  # [B, L, 2]
    seq_full = batch["seq_target_full"].to(device)  # [B, S]
    struct_full = batch["struct_target_full"].to(device)
    B = x0.size(0)

    strategy = str(
        getattr(cfg.train, "sigma_sampling_strategy", "log-uniform")
    )
    independent = bool(getattr(cfg.model, "independent_modality_noise", True))
    sigma_seq, sigma_struct = draw_modality_sigmas(
        proc, B, strategy, independent=independent, draw_fn=sigma_draw_fn
    )
    sigma_seq = sigma_seq.to(device)
    sigma_struct = sigma_struct.to(device)

    sigma_map = per_bit_sigma_map_torch(
        states, sigma_seq, sigma_struct
    )  # [B, S]
    noise = torch.randn_like(x0)
    x_t = (
        x0 + sigma_map * noise
    )  # observed/absent bits have sigma 0 -> stay clean
    modality_sigmas = torch.stack([sigma_seq, sigma_struct], dim=-1)  # [B, 2]

    # Self-conditioning (mirrors _step_continuous): with probability
    # self_condition_prob a no-grad forward produces a detached x0_hat estimate
    # that is fed back into the supervised forward. The Heun sampler self-conditions
    # the same way (evaluation/proteins/generate_multimodal._MultimodalDenoiser),
    # so training and generation must agree: without this the self-cond channels
    # would only ever see zeros in training yet a real estimate at inference.
    sc_enabled = bool(getattr(cfg.model, "self_condition", False))
    p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))
    x0_hat = None
    if sc_enabled and is_train and p_sc > 0.0:
        sc_mask = torch.rand(B, device=device) < p_sc  # [B]
        if bool(sc_mask.any()):
            x0_hat = torch.zeros_like(x_t)
            with torch.no_grad():
                logits_sc = model(
                    x_t,
                    sigma_map,
                    None,
                    slot_state=states,
                    modality_sigmas=modality_sigmas,
                )
                if logits_sc.dim() == 3:
                    logits_sc = logits_sc.squeeze(-1)
                est = torch.sigmoid(logits_sc.float()).detach().to(x_t.dtype)
            x0_hat[sc_mask] = est[sc_mask]

    logits = model(
        x_t,
        sigma_map,
        x0_hat,
        slot_state=states,
        modality_sigmas=modality_sigmas,
    )
    loss, components = multimodal_bit_loss(
        logits,
        x0,
        sigma_map,
        cfg,
        seq_full,
        struct_full,
        lambda_seq=float(getattr(cfg.model, "lambda_seq", 1.0)),
        lambda_struct=float(getattr(cfg.model, "lambda_struct", 1.0)),
    )

    # Optional validation-only sufficient statistics. Keep sums rather than
    # reduced means so the trainer can aggregate without batch-size or mask bias.
    if diagnostics_sink is not None:
        with torch.no_grad():
            lg = logits.squeeze(-1) if logits.dim() == 3 else logits
            probs = torch.sigmoid(lg.float())
            sq_err = (probs - x0.float()) ** 2
            correct = ((lg > 0) == (x0 > 0.5)).to(torch.float32)
            sigma2 = sigma_map.float() ** 2
            positive = sigma2 > 0
            safe = torch.where(positive, sigma2, torch.ones_like(sigma2))
            sd2 = float(cfg.diffusion.continuous.sigma_data) ** 2
            edm_weight = (safe + sd2) / (safe * sd2)
            edm_weight = torch.where(positive, edm_weight, torch.zeros_like(edm_weight))

            diagnostics_sink["sigma_seq"] = sigma_seq.detach()
            diagnostics_sink["sigma_struct"] = sigma_struct.detach()
            for name, mask in (("seq", seq_full), ("struct", struct_full)):
                m = mask.to(torch.float32)
                diagnostics_sink[f"{name}_mse_sum"] = (sq_err * m).sum(dim=1).detach()
                diagnostics_sink[f"{name}_edm_sum"] = (edm_weight * sq_err * m).sum(dim=1).detach()
                diagnostics_sink[f"{name}_correct_sum"] = (correct * m).sum(dim=1).detach()
                diagnostics_sink[f"{name}_count"] = m.sum(dim=1).detach()

    # Online entropy schedule: report each modality's noise level and its
    # per-example unweighted denoising MSE so the trainer can bin the entropy
    # rate (metric / sigma^2). The two modalities carry independent sigmas, so
    # each contributes its own (sigma, metric) sample; examples with no target
    # bits for a modality (marginal tasks) are flagged invalid and skipped.
    if entropy_sink is not None:
        with torch.no_grad():
            lg = logits.squeeze(-1) if logits.dim() == 3 else logits
            sq_err = (torch.sigmoid(lg.float()) - x0.float()) ** 2  # [B, S]
            seq_m = seq_full.to(torch.float32)
            struct_m = struct_full.to(torch.float32)
            seq_den = seq_m.sum(dim=1)  # [B]
            struct_den = struct_m.sum(dim=1)  # [B]
            entropy_sink["sigma_seq"] = sigma_seq.detach()
            entropy_sink["sigma_struct"] = sigma_struct.detach()
            entropy_sink["metric_seq"] = (sq_err * seq_m).sum(dim=1) / seq_den.clamp_min(1.0)
            entropy_sink["metric_struct"] = (sq_err * struct_m).sum(dim=1) / struct_den.clamp_min(1.0)
            entropy_sink["valid_seq"] = seq_den > 0
            entropy_sink["valid_struct"] = struct_den > 0

    return loss, components
