"""Multimodal training step for the 18-bit paired model."""

from __future__ import annotations

import torch

from data.protein_tasks import (
    apply_condition_dropout_torch,
    null_content_bit_mask_torch,
    per_bit_sigma_map_torch,
)
from diffusion.continuous.losses import multimodal_bit_loss


def draw_modality_sigmas(
    proc,
    batch_size: int,
    strategy: str,
    *,
    independent: bool = True,
    draw_fn=None,
    coupled_prob: float = 0.0,
    device=None,
):
    """Draw (sigma_seq, sigma_struct) each [B] for the two modalities."""
    if draw_fn is not None:
        sigma_seq = draw_fn(batch_size)
        sigma_struct = (
            draw_fn(batch_size) if independent else sigma_seq.clone()
        )
    else:
        sigma_seq = proc.sample_sigma(batch_size, strategy=strategy)
        if independent:
            sigma_struct = proc.sample_sigma(batch_size, strategy=strategy)
        else:
            sigma_struct = sigma_seq.clone()

    if independent and coupled_prob > 0.0:
        dev = device if device is not None else sigma_seq.device
        couple = torch.rand(batch_size, device=dev) < float(coupled_prob)
        couple = couple.to(sigma_struct.device)
        sigma_struct = torch.where(couple, sigma_seq, sigma_struct)
    return sigma_seq, sigma_struct


def multimodal_training_step(
    model,
    batch: dict[str, object],
    proc,
    cfg,
    *,
    device=None,
    is_train: bool = True,
    sigma_draw_fn=None,
    entropy_sink: dict[str, torch.Tensor] | None = None,
    diagnostics_sink: dict[str, object] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one multimodal denoising step; returns (loss, components)."""
    if device is None:
        device = next(model.parameters()).device
    x0 = batch["x0"].to(device).float()  # [B, S]
    states = batch["states"].to(device)  # [B, L, 2]
    seq_full = batch["seq_target_full"].to(device)  # [B, S]
    struct_full = batch["struct_target_full"].to(device)
    residue_mask = batch.get("residue_mask")
    if residue_mask is not None:
        residue_mask = residue_mask.to(device)
    B = x0.size(0)

    strategy = str(
        getattr(cfg.train, "sigma_sampling_strategy", "log-uniform")
    )
    independent = bool(getattr(cfg.model, "independent_modality_noise", True))
    coupled_prob = float(getattr(cfg.train, "coupled_sigma_prob", 0.0) or 0.0)
    sigma_seq, sigma_struct = draw_modality_sigmas(
        proc,
        B,
        strategy,
        independent=independent,
        draw_fn=sigma_draw_fn,
        coupled_prob=coupled_prob if is_train else 0.0,
        device=device,
    )
    sigma_seq = sigma_seq.to(device)
    sigma_struct = sigma_struct.to(device)

    # ------------------------------------------------------------------
    # Classifier-free guidance: drop the condition on a fraction of
    # examples.
    #
    # Plan section 6.5. cfg.cond.p_uncond was read only by the
    # sequence-only
    # _step_continuous; the multimodal step never touched cfg.cond at all, so
    # the model had never seen a dropped condition. That is why turning
    # guidance up today would make results WORSE: the sampler's
    # "unconditional" branch is not unconditional, so the guidance
    # direction is a difference between two conditional predictions.
    #
    # Dropping rewrites OBSERVED -> DROPPED for the whole example and
    # blanks the content at those bits to the null constant, which is
    # exactly what the sampler's unconditional half will present at
    # inference.
    # ------------------------------------------------------------------
    # Reuse the repository-wide CFG probability. Protein conditioning is
    # represented by modality-state grids rather than a prefix, so the
    # probability is shared while the state rewrite below is protein-specific.
    p_cfg_drop = float(getattr(cfg.cond, "p_uncond", 0.0) or 0.0)
    if is_train and p_cfg_drop > 0.0:
        drop = torch.rand(B, device=device) < p_cfg_drop
        if bool(drop.any()):
            states = apply_condition_dropout_torch(states, drop)
            null_bits = null_content_bit_mask_torch(states)  # [B, S]
            null_value = float(
                getattr(
                    getattr(
                        getattr(cfg, "diffusion", None), "continuous", None
                    ),
                    "data_center",
                    0.5,
                )
            )
            x0 = torch.where(null_bits, torch.full_like(x0, null_value), x0)

    sigma_map = per_bit_sigma_map_torch(
        states, sigma_seq, sigma_struct
    )  # [B, S]
    noise = torch.randn_like(x0)
    x_t = (
        x0 + sigma_map * noise
    )  # observed/absent/dropped bits have sigma 0 -> stay clean
    modality_sigmas = torch.stack([sigma_seq, sigma_struct], dim=-1)  # [B, 2]

    # Self-conditioning (mirrors _step_continuous): with probability
    # self_condition_prob a no-grad forward produces a detached x0_hat
    # estimate that is fed back into the supervised forward. The Heun
    # sampler self-conditions the same way
    # (evaluation/proteins/generate_multimodal._MultimodalDenoiser), so
    # training and generation must agree: without this the self-cond
    # channels would only ever see zeros in training yet a real estimate
    # at inference.
    sc_enabled = bool(getattr(cfg.model, "self_condition", False))
    p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))
    # Self-conditioning is disabled during validation by the is_train
    # gate, which means best.pt was being chosen in a regime the sampler
    # never uses while sampling self-conditions on every one of its 250
    # steps. Validation now runs it too; set
    # validation_self_condition=False to recover the old behaviour for a
    # like-for-like comparison.
    sc_in_val = bool(getattr(cfg.train, "validation_self_condition", True))
    x0_hat = None
    if sc_enabled and p_sc > 0.0 and (is_train or sc_in_val):
        sc_mask = torch.rand(B, device=device) < p_sc  # [B]
        if bool(sc_mask.any()):
            # Only the selected rows need the extra forward. Running it
            # on the whole batch made the estimate pass unconditional in
            # practice (P(any) = 1 - 2^-B), so training was 4x forward
            # rather than 3.5x and every compute estimate was ~14% low.
            # Slicing is a free ~12% saving at p_sc=0.5.
            idx = sc_mask.nonzero(as_tuple=True)[0]
            x0_hat = torch.zeros_like(x_t)
            with torch.no_grad():
                logits_sc = model(
                    x_t[idx],
                    sigma_map[idx],
                    None,
                    slot_state=states[idx],
                    modality_sigmas=modality_sigmas[idx],
                    residue_mask=(
                        None if residue_mask is None else residue_mask[idx]
                    ),
                )
                if logits_sc.dim() == 3:
                    logits_sc = logits_sc.squeeze(-1)
                est = torch.sigmoid(logits_sc.float()).detach().to(x_t.dtype)
            x0_hat[idx] = est

    logits = model(
        x_t,
        sigma_map,
        x0_hat,
        slot_state=states,
        modality_sigmas=modality_sigmas,
        residue_mask=residue_mask,
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
        global_normalizer=bool(
            getattr(cfg.train, "global_loss_normalizer", True)
        ),
    )

    # Optional validation-only sufficient statistics. Keep sums rather
    # than reduced means so the trainer can aggregate without batch-size
    # or mask bias.
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
            edm_weight = torch.where(
                positive, edm_weight, torch.zeros_like(edm_weight)
            )

            diagnostics_sink["sigma_seq"] = sigma_seq.detach()
            diagnostics_sink["sigma_struct"] = sigma_struct.detach()
            for name, mask in (("seq", seq_full), ("struct", struct_full)):
                m = mask.to(torch.float32)
                diagnostics_sink[f"{name}_mse_sum"] = (
                    (sq_err * m).sum(dim=1).detach()
                )
                diagnostics_sink[f"{name}_edm_sum"] = (
                    (edm_weight * sq_err * m).sum(dim=1).detach()
                )
                diagnostics_sink[f"{name}_correct_sum"] = (
                    (correct * m).sum(dim=1).detach()
                )
                diagnostics_sink[f"{name}_count"] = m.sum(dim=1).detach()

    # Online entropy schedule: report each modality's noise level and
    # its per-example unweighted denoising MSE so the trainer can bin
    # the entropy rate (metric / sigma^2). The two modalities carry
    # independent sigmas, so each contributes its own (sigma, metric)
    # sample; examples with no target bits for a modality (marginal
    # tasks) are flagged invalid and skipped.
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
            entropy_sink["metric_seq"] = (sq_err * seq_m).sum(
                dim=1
            ) / seq_den.clamp_min(1.0)
            entropy_sink["metric_struct"] = (sq_err * struct_m).sum(
                dim=1
            ) / struct_den.clamp_min(1.0)
            entropy_sink["valid_seq"] = seq_den > 0
            entropy_sink["valid_struct"] = struct_den > 0

    return loss, components
