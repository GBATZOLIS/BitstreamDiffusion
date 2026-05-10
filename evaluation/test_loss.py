# evaluation/test_loss.py
import math
import torch
import torch.nn.functional as F

from diffusion.continuous.losses import binary_score_interpolation_loss


@torch.no_grad()
def evaluate_continuous_test_loss(cfg, model, proc, test_loader, device, *, max_batches=None):
    """
    Sanity metric: evaluate the *training loss* on the test split.

    Returns:
      test_loss: weighted training objective (matches cfg.train.loss_weighting)
      test_bpb:  unweighted BCE-with-logits per bit, converted to bits (divide by ln 2)
    """
    model.eval()

    use_amp = bool(getattr(cfg.train, "use_fp16", False)) and (device.type == "cuda")
    amp_dtype = torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        # keep fp16 unless you explicitly prefer bf16
        pass

    total_loss = 0.0
    total_bpb = 0.0
    n_batches = 0

    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        x0 = batch.to(device)

        # Expect [B,S] bits (Text8 semantic bits etc). If [B,1,S] or similar, flatten safely.
        x0 = x0.view(x0.size(0), -1).to(torch.float32)

        B = x0.size(0)

        # --- sigma sampling: matches your config ("log-uniform") ---
        sigma = proc.sample_sigma(B, strategy="log-uniform").to(device).to(torch.float32)  # [B]

        # --- forward corruption (typical continuous-score / EDM style) ---
        eps = torch.randn_like(x0)
        x = x0 + sigma.view(B, 1) * eps

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(x, sigma)  # [B,S] or [B,S,1]

        # 1) training objective (weighted)
        loss = binary_score_interpolation_loss(
            logits=logits,
            x0=x0,
            sigma=sigma,
            cfg=cfg,
            return_entropy_metric=False,
        )

        # 2) unweighted BCE per bit in bits (interpretability / sanity)
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits_b = logits.squeeze(-1)
        else:
            logits_b = logits
        bce = F.binary_cross_entropy_with_logits(
            logits_b.to(torch.float32),
            x0.to(torch.float32),
            reduction="mean",
        )
        bpb = bce / math.log(2.0)

        total_loss += float(loss.item())
        total_bpb += float(bpb.item())
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")

    return total_loss / n_batches, total_bpb / n_batches
