# autoregression/train.py
from __future__ import annotations

import os
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "0")

# Allow running both:
#   python -m autoregression.train ...
# and:
#   python autoregression/train.py ...
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math
import argparse
import inspect
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_dataloaders
from utils.ema import EMA
from models.autoregressive import AutoregressiveGPT, ARGPTConfig

from autoregression.callbacks import maybe_generate_text8, maybe_generate_wikitext103

from autoregression.utils import (
    load_config,
    cfg_to_dict,
    maybe_init_distributed,
    destroy_distributed_if_needed,
    set_seed,
    unwrap_state_dict,
    save_json,
    count_params,
    human_params,
)


def _require(cfg, path: str):
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Missing required config field cfg.{path}")
        cur = getattr(cur, part)
    return cur

def _norm_ds(name: object) -> str:
    return str(name or "").strip().lower()


def build_argpt_config(cfg) -> ARGPTConfig:
    return ARGPTConfig(
        vocab_size=int(_require(cfg, "model.vocab_size")),
        max_seq_len=int(_require(cfg, "model.max_seq_len")),
        n_layer=int(_require(cfg, "model.n_layer")),
        n_head=int(_require(cfg, "model.n_head")),
        d_model=int(_require(cfg, "model.d_model")),
        mlp_mult=float(_require(cfg, "model.mlp_mult")),
        dropout=float(_require(cfg, "model.dropout")),
        rope_base=float(getattr(cfg.model, "rope_base", 10000.0)),
        use_flash_attn=bool(getattr(cfg.model, "use_flash_attn", True)),
    )


def build_optimizer(cfg, model):
    name = str(_require(cfg, "optim.optimizer")).lower()
    if name != "adamw":
        raise ValueError(f"Unknown optimizer {cfg.optim.optimizer}")

    kwargs = dict(
        lr=float(_require(cfg, "optim.lr")),
        betas=(float(_require(cfg, "optim.beta1")), float(_require(cfg, "optim.beta2"))),
        eps=float(getattr(cfg.optim, "eps", 1e-8)),
        weight_decay=float(_require(cfg, "optim.weight_decay")),
    )

    # fused AdamW if available
    sig = inspect.signature(torch.optim.AdamW)
    if torch.cuda.is_available() and ("fused" in sig.parameters):
        kwargs["fused"] = bool(getattr(cfg.optim, "fused", True))

    return torch.optim.AdamW(model.parameters(), **kwargs)


def lr_schedule(cfg, opt, step: int, total_steps: int) -> float:
    sch = str(getattr(cfg.optim, "scheduler", "cosine")).lower()
    warmup = int(getattr(cfg.optim, "warmup", 0))
    base_lr = float(_require(cfg, "optim.lr"))

    if warmup > 0 and step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        if sch == "constant":
            lr = base_lr
        elif sch == "cosine":
            t = (step - warmup) / max(1, total_steps - warmup)
            t = min(max(t, 0.0), 1.0)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown scheduler {sch}")

    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr



@torch.no_grad()
def validate_bits_per_token(model, loader, device, use_amp: bool, amp_dtype, is_distributed: bool) -> float:
    """
    Returns bits per predicted token (BPT).
    - For Text8 (char tokens): BPT == BPC.
    - For WikiText (BPE tokens): BPT is bits/token.
    """
    model.eval()
    sum_bits = torch.tensor(0.0, device=device)
    cnt_tok  = torch.tensor(0.0, device=device)

    is_master = (not is_distributed) or (dist.get_rank() == 0)
    pbar = tqdm(loader, desc="Validating", leave=False, disable=not is_master)

    for batch in pbar:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True).long()
        inp, tgt = x[:, :-1], x[:, 1:]
        B, Tm1 = tgt.shape

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(inp)
            loss_nats = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum",  # sum over predicted tokens
            )

        sum_bits += (loss_nats / math.log(2.0)).detach().float()
        cnt_tok  += float(B * Tm1)

        if is_master:
            # Optional: live display of running average
            pbar.set_postfix(bpt=(sum_bits / cnt_tok).item())

    if is_distributed:
        dist.all_reduce(sum_bits, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tok,  op=dist.ReduceOp.SUM)

    return (sum_bits / cnt_tok).item()



def _configure_sdpa_from_cfg(use_flash_attn: bool):
    """
    Configure SDPA backend preference once (global).
    This avoids per-layer context overhead.
    """
    if not torch.cuda.is_available():
        return

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

    # If user disables flash, leave defaults (still may use mem_efficient).
    if not use_flash_attn:
        return


def main():
    ap = argparse.ArgumentParser("Autoregressive (char-level) trainer")
    ap.add_argument("--config", required=True, help="Path to configs/autoregressive/...py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = maybe_init_distributed(cfg)

    is_distributed = bool(cfg.system.distributed)
    rank = int(cfg.system.global_rank)
    local_rank = int(cfg.system.local_rank)
    world_size = int(cfg.system.world_size)
    is_master = (rank == 0)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        cfg.device = str(device)
    else:
        device = torch.device("cpu")
        cfg.device = "cpu"

    # Perf knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    set_seed(int(_require(cfg, "train.seed")), bool(_require(cfg, "train.deterministic")))

    # representation constraints
    if str(_require(cfg, "data.representation")).lower() != "tokens":
        raise ValueError("For autoregressive char LM you must set cfg.data.representation='tokens'.")

    # ── sequence length check (token-generic, legacy-compatible) ─────────────
    expected = int(_require(cfg, "model.max_seq_len")) + 1

    seq_len_field = None
    if hasattr(cfg.data, "sequence_len_tokens"):
        seq_len_field = "sequence_len_tokens"
    elif hasattr(cfg.data, "sequence_len_chars"):
        seq_len_field = "sequence_len_chars"  # legacy name
    else:
        raise ValueError("Config must define cfg.data.sequence_len_tokens (or legacy cfg.data.sequence_len_chars).")

    got = int(_require(cfg, f"data.{seq_len_field}"))
    if got != expected:
        raise ValueError(
            f"cfg.data.{seq_len_field} must equal cfg.model.max_seq_len + 1 "
            f"({expected}), got {got}"
        )

    # loaders
    train_loader, val_loader, _ = get_dataloaders(cfg)
    drop_last_train = bool(getattr(cfg.data, "drop_last_train", True))

    if is_distributed:
        global_bsz = int(_require(cfg, "train.batch_size"))
        if global_bsz % world_size != 0:
            raise ValueError(f"Global batch_size {global_bsz} must be divisible by world_size {world_size}.")
        bsz_per_gpu = global_bsz // world_size

        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last_train,
        )
        val_sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=bsz_per_gpu,
            sampler=train_sampler,
            num_workers=int(_require(cfg, "data.num_workers")),
            pin_memory=bool(_require(cfg, "data.pin_memory")),
            prefetch_factor=int(_require(cfg, "data.prefetch_factor")) if int(_require(cfg, "data.num_workers")) > 0 else None,
            persistent_workers=int(_require(cfg, "data.num_workers")) > 0,
            drop_last=drop_last_train,
        )
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=bsz_per_gpu,
            sampler=val_sampler,
            num_workers=int(_require(cfg, "data.num_workers")),
            pin_memory=bool(_require(cfg, "data.pin_memory")),
            persistent_workers=int(_require(cfg, "data.num_workers")) > 0,
            drop_last=False,
        )

    # model
    mcfg = build_argpt_config(cfg)
    raw_model = AutoregressiveGPT(mcfg).to(device)

    # Configure SDPA backends once (global)
    _configure_sdpa_from_cfg(use_flash_attn=bool(mcfg.use_flash_attn))

    # EMA (optional)
    ema_decay = float(_require(cfg, "train.ema_decay"))
    use_ema = ema_decay > 0.0
    ema = EMA(raw_model, decay=ema_decay) if use_ema else None

    # DDP wrap
    model = raw_model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # torch.compile (optional)
    if bool(_require(cfg, "train.use_compile")) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(_require(cfg, "train.compile_mode")), fullgraph=False)
        except Exception:
            if is_master:
                print("[torch.compile] WARNING: compile failed; continuing in eager mode.")

    # optimizer / amp
    opt = build_optimizer(cfg, model)
    grad_clip = float(_require(cfg, "optim.grad_clip"))

    use_amp = bool(_require(cfg, "train.use_fp16"))
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    # BF16 does not need GradScaler; enabling it adds overhead.
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # grad accumulation
    grad_accum = int(getattr(cfg.train, "grad_accum_steps", 1))
    if grad_accum < 1:
        raise ValueError("cfg.train.grad_accum_steps must be >= 1")

    # run dirs
    run_dir = Path("runs") / str(_require(cfg, "experiment"))
    ckpt_dir = run_dir / "checkpoints_ar"
    writer = None

    if is_master:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(run_dir / "training_logs_ar"))
        save_json(cfg_to_dict(cfg), run_dir / "config_ar.json")
        n_params = count_params(raw_model)
        print(f"[AR] Model params: {human_params(n_params)} ({n_params})")
        print(f"[AR] max_seq_len={mcfg.max_seq_len} (sequence_len_tokens={expected})")
        print(
            f"[AR] use_flash_attn={mcfg.use_flash_attn} amp_dtype={amp_dtype} "
            f"scaler={'on' if scaler.is_enabled() else 'off'}"
        )

    # checkpoint bookkeeping
    save_last = bool(_require(cfg, "train.save_last"))
    save_top_k = int(_require(cfg, "train.save_top_k"))
    checkpoint_mode = str(_require(cfg, "train.checkpoint_mode")).lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("cfg.train.checkpoint_mode must be 'min' or 'max'")

    best_metric = math.inf if checkpoint_mode == "min" else -math.inf
    best_ckpts: List[dict] = []

    def is_better(metric: float) -> bool:
        return metric < best_metric if checkpoint_mode == "min" else metric > best_metric

    def checkpoint_path(name: str) -> Path:
        return ckpt_dir / f"{name}.pt"

    def save_ckpt(name: str, epoch: int, val_bpt: float, global_step_: int):
        if not is_master:
            return

        m_to_save = model
        if hasattr(m_to_save, "module"):
            m_to_save = m_to_save.module
        if hasattr(m_to_save, "_orig_mod"):
            m_to_save = m_to_save._orig_mod

        state = {
            "epoch": epoch,
            "global_step": global_step_,
            "model": m_to_save.state_dict(),
            "opt": opt.state_dict(),
            "best_metric": best_metric,
            "best_ckpts": best_ckpts,
            "val_bpt": float(val_bpt),
        }
        if use_ema and ema is not None:
            state["ema"] = ema.state_dict()

        torch.save(state, checkpoint_path(name))

    # resume
    global_step = 0
    start_epoch = 0
    last_path = checkpoint_path("last")
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu")
        sd = unwrap_state_dict(ckpt["model"])
        raw_model.load_state_dict(sd, strict=True)
        opt.load_state_dict(ckpt["opt"])
        if use_ema and ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_ckpts[:] = ckpt.get("best_ckpts", best_ckpts)

        # Backward-compatible last validation metric (optional)
        if "val_bpt" in ckpt:
            last_val = ckpt["val_bpt"]
            last_name = "val_bpt"
        elif "val_bpc" in ckpt:
            last_val = ckpt["val_bpc"]
            last_name = "val_bpc"
        else:
            last_val = None
            last_name = None

        if is_master:
            if last_val is None:
                print(f"[AR] Resumed from {last_path} at epoch={start_epoch}, step={global_step}")
            else:
                print(
                    f"[AR] Resumed from {last_path} at epoch={start_epoch}, step={global_step} "
                    f"({last_name}={float(last_val):.4f})"
                )


    # training loop
    epochs = int(_require(cfg, "train.epochs"))
    total_steps = epochs * max(1, (len(train_loader) // grad_accum))
    eval_with_ema = bool(getattr(cfg.train, "eval_with_ema", False))

    for epoch in range(start_epoch, epochs):
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        is_master_epoch = (not is_distributed) or (rank == 0)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, disable=not is_master_epoch)

        opt.zero_grad(set_to_none=True)
        accum_i = 0

        for batch in pbar:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).long()
            inp, tgt = x[:, :-1], x[:, 1:]

            lr = lr_schedule(cfg, opt, global_step, max(1, total_steps))

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(inp)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="mean",
                )
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_i += 1

            if accum_i == grad_accum:
                if scaler.is_enabled():
                    scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)

                if use_ema and ema is not None:
                    ema.update(raw_model)

                accum_i = 0
                global_step += 1

                if is_master_epoch and writer is not None:
                    bpt_step = (loss.detach().float() * grad_accum / math.log(2.0)).item()
                    writer.add_scalar("ar/loss_nats", (loss.item() * grad_accum), global_step)
                    writer.add_scalar("ar/bpt_train_step", bpt_step, global_step)
                    writer.add_scalar("ar/lr", lr, global_step)
                    if global_step % int(getattr(cfg.logging, "log_freq", 50)) == 0:
                        pbar.set_postfix(loss=f"{(loss.item()*grad_accum):.4f}", bpt=f"{bpt_step:.3f}")

        # validation (optionally on EMA weights)
        if eval_with_ema and use_ema and ema is not None:
            ema.apply(raw_model)

        val_bpt = validate_bits_per_token(model, val_loader, device, use_amp, amp_dtype, is_distributed)

        if eval_with_ema and use_ema and ema is not None:
            ema.restore(raw_model)

        if is_master_epoch and writer is not None:
            writer.add_scalar("ar/bpt_val", val_bpt, global_step)
            print(f"[AR] epoch={epoch} val_bpt={val_bpt:.4f}")


        dataset_name = _norm_ds(getattr(cfg.data, "dataset", ""))
        if dataset_name == "text8":
            maybe_generate_text8(
                raw_model=raw_model,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                device=device,
                run_dir=run_dir,
                writer=writer,
                is_master=is_master_epoch,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                ema=ema,
                use_ema_available=use_ema,
            )
        elif dataset_name in {"wikitext-103", "wikitext103", "wikitext"}:
            maybe_generate_wikitext103(
                raw_model=raw_model,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                device=device,
                run_dir=run_dir,
                writer=writer,
                is_master=is_master_epoch,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                ema=ema,
                use_ema_available=use_ema,
            )


        # checkpointing
        if save_last:
            save_ckpt("last", epoch, val_bpt, global_step)

        if save_top_k > 0 and is_better(val_bpt):
            best_metric = float(val_bpt)
            name = f"epoch={epoch:04d}-val={val_bpt:.4f}"
            best_ckpts.append({"path": f"{name}.pt", "metric": float(val_bpt), "epoch": int(epoch)})

            reverse = (checkpoint_mode == "max")
            best_ckpts.sort(key=lambda d: d["metric"], reverse=reverse)

            while len(best_ckpts) > save_top_k:
                worst = best_ckpts.pop(-1)
                try:
                    os.remove(ckpt_dir / worst["path"])
                except FileNotFoundError:
                    pass

            save_ckpt(name, epoch, val_bpt, global_step)
            save_ckpt("best", epoch, val_bpt, global_step)

            if is_master_epoch:
                print(f"✨ New best AR model: {name} (val_bpt={val_bpt:.4f})")

        if is_distributed:
            dist.barrier()

    destroy_distributed_if_needed(cfg)


if __name__ == "__main__":
    main()
