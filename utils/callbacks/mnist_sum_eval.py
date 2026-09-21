"""
utils/callbacks/mnist_sum_eval.py

Training-time callback for the MNIST-Sum corpus.

Runs both conditional directions on a held-out batch and logs to TensorBoard:

  image -> text : generate the equation from the image, parse it, and report the
                  EXACT three-way decomposition (perception / arithmetic /
                  end_to_end, plus per-slot and malformed rate) as scalars, with
                  a sample table of predicted-vs-true strings.
  text -> image : generate the image from the equation and log a grid, with the
                  prompt equations as a caption table. Scored visually for now;
                  an exact score needs a quadrant classifier, which is a separate
                  final-evaluation step.

The i2t accuracy is the reason this corpus exists: it is a real number with a
unique correct answer, available every time the callback fires, unlike FID or
CIDEr which only become meaningful at the end of a full protocol run.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, List, Optional

import torch

from utils.callbacks.base import Callback

from data import mnist_sum_codec as C
from evaluation.mnist_sum import ms_sample, score_i2t, parse_equation


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _gs_suffix(index: int, gs: float) -> str:
    """Tag suffix for a guidance setting: the first is primary and keeps the
    bare tag, so mnist_sum/i2t stays the headline curve."""
    if index == 0:
        return ""
    if abs(gs - 1.0) < 1e-8:
        return "_noguid"
    return f"_g{gs:g}"


def _find_ddp(model):
    """Unwrap torch.compile / DDP layers to the DDP instance, if any."""
    m = model
    for _ in range(4):
        if isinstance(m, torch.nn.parallel.DistributedDataParallel):
            return m
        m = getattr(m, "_orig_mod", None)
        if m is None:
            return None
    return None


@contextmanager
def _no_buffer_broadcast(model):
    """Suppress DDP's pre-forward buffer broadcast for the duration of the block.

    This callback is registered rank-0-only, so rank 0 runs sampler forwards
    while every other rank is already waiting in the epoch-end dist.barrier().
    DDP.forward broadcasts module buffers when broadcast_buffers is on and
    require_forward_param_sync is still set -- a collective with no matching
    call on the other ranks, i.e. a deadlock until the 60-minute process-group
    timeout fires.

    In practice the epoch-end path survives only by accident: _validate_epoch
    runs no_grad forwards on all ranks immediately before callbacks, and a
    no_grad forward clears require_forward_param_sync, so the broadcast is
    skipped. Verified both ways on 2x A6000 -- without that preceding all-rank
    pass the same pattern deadlocks in _distributed_broadcast_coalesced. Relying
    on it means one refactor of the validation step turns a viz callback into a
    60-minute hang partway through a multi-day run.

    Suppressing the broadcast is a semantic no-op for this model: every buffer
    (the sinusoidal time table, RoPE inv_freq, segment_ids) is a deterministic
    constant built at init and never mutated by training, so the ranks cannot
    diverge. No-op at world_size=1, where there is no DDP to find.
    """
    ddp = _find_ddp(model)
    if ddp is None:
        yield
        return
    prev = ddp.broadcast_buffers
    ddp.broadcast_buffers = False
    try:
        yield
    finally:
        ddp.broadcast_buffers = prev


def _writer(trainer) -> Optional[Any]:
    for attr in ("writer", "tb_writer", "summary_writer"):
        w = getattr(trainer, attr, None)
        if w is not None:
            return w
    return None


class MNISTSumEvalCallback(Callback):
    """Exact image->text accuracy + text->image samples, every N steps."""

    # Run on EVERY rank, not just the master, matching VisualizationCallback /
    # VLBBoundCallback / ExternalPPLCallback. Only rank 0 writes anything (see
    # the _rank0() guards), so the extra ranks' sampling is discarded -- that
    # waste buys DDP safety, and at world_size=1 there is no extra rank at all.
    #
    # Sampling rank-0-only through the compiled DDP module deadlocks the NEXT
    # training step. The eval drives shapes training never sees (the 2B CFG
    # batch, no_grad), so rank 0 recompiles and the other ranks do not --
    # measured on 2x A6000 with TORCH_LOGS=recompiles: 28 recompiles on rank 0
    # against 21 on rank 1, triggered by "x0_hat size mismatch 8 -> 16",
    # "sigma stride mismatch" and "GLOBAL_STATE changed: grad_mode". Since
    # dynamo's DDPOptimizer splits the graph into gradient-allreduce buckets, a
    # graph that exists on one rank only means a different collective sequence
    # there, and the ranks wedge on the next backward -- both spinning at 100%
    # GPU, no error, until the 60-minute process-group timeout.
    #
    # Running everywhere makes the recompiles symmetric, which is the actual
    # fix; the buffer-broadcast guard below stays as belt-and-braces.
    run_on_all_ranks = True

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._last_bucket = -1
        self._holdout_ds = None
        self._holdout_missing = False
        self._did_sanity = False

    # ---- config -----------------------------------------------------------
    def _c(self, cfg):
        return getattr(getattr(cfg, "train", object()), "mnist_sum_eval", None)

    def _enabled(self, cfg) -> bool:
        c = self._c(cfg)
        return bool(getattr(c, "enabled", False)) if c is not None else False

    def _every(self, cfg) -> int:
        return int(getattr(self._c(cfg), "every_k_steps", 50_000))

    # ---- hooks ------------------------------------------------------------
    # The trainer only exposes on_epoch_end, so we gate on global_step and fire
    # the first time an epoch boundary lands past the next multiple of
    # every_k_steps. With 1M rows at batch 512 an epoch is ~1953 steps, so a
    # 50k-step cadence fires roughly every 26 epochs.
    def on_epoch_end(self, trainer, epoch: int):
        cfg = trainer.cfg
        if not self._enabled(cfg):
            return
        step = int(getattr(trainer, "global_step", 0))
        every = self._every(cfg)
        if every <= 0:
            return
        bucket = step // every
        if step == 0:
            # The pre-train sanity pass runs on_epoch_end once at global_step 0.
            # Fire for it (that is the whole point of a sanity pass -- an early
            # warning before committing to a long run) but do NOT claim bucket 0,
            # or the first real evaluation would be suppressed as a repeat.
            if self._did_sanity:
                return
            self._did_sanity = True
        elif bucket <= self._last_bucket:
            return
        else:
            self._last_bucket = bucket
        applied = False
        try:
            ema = getattr(trainer, "ema", None)
            if ema is not None and bool(getattr(self._c(cfg), "use_ema", True)):
                ema.apply(trainer.model); applied = True
            with _no_buffer_broadcast(trainer.model):
                self.run(trainer, step)
        except Exception as e:      # never kill a training run for a viz failure
            if _rank0():
                import traceback
                print(f"[mnist-sum-eval] skipped at step {step}: "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
        finally:
            if applied:
                trainer.ema.restore(trainer.model)

    # ---- splits -----------------------------------------------------------
    def _holdout(self, trainer, ds):
        """Dataset for the never-seen-tuple split.

        get_dataloaders() builds `val` twice and returns it as both the val and
        test loader, so the val_holdout split the builder writes is otherwise
        scored by nothing -- and the memorisation-vs-generalisation gap is the
        point of having the split. Instantiate it here rather than changing the
        loader triple, which other consumers of test_loader would inherit.
        """
        if self._holdout_ds is not None or self._holdout_missing:
            return self._holdout_ds
        try:
            self._holdout_ds = type(ds)(trainer.cfg, split="val_holdout")
        except Exception as e:
            self._holdout_missing = True
            if _rank0():
                print(f"[mnist-sum-eval] no val_holdout split "
                      f"({type(e).__name__}: {e}); generalisation gap not logged",
                      flush=True)
        return self._holdout_ds

    # ---- image -> text : the exact metric ---------------------------------
    @torch.no_grad()
    def _score_i2t(self, *, cfg, c, model, proc, ds, device, step, w, tag,
                   n_i2t, micro, sampler, steps_i2t, gs_i2t,
                   gamma=0.0, s_noise=1.003, win="full"):
        """Sample i2t on `ds` and log under mnist_sum/<tag>/. Returns the metrics."""
        layout = ds.layout
        idx = torch.arange(min(n_i2t, len(ds)))
        refs = torch.stack([ds[int(i)] for i in idx]).float().to(device)
        digits, sums = ds.labels(idx.numpy())

        texts: List[str] = []
        for s0 in range(0, refs.shape[0], micro):
            chunk = refs[s0:s0 + micro]
            bits = ms_sample(cfg, model, proc, chunk, layout=layout, task="i2t",
                             sampler_name=sampler, num_steps=steps_i2t,
                             guidance_scale=gs_i2t, device=device,
                             gamma=gamma, s_noise=s_noise, churn_window_mode=win,
                             sigma_decode=float(getattr(c, "sigma_decode", 0.1)))
            texts.extend(ds.reconstruct_batch_from_bits(bits)["text"])

        m = score_i2t(texts, digits[:len(texts)], sums[:len(texts)])
        base = ds.modal_sum_baseline
        if _rank0():
            print(f"[mnist-sum-eval step {step}] {tag} "
                  f"gs={gs_i2t:g} gamma={gamma:g} n={int(m.get('n',0))} "
                  f"perception={m.get('perception',0):.3f} "
                  f"arithmetic={m.get('arithmetic',0):.3f} "
                  f"end_to_end={m.get('end_to_end',0):.3f} "
                  f"(modal-sum baseline {base:.3f}) "
                  f"malformed={m.get('malformed',0):.3f}", flush=True)
            if w is not None:
                for k, v in m.items():
                    if k != "n":
                        w.add_scalar(f"mnist_sum/{tag}/{k}", float(v), step)
                w.add_scalar(f"mnist_sum/{tag}/modal_sum_baseline", base, step)
                rows = []
                for i, t in enumerate(texts[:16]):
                    true_eq = " ".join(C.equation_words(digits[i].tolist()))
                    parsed = parse_equation(t)
                    ok = parsed is not None and parsed[1] == int(sums[i])
                    rows.append(f"| {'ok' if ok else '  '} | `{t}` | `{true_eq}` |")
                w.add_text(f"mnist_sum/{tag}/samples",
                           "\n".join(["| | predicted | true |", "|---|---|---|"] + rows),
                           step)
        return m

    # ---- main -------------------------------------------------------------
    @torch.no_grad()
    def run(self, trainer, step: int):
        cfg = trainer.cfg
        c = self._c(cfg)
        device = trainer.device
        ds = trainer.val_loader.dataset
        layout = ds.layout
        model = trainer.model   # EMA already applied by the caller
        was_training = model.training
        model.eval()

        n_i2t = int(getattr(c, "n_i2t", 256))
        n_t2i = int(getattr(c, "n_t2i", 64))
        steps_i2t = int(getattr(c, "steps_i2t", 256))
        steps_t2i = int(getattr(c, "steps_t2i", 256))
        gs_list_i2t = [float(x) for x in getattr(c, "guidance_scales_i2t", [2.0, 1.0])]
        gs_list_t2i = [float(x) for x in getattr(c, "guidance_scales_t2i", [2.0, 1.0])]
        gamma_i2t = float(getattr(c, "gamma_i2t", 0.0))
        gamma_t2i = float(getattr(c, "gamma_t2i", 0.0))
        s_noise = float(getattr(c, "s_noise", 1.003))
        win = str(getattr(c, "churn_window_mode", "full"))
        micro = int(getattr(c, "micro_batch", 64))
        sampler = str(getattr(c, "sampler", "ddim_entropic"))
        w = _writer(trainer)

        n = max(n_i2t, n_t2i)
        idx = torch.arange(min(n, len(ds)))
        refs = torch.stack([ds[int(i)] for i in idx]).float().to(device)
        digits, sums = ds.labels(idx.numpy())

        proc = getattr(trainer, "proc", None)

        # ---------- image -> text : the exact metric ----------
        # Scored at every guidance setting, so the CFG effect on EXACT accuracy
        # is a measured curve. The first entry keeps the bare tag.
        hds = self._holdout(trainer, ds)
        for gi, gs in enumerate(gs_list_i2t):
            sfx = _gs_suffix(gi, gs)
            kw = dict(cfg=cfg, c=c, model=model, proc=proc, device=device, step=step,
                      w=w, n_i2t=n_i2t, micro=micro, sampler=sampler,
                      steps_i2t=steps_i2t, gs_i2t=gs, gamma=gamma_i2t,
                      s_noise=s_noise, win=win)
            m = self._score_i2t(ds=ds, tag=f"i2t{sfx}", **kw)
            if _rank0() and w is not None:
                w.add_scalar(f"mnist_sum/i2t{sfx}/guidance_scale", gs, step)
                w.add_scalar(f"mnist_sum/i2t{sfx}/gamma", gamma_i2t, step)

            # Same scorer on tuples never seen in training: in-distribution
            # accuracy alone cannot separate learning the mapping from
            # memorising 10^4 of it.
            if hds is not None:
                mh = self._score_i2t(ds=hds, tag=f"i2t_holdout{sfx}", **kw)
                if _rank0() and w is not None:
                    for k in ("perception", "arithmetic", "end_to_end"):
                        if k in m and k in mh:
                            w.add_scalar(f"mnist_sum/gap{sfx}/{k}",
                                         float(m[k]) - float(mh[k]), step)

        # ---------- text -> image : visual for now ----------
        if n_t2i > 0:
            for gi, gs in enumerate(gs_list_t2i):
                sfx = _gs_suffix(gi, gs)
                imgs = []
                for s0 in range(0, min(n_t2i, refs.shape[0]), micro):
                    chunk = refs[s0:s0 + micro]
                    bits = ms_sample(cfg, model, proc, chunk, layout=layout, task="t2i",
                                     sampler_name=sampler, num_steps=steps_t2i,
                                     guidance_scale=gs, device=device,
                                     gamma=gamma_t2i, s_noise=s_noise,
                                     churn_window_mode=win,
                                     sigma_decode=float(getattr(c, "sigma_decode", 0.1)))
                    imgs.append(ds.reconstruct_batch_from_bits(bits)["image"].cpu())
                if imgs and _rank0() and w is not None:
                    from torchvision.utils import make_grid
                    grid = make_grid(torch.cat(imgs, 0).clamp(0, 1), nrow=8)
                    w.add_image(f"mnist_sum/t2i{sfx}/samples", grid, step)
                    w.add_scalar(f"mnist_sum/t2i{sfx}/guidance_scale", gs, step)
                    prompts = [" ".join(C.equation_words(digits[i].tolist()))
                               for i in range(min(8, len(digits)))]
                    w.add_text(f"mnist_sum/t2i{sfx}/prompts",
                               "\n".join(f"- `{p}`" for p in prompts), step)

        if was_training:
            model.train()
