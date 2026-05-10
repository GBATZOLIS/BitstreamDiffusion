#evaluation/generation_driver.py
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world() -> Tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _split_counts(N: int, world_size: int) -> List[int]:
    base = N // world_size
    rem = N % world_size
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def _stable_int_hash(s: str) -> int:
    """Stable across processes/runs (unlike Python's hash())."""
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_sum_scalar(value: float, device: torch.device) -> float:
    if not _ddp_is_on():
        return float(value)
    rank, _ = _rank_world()
    print(f"[rank {rank}] entering all_reduce(sum) value={value}", flush=True)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] leaving all_reduce(sum) -> {float(t.item())}", flush=True)
    return float(t.item())


def _reduce_max_scalar(value: float, device: torch.device) -> float:
    if not _ddp_is_on():
        return float(value)
    rank, _ = _rank_world()
    print(f"[rank {rank}] entering all_reduce(max) value={value}", flush=True)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    print(f"[rank {rank}] leaving all_reduce(max) -> {float(t.item())}", flush=True)
    return float(t.item())


def _resolve_autocast_dtype(cfg: Any, amp_dtype: str, device: torch.device) -> torch.dtype:
    mode = str(amp_dtype).lower()

    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16", "half"}:
        return torch.float16

    if mode == "auto":
        preferred = getattr(getattr(cfg, "evaluation", object()), "amp_dtype", None)
        if preferred is None:
            preferred = getattr(getattr(cfg, "train", object()), "amp_dtype", None)
        preferred = "" if preferred is None else str(preferred).lower()

        bf16_supported = bool(
            device.type == "cuda"
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )

        if preferred in {"bf16", "bfloat16"} and bf16_supported:
            return torch.bfloat16
        return torch.float16 if device.type == "cuda" else torch.float32

    raise ValueError(f"Unknown amp_dtype='{amp_dtype}'")


def _build_speed_stats(
    *,
    device: torch.device,
    wall_time_local_sec: float,
    num_samples_local: int,
    seq_len_tokens: int,
    seq_len_model: int,
    prompt_len_tokens: torch.Tensor,
    prompt_len_model: torch.Tensor,
) -> Dict[str, Any]:
    prompt_len_tokens_sum_local = float(prompt_len_tokens.to(torch.float64).sum().item())
    prompt_len_model_sum_local = float(prompt_len_model.to(torch.float64).sum().item())

    total_samples = int(round(_reduce_sum_scalar(num_samples_local, device)))
    full_lm_tokens = int(round(_reduce_sum_scalar(num_samples_local * seq_len_tokens, device)))
    generated_lm_tokens = int(
        round(
            _reduce_sum_scalar(
                num_samples_local * seq_len_tokens - prompt_len_tokens_sum_local,
                device,
            )
        )
    )

    full_model_positions = int(round(_reduce_sum_scalar(num_samples_local * seq_len_model, device)))
    generated_model_positions = int(
        round(
            _reduce_sum_scalar(
                num_samples_local * seq_len_model - prompt_len_model_sum_local,
                device,
            )
        )
    )

    wall_time_sec = _reduce_max_scalar(wall_time_local_sec, device)
    denom = max(wall_time_sec, 1e-12)
    _, world = _rank_world()

    return dict(
        generation_wall_time_sec=float(wall_time_sec),
        world_size=int(world),
        num_samples=int(total_samples),
        sequence_len_tokens=int(seq_len_tokens),
        sequence_len_model=int(seq_len_model),
        full_lm_tokens=int(full_lm_tokens),
        generated_lm_tokens=int(generated_lm_tokens),
        full_model_positions=int(full_model_positions),
        generated_model_positions=int(generated_model_positions),
        samples_per_sec=float(total_samples / denom),
        full_lm_tokens_per_sec=float(full_lm_tokens / denom),
        generated_lm_tokens_per_sec=float(generated_lm_tokens / denom),
        lm_tokens_per_sec=float(generated_lm_tokens / denom),
        full_model_positions_per_sec=float(full_model_positions / denom),
        generated_model_positions_per_sec=float(generated_model_positions / denom),
        model_positions_per_sec=float(generated_model_positions / denom),
    )


@dataclass
class SamplerBundle:
    sampler: Any
    schedule: Optional[str] = None
    is_discrete: bool = False


def _framework(cfg: Any) -> str:
    return str(getattr(cfg, "framework", "continuous_score")).lower()


def _get_representation(cfg: Any) -> str:
    d = getattr(cfg, "data", None)
    if d is None:
        return "binary"
    return str(getattr(d, "representation", "binary")).lower()


def _get_sequence_len_tokens(cfg: Any) -> int:
    d = getattr(cfg, "data", None)
    if d is None:
        return int(getattr(cfg, "sequence_len_tokens", getattr(cfg, "sequence_len", 0)))
    return int(
        getattr(
            d,
            "sequence_len_tokens",
            getattr(d, "sequence_len_chars", getattr(d, "sequence_len", 0)),
        )
    )


def _get_bits_per_token(cfg: Any) -> int:
    d = getattr(cfg, "data", None)
    return int(getattr(d, "bits_per_token", 1)) if d is not None else 1


def _get_bits_per_token_model(cfg: Any) -> int:
    """
    Bits-per-token in model bitstream space.
    - If ECC enabled: chunk length (data+parity(+p0))
    - Else: cfg.data.bits_per_token
    """
    ecc = ecc_from_cfg(cfg)
    if ecc is not None and bool(getattr(ecc, "enabled", False)):
        return int(ecc_chunk_len(ecc))
    return _get_bits_per_token(cfg)


def _get_discrete_sequence_len(cfg: Any) -> int:
    d = getattr(cfg, "data", None)
    if d is None:
        return 0
    repr_mode = str(getattr(d, "representation", "tokens")).lower()
    if repr_mode == "binary":
        return int(getattr(d, "sequence_len", 0))
    return int(
        getattr(
            d,
            "sequence_len_tokens",
            getattr(d, "sequence_len_chars", getattr(d, "sequence_len", 0)),
        )
    )


def _discrete_positions_per_token(cfg: Any) -> int:
    d = getattr(cfg, "data", None)
    repr_mode = str(getattr(d, "representation", "tokens")).lower()
    return _get_bits_per_token_model(cfg) if repr_mode == "binary" else 1


def _discrete_is_bitstream(cfg: Any) -> bool:
    d = getattr(cfg, "data", None)
    return str(getattr(d, "representation", "tokens")).lower() == "binary"


def _default_ati_eta(cfg: Any) -> float:
    ev = getattr(cfg, "evaluation", None)
    if ev is None:
        return 0.0

    ati_cfg = getattr(ev, "ati", None)
    if ati_cfg is not None:
        if not bool(getattr(ati_cfg, "enabled", True)):
            return 0.0
        return max(0.0, float(getattr(ati_cfg, "eta", 0.0)))

    return max(0.0, float(getattr(ev, "ati_eta", 0.0)))


def create_sampler(
    cfg: Any,
    model: Any,
    proc: Any,
    sampler_name: str,
    *,
    device: torch.device,
    num_steps: Optional[int] = None,
) -> SamplerBundle:
    """
    Factory that instantiates either a Continuous or Discrete sampler based on cfg.framework.
    """
    name = str(sampler_name).lower()
    fw = _framework(cfg)

    # --- Discrete branch ---
    if fw.startswith("discrete"):
        from diffusion.discrete.samplers import EulerRateSampler, TweedieTauLeapingSampler

        vocab_size = int(
            getattr(proc, "vocab_size", getattr(getattr(cfg, "data", None), "vocab_size", 0))
        )
        if vocab_size <= 0:
            raise ValueError(
                "Discrete sampler: vocab_size not found (proc.vocab_size or cfg.data.vocab_size)."
            )

        is_absorb = bool(getattr(proc, "is_absorb", True))
        mask_id = getattr(proc, "mask_id", None)
        if mask_id is None:
            g = getattr(proc, "graph", None)
            mask_id = getattr(g, "mask_id", None) if g is not None else None

        seq_len = _get_discrete_sequence_len(cfg)
        gen_cfg = getattr(getattr(cfg, "train", None), "generation", None)
        num_steps = int(
            num_steps
            if num_steps is not None
            else (getattr(gen_cfg, "num_sampling_steps", 128) if gen_cfg is not None else 128)
        )
        t_eps = float(getattr(getattr(cfg.diffusion, "discrete", object()), "eps", 1e-3))

        if name in {"euler", "euler_rate"}:
            return SamplerBundle(
                sampler=EulerRateSampler(
                    model=model,
                    process=proc,
                    device=device,
                    vocab_size=vocab_size,
                    is_absorb=is_absorb,
                    mask_id=mask_id,
                    seq_len=seq_len,
                    num_steps=num_steps,
                    t_eps=t_eps,
                ),
                is_discrete=True,
            )

        if name in {"tweedie", "tweedie_tau", "tweedie_tau_leaping"}:
            return SamplerBundle(
                sampler=TweedieTauLeapingSampler(
                    model=model,
                    process=proc,
                    device=device,
                    vocab_size=vocab_size,
                    is_absorb=is_absorb,
                    mask_id=mask_id,
                    seq_len=seq_len,
                    num_steps=num_steps,
                    t_eps=t_eps,
                ),
                is_discrete=True,
            )

        raise ValueError(f"Unknown discrete sampler '{sampler_name}' for framework '{fw}'")

    # --- Continuous branch ---
    if name in {"heun", "heun_karras", "karras"}:
        from diffusion.continuous.samplers import HeunSampler

        return SamplerBundle(
            sampler=HeunSampler(model, proc, cfg),
            schedule="karras",
            is_discrete=False,
        )

    if name in {"ddim", "ddim_entropic", "entropic"}:
        from diffusion.continuous.samplers import DDIMSampler

        return SamplerBundle(
            sampler=DDIMSampler(model, proc, cfg),
            schedule="entropic",
            is_discrete=False,
        )

    if name in {"ddim_karras"}:
        from diffusion.continuous.samplers import DDIMSampler

        return SamplerBundle(
            sampler=DDIMSampler(model, proc, cfg),
            schedule="karras",
            is_discrete=False,
        )

    raise ValueError(f"Unknown sampler '{sampler_name}' for framework '{fw}'")


@torch.no_grad()
def gather_varlen_firstdim_to_rank0(x: torch.Tensor, dst: int = 0) -> Optional[torch.Tensor]:
    if not _ddp_is_on():
        return x

    rank, world = _rank_world()

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"[rank {rank}] gather_varlen_firstdim_to_rank0 expected Tensor, got {type(x)}")

    if x.dim() < 1:
        raise ValueError(f"[rank {rank}] gather_varlen_firstdim_to_rank0 expected dim>=1, got shape={tuple(x.shape)}")

    print(
        f"[rank {rank}] gather_varlen_firstdim_to_rank0 ENTER "
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} dst={dst}",
        flush=True,
    )

    my = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
    sizes = [torch.zeros_like(my) for _ in range(world)]

    print(f"[rank {rank}] gather sizes: before all_gather", flush=True)
    dist.all_gather(sizes, my)
    print(f"[rank {rank}] gather sizes: after all_gather", flush=True)

    sizes_int = [int(s.item()) for s in sizes]
    mx = max(sizes_int)

    print(
        f"[rank {rank}] gather sizes: local_n={int(x.size(0))} all_sizes={sizes_int} max_n={mx}",
        flush=True,
    )

    x_pad = x.detach()

    if x_pad.size(0) < mx:
        pad = torch.zeros(
            (mx - x_pad.size(0),) + x_pad.shape[1:],
            device=x.device,
            dtype=x.dtype,
        )
        x_pad = torch.cat([x_pad, pad], dim=0)

    print(
        f"[rank {rank}] gather padded tensor shape={tuple(x_pad.shape)} dtype={x_pad.dtype} device={x_pad.device}",
        flush=True,
    )

    gathered = [torch.empty_like(x_pad) for _ in range(world)]

    print(f"[rank {rank}] gather payload: before all_gather", flush=True)
    dist.all_gather(gathered, x_pad)
    print(f"[rank {rank}] gather payload: after all_gather", flush=True)

    if rank != dst:
        print(f"[rank {rank}] gather returning None (non-dst)", flush=True)
        return None

    chunks = [gathered[i][:sizes_int[i]] for i in range(world) if sizes_int[i] > 0]
    out = torch.cat(chunks, dim=0) if chunks else x_pad[:0]

    print(
        f"[rank {rank}] gather returning tensor shape={tuple(out.shape)} dtype={out.dtype} device={out.device}",
        flush=True,
    )
    return out


@dataclass
class GenerationBatch:
    gen_bits: Optional[torch.Tensor] = None
    ref_bits: Optional[torch.Tensor] = None
    prompt_len_bits: Optional[torch.Tensor] = None
    gen_tokens: Optional[torch.Tensor] = None
    ref_tokens: Optional[torch.Tensor] = None
    prompt_len_tokens: Optional[torch.Tensor] = None
    stats: Optional[Dict[str, Any]] = None


class GenerationDriver:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def _sample_prompt_len_tokens(
        self,
        B: int,
        S_tok: int,
        device: torch.device,
        seed: int,
    ) -> torch.Tensor:
        cond = getattr(self.cfg, "cond", None)
        if not cond or not bool(getattr(cond, "enabled", False)):
            return torch.zeros(B, device=device, dtype=torch.long)

        if bool(getattr(cond, "sample_prompt_len", False)):
            mn = int(getattr(cond, "cond_len_tokens_min", getattr(cond, "cond_len_chars_min", 0)))
            mx = int(getattr(cond, "cond_len_tokens_max", getattr(cond, "cond_len_chars_max", 128)))
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            toks = torch.randint(mn, mx + 1, (B,), generator=g, device=device)
            return torch.clamp(toks, 0, S_tok)

        toks = int(getattr(cond, "cond_len_tokens", getattr(cond, "cond_len_chars", 64)))
        return torch.full((B,), int(toks), device=device, dtype=torch.long)

    def _iter_take_ref(
        self,
        loader: DataLoader,
        n_local: int,
        device: torch.device,
        *,
        discrete: bool,
        cont_tokens: bool = False,
    ) -> torch.Tensor:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(0)

        chunks = []
        remaining = n_local

        for batch in loader:
            if remaining <= 0:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if batch.dim() == 3:
                batch = batch.squeeze(1)
            take = min(remaining, batch.size(0))
            chunks.append(batch[:take].to(device))
            remaining -= take

        if n_local == 0 or not chunks:
            if discrete:
                S = _get_discrete_sequence_len(self.cfg)
                dtype = torch.long
            elif cont_tokens:
                S = _get_sequence_len_tokens(self.cfg)
                dtype = torch.long
            else:
                S = int(getattr(getattr(self.cfg, "data", None), "sequence_len", 0))
                dtype = torch.float32
            return torch.zeros((max(0, n_local), S), device=device, dtype=dtype)

        ref = torch.cat(chunks, dim=0)
        if discrete or cont_tokens:
            return ref.to(dtype=torch.long)

        if ref.dtype not in {torch.float32, torch.float64}:
            ref = ref.float()
        return ref

    def _resolve_safe_batch_size(self, explicit_size: Optional[int]) -> int:
        if explicit_size is not None and int(explicit_size) > 0:
            return int(explicit_size)

        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is not None:
            gen_cfg = getattr(train_cfg, "generation", None)
            if gen_cfg is not None:
                mbs = getattr(gen_cfg, "micro_batch_size", None)
                if mbs is not None and int(mbs) > 0:
                    return int(mbs)

        return 128

    @torch.no_grad()
    def generate_prompt_completion(
        self,
        *,
        model: Any,
        proc: Any,
        device: torch.device,
        loader: DataLoader,
        num_samples: int,
        sampler_names: List[str],
        terminal_sigmas: List[float],
        guidance_scales: List[float],
        num_steps: int,
        sampling_specs: Optional[List[Dict[str, Any]]] = None,
        entropic_blend_alpha: float = 0.0,
        entropy_run_dir: Optional[str] = None,
        seed: int = 42,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        micro_batch_size: Optional[int] = None,
        sigma_max: Optional[float] = None,
        gather_to_rank0: bool = True,
    ) -> Dict[str, GenerationBatch]:
        fw = _framework(self.cfg)
        is_discrete = fw.startswith("discrete")
        repr_mode = _get_representation(self.cfg)
        is_cont_tokens = (not is_discrete) and (repr_mode == "tokens")
        is_cont_bits = (not is_discrete) and (repr_mode == "binary")

        rank, world = _rank_world()
        counts = _split_counts(int(num_samples), world)
        n_local = counts[rank]
        global_offset = int(sum(counts[:rank]))
        MOD = 2**63 - 1

        devices_for_fork = [int(device.index)] if (device.type == "cuda" and device.index is not None) else []

        if micro_batch_size is not None and int(micro_batch_size) == 0:
            n_local = 0
            local_bs = 1
        else:
            local_bs = self._resolve_safe_batch_size(micro_batch_size)

        local_bs = max(1, min(int(local_bs), int(n_local) if n_local > 0 else 1))

        ref_local = self._iter_take_ref(
            loader,
            n_local,
            device=device,
            discrete=is_discrete,
            cont_tokens=is_cont_tokens,
        )

        if is_discrete:
            S_model = ref_local.size(1) if ref_local.numel() else _get_discrete_sequence_len(self.cfg)
            pos_per_tok = _discrete_positions_per_token(self.cfg)
            S_tok = max(1, S_model // max(1, pos_per_tok))

            prompt_len_tokens = self._sample_prompt_len_tokens(
                n_local,
                S_tok,
                device=device,
                seed=int(seed + 999 * rank),
            )
            prompt_len_model = torch.clamp(prompt_len_tokens * pos_per_tok, 0, S_model)

            ar = torch.arange(S_model, device=device).view(1, S_model)
            prefix_mask_full = ar < prompt_len_model.view(-1, 1)

        elif is_cont_tokens:
            S_tok = ref_local.size(1) if ref_local.numel() else _get_sequence_len_tokens(self.cfg)
            S_model = S_tok

            prompt_len_tokens = self._sample_prompt_len_tokens(
                n_local,
                S_tok,
                device=device,
                seed=int(seed + 999 * rank),
            )
            ar = torch.arange(S_tok, device=device).view(1, S_tok)
            prefix_mask_full = ar < prompt_len_tokens.view(-1, 1)

        else:
            S_bits = (
                ref_local.size(1)
                if ref_local.numel()
                else int(getattr(getattr(self.cfg, "data", None), "sequence_len", 0))
            )
            bpt_model = _get_bits_per_token_model(self.cfg)
            S_tok = _get_sequence_len_tokens(self.cfg) if bpt_model > 1 else S_bits

            prompt_len_tokens = self._sample_prompt_len_tokens(
                n_local,
                S_tok,
                device=device,
                seed=int(seed + 999 * rank),
            )
            prompt_len_bits = torch.clamp(prompt_len_tokens * bpt_model, 0, S_bits)
            ar = torch.arange(S_bits, device=device).view(1, S_bits)
            prefix_mask_full = ar < prompt_len_bits.view(-1, 1)

        amp_enabled = bool(use_amp and device.type == "cuda")
        dtype = _resolve_autocast_dtype(self.cfg, amp_dtype, device)
        results: Dict[str, GenerationBatch] = {}

        spec_list = sampling_specs
        if spec_list is None:
            spec_list = []
            default_ati_eta = _default_ati_eta(self.cfg)

            for sampler_name in sampler_names:
                for sigma in terminal_sigmas:
                    for gs in guidance_scales:
                        tag = f"{sampler_name}_term{math.log10(float(sigma)):.2f}_gs{float(gs):.1f}"
                        if default_ati_eta > 0.0:
                            tag += f"_ati{default_ati_eta:.2f}"

                        spec_list.append(
                            dict(
                                tag=tag,
                                sampler_name=str(sampler_name),
                                terminal_sigma=float(sigma),
                                guidance_scale=float(gs),
                                num_steps=int(num_steps),
                                sc_refresh_mode="refined",
                                ati_eta=float(default_ati_eta),
                            )
                        )

        for spec in spec_list:
            sampler_name = str(spec["sampler_name"])
            sigma = float(spec["terminal_sigma"])
            gs = float(spec["guidance_scale"])
            spec_steps = int(spec["num_steps"])
            sc_refresh_mode = str(spec.get("sc_refresh_mode", "refined"))
            ati_eta = float(spec.get("ati_eta", _default_ati_eta(self.cfg)))
            tag = str(spec["tag"])

            if tag in results:
                raise ValueError(
                    f"Duplicate sampling spec tag '{tag}'. "
                    "Sampling spec tags must be unique. "
                    "Include stochastic settings in the tag."
                )

            # ------------------------------------------------------------
            # Apply per-spec stochastic settings robustly.
            #
            # Current continuous samplers read stochastic controls from
            # cfg.evaluation.stochastic. Therefore every resolved sampling spec
            # must be copied into that block before sampler construction/call.
            #
            # Important:
            #   - Create cfg.evaluation.stochastic if the config forgot to define it.
            #   - Always set deterministic defaults for missing fields, otherwise a
            #     previous stochastic spec can leak into a later deterministic spec.
            #   - Restore the original config values in the finally block below.
            # ------------------------------------------------------------
            st_backup = {}
            st_block_existed = False

            ev = getattr(self.cfg, "evaluation", None)
            if ev is None:
                raise ValueError("cfg.evaluation is missing; cannot apply sampling spec.")

            st_block = getattr(ev, "stochastic", None)
            st_block_existed = st_block is not None

            if st_block is None:
                from ml_collections import config_dict

                ev.stochastic = config_dict.ConfigDict()
                st_block = ev.stochastic

            st_keys_map = {
                "stochastic_enabled": "enabled",
                "s_churn": "s_churn",
                "s_noise": "s_noise",
                "window_mode": "window_mode",
                "entropy_quantile_lo": "entropy_quantile_lo",
                "entropy_quantile_hi": "entropy_quantile_hi",
                "s_tmin": "s_tmin",
                "s_tmax": "s_tmax",
                "entropy_fallback": "entropy_fallback",
            }

            st_defaults = {
                "stochastic_enabled": False,
                "s_churn": 0.0,
                "s_noise": 1.0,
                "window_mode": "entropy_cdf",
                "entropy_quantile_lo": 0.10,
                "entropy_quantile_hi": 0.90,
                "s_tmin": None,
                "s_tmax": None,
                "entropy_fallback": "deterministic",
            }

            for spec_k, cfg_k in st_keys_map.items():
                old_exists = hasattr(st_block, cfg_k)
                old_val = getattr(st_block, cfg_k, None)
                st_backup[cfg_k] = (old_exists, old_val)

                new_val = spec.get(spec_k, st_defaults[spec_k])
                setattr(st_block, cfg_k, new_val)

            if bool(spec.get("stochastic_enabled", False)):
                if not bool(getattr(st_block, "enabled", False)):
                    raise RuntimeError(
                        f"Stochastic spec requested but cfg.evaluation.stochastic.enabled=False "
                        f"after applying spec. tag={tag}"
                    )

                if float(getattr(st_block, "s_churn", 0.0)) <= 0.0:
                    raise RuntimeError(
                        f"Stochastic spec requested but s_churn<=0 after applying spec. "
                        f"tag={tag}, s_churn={getattr(st_block, 's_churn', None)}"
                    )

                if rank == 0:
                    print(
                        f"[generation_driver][stochastic] tag={tag} "
                        f"enabled={getattr(st_block, 'enabled', None)} "
                        f"s_churn={getattr(st_block, 's_churn', None)} "
                        f"s_noise={getattr(st_block, 's_noise', None)} "
                        f"window_mode={getattr(st_block, 'window_mode', None)} "
                        f"qlo={getattr(st_block, 'entropy_quantile_lo', None)} "
                        f"qhi={getattr(st_block, 'entropy_quantile_hi', None)} "
                        f"s_tmin={getattr(st_block, 's_tmin', None)} "
                        f"s_tmax={getattr(st_block, 's_tmax', None)} "
                        f"fallback={getattr(st_block, 'entropy_fallback', None)}",
                        flush=True,
                    )

            tag_hash = _stable_int_hash(tag)

            try:
                bundle = create_sampler(
                    self.cfg,
                    model,
                    proc,
                    sampler_name,
                    device=device,
                    num_steps=spec_steps,
                )

                _sync_device(device)
                t_gen_start = time.perf_counter()

                # ------------------------------------------------------------
                # Discrete branch
                # ------------------------------------------------------------
                if bundle.is_discrete:
                    if n_local > 0:
                        xt_chunks = []
                        for i in range(0, n_local, local_bs):
                            end = min(i + local_bs, n_local)
                            global_i = global_offset + i
                            chunk_seed = (int(seed) * 1_000_003 + tag_hash * 10_007 + int(global_i)) % MOD

                            with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                torch.manual_seed(int(chunk_seed))
                                if device.type == "cuda":
                                    torch.cuda.manual_seed_all(int(chunk_seed))

                                with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                    xt_chunk = bundle.sampler.sample(
                                        num_samples=end - i,
                                        conditioning_prefix=ref_local[i:end],
                                        cond_mask=prefix_mask_full[i:end],
                                    )
                            xt_chunks.append(xt_chunk)

                        xt = (
                            torch.cat(xt_chunks, dim=0)
                            if xt_chunks
                            else torch.empty((0, _get_discrete_sequence_len(self.cfg)), device=device, dtype=torch.long)
                        )
                    else:
                        xt = torch.empty((0, _get_discrete_sequence_len(self.cfg)), device=device, dtype=torch.long)

                    _sync_device(device)
                    wall_time_local_sec = time.perf_counter() - t_gen_start

                    print(
                        f"[rank {rank}] finished local generation for tag={tag} "
                        f"xt_shape={tuple(xt.shape)} xt_dtype={xt.dtype} "
                        f"ref_shape={tuple(ref_local.shape)} ref_dtype={ref_local.dtype} "
                        f"prompt_len_tokens_shape={tuple(prompt_len_tokens.shape)} prompt_len_tokens_dtype={prompt_len_tokens.dtype} "
                        f"prompt_len_model_shape={tuple(prompt_len_model.shape)} prompt_len_model_dtype={prompt_len_model.dtype} "
                        f"device={xt.device}",
                        flush=True,
                    )

                    stats = _build_speed_stats(
                        device=device,
                        wall_time_local_sec=wall_time_local_sec,
                        num_samples_local=n_local,
                        seq_len_tokens=S_tok,
                        seq_len_model=S_model,
                        prompt_len_tokens=prompt_len_tokens,
                        prompt_len_model=prompt_len_model,
                    )

                    is_discrete_bitstream = _discrete_is_bitstream(self.cfg)

                    if gather_to_rank0:
                        print(f"[rank {rank}] about to gather DISCRETE xt for tag={tag}", flush=True)
                        xt_g = gather_varlen_firstdim_to_rank0(xt, dst=0)

                        print(f"[rank {rank}] about to gather DISCRETE ref_local for tag={tag}", flush=True)
                        ref_g = gather_varlen_firstdim_to_rank0(ref_local, dst=0)

                        print(f"[rank {rank}] about to gather DISCRETE prompt_len_tokens for tag={tag}", flush=True)
                        len_tok_g = gather_varlen_firstdim_to_rank0(prompt_len_tokens.to(torch.long), dst=0)

                        print(f"[rank {rank}] about to gather DISCRETE prompt_len_model for tag={tag}", flush=True)
                        len_model_g = gather_varlen_firstdim_to_rank0(prompt_len_model.to(torch.long), dst=0)

                        print(f"[rank {rank}] finished all DISCRETE gathers for tag={tag}", flush=True)

                        if rank == 0:
                            if is_discrete_bitstream:
                                results[tag] = GenerationBatch(
                                    gen_bits=xt_g.cpu().to(torch.uint8)[:num_samples] if xt_g is not None else None,
                                    ref_bits=ref_g.cpu().to(torch.uint8)[:num_samples] if ref_g is not None else None,
                                    prompt_len_bits=len_model_g.cpu()[:num_samples] if len_model_g is not None else None,
                                    stats=stats,
                                )
                            else:
                                results[tag] = GenerationBatch(
                                    gen_tokens=xt_g.cpu()[:num_samples] if xt_g is not None else None,
                                    ref_tokens=ref_g.cpu()[:num_samples] if ref_g is not None else None,
                                    prompt_len_tokens=len_tok_g.cpu()[:num_samples] if len_tok_g is not None else None,
                                    stats=stats,
                                )
                    else:
                        if is_discrete_bitstream:
                            results[tag] = GenerationBatch(
                                gen_bits=xt.cpu().to(torch.uint8),
                                ref_bits=ref_local.cpu().to(torch.uint8),
                                prompt_len_bits=prompt_len_model.to(torch.long).cpu(),
                                stats=stats,
                            )
                        else:
                            results[tag] = GenerationBatch(
                                gen_tokens=xt.cpu(),
                                ref_tokens=ref_local.cpu(),
                                prompt_len_tokens=prompt_len_tokens.to(torch.long).cpu(),
                                stats=stats,
                            )

                    continue

                # ------------------------------------------------------------
                # Continuous token branch
                # ------------------------------------------------------------
                if is_cont_tokens:
                    sampler, schedule = bundle.sampler, bundle.schedule

                    if n_local > 0:
                        gen_token_chunks = []

                        for i in range(0, n_local, local_bs):
                            end = min(i + local_bs, n_local)
                            show_prog = (rank == 0 and i == 0)

                            global_i = global_offset + i
                            chunk_seed = (int(seed) * 1_000_003 + tag_hash * 10_007 + int(global_i)) % MOD

                            try:
                                with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                    torch.manual_seed(int(chunk_seed))
                                    if device.type == "cuda":
                                        torch.cuda.manual_seed_all(int(chunk_seed))

                                    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                        _, probs_chunk = sampler.sample(
                                            end - i,
                                            int(ref_local.size(1)),
                                            schedule=schedule,
                                            num_steps=int(spec_steps),
                                            sigma_min_override=float(sigma),
                                            sigma_max_override=sigma_max,
                                            entropic_blend_alpha=float(entropic_blend_alpha),
                                            entropy_run_dir=entropy_run_dir,
                                            conditioning_prefix_full=ref_local[i:end],
                                            cond_prefix_mask=prefix_mask_full[i:end],
                                            guidance_scale=float(gs),
                                            sc_refresh_mode=sc_refresh_mode,
                                            ati_eta=float(ati_eta),
                                            return_probs=True,
                                            progress=show_prog,
                                        )

                            except TypeError as e:
                                if "sigma_max_override" in str(e):
                                    with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                        torch.manual_seed(int(chunk_seed))
                                        if device.type == "cuda":
                                            torch.cuda.manual_seed_all(int(chunk_seed))

                                        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                            _, probs_chunk = sampler.sample(
                                                end - i,
                                                int(ref_local.size(1)),
                                                schedule=schedule,
                                                num_steps=int(spec_steps),
                                                sigma_min_override=float(sigma),
                                                entropic_blend_alpha=float(entropic_blend_alpha),
                                                entropy_run_dir=entropy_run_dir,
                                                conditioning_prefix_full=ref_local[i:end],
                                                cond_prefix_mask=prefix_mask_full[i:end],
                                                guidance_scale=float(gs),
                                                sc_refresh_mode=sc_refresh_mode,
                                                ati_eta=float(ati_eta),
                                                return_probs=True,
                                                progress=show_prog,
                                            )
                                else:
                                    raise

                            gen_chunk = probs_chunk.argmax(dim=-1).long()
                            del probs_chunk

                            if prefix_mask_full.numel() > 0:
                                mask_chunk = prefix_mask_full[i:end]
                                gen_chunk[mask_chunk] = ref_local[i:end][mask_chunk]

                            gen_token_chunks.append(gen_chunk)

                        gen_tokens = torch.cat(gen_token_chunks, dim=0)
                        ref_tokens = ref_local

                    else:
                        S_tok = int(
                            getattr(getattr(self.cfg, "data", None), "sequence_len_tokens", 0)
                            or getattr(getattr(self.cfg, "data", None), "sequence_len", 0)
                        )
                        S_model = S_tok
                        gen_tokens = torch.empty((0, S_tok), device=device, dtype=torch.long)
                        ref_tokens = torch.empty((0, S_tok), device=device, dtype=torch.long)

                    _sync_device(device)
                    wall_time_local_sec = time.perf_counter() - t_gen_start

                    print(
                        f"[rank {rank}] finished local generation for tag={tag} "
                        f"gen_tokens_shape={tuple(gen_tokens.shape)} gen_tokens_dtype={gen_tokens.dtype} "
                        f"ref_tokens_shape={tuple(ref_tokens.shape)} ref_tokens_dtype={ref_tokens.dtype} "
                        f"prompt_len_tokens_shape={tuple(prompt_len_tokens.shape)} prompt_len_tokens_dtype={prompt_len_tokens.dtype} "
                        f"device={gen_tokens.device}",
                        flush=True,
                    )

                    stats = _build_speed_stats(
                        device=device,
                        wall_time_local_sec=wall_time_local_sec,
                        num_samples_local=n_local,
                        seq_len_tokens=S_tok,
                        seq_len_model=S_model,
                        prompt_len_tokens=prompt_len_tokens,
                        prompt_len_model=prompt_len_tokens,
                    )

                    if gather_to_rank0:
                        print(f"[rank {rank}] about to gather CONT-TOK gen_tokens for tag={tag}", flush=True)
                        gen_g = gather_varlen_firstdim_to_rank0(gen_tokens, dst=0)

                        print(f"[rank {rank}] about to gather CONT-TOK ref_tokens for tag={tag}", flush=True)
                        ref_g = gather_varlen_firstdim_to_rank0(ref_tokens, dst=0)

                        print(f"[rank {rank}] about to gather CONT-TOK prompt_len_tokens for tag={tag}", flush=True)
                        len_g = gather_varlen_firstdim_to_rank0(prompt_len_tokens.to(torch.long), dst=0)

                        print(f"[rank {rank}] finished all CONT-TOK gathers for tag={tag}", flush=True)

                        if rank == 0:
                            results[tag] = GenerationBatch(
                                gen_tokens=gen_g.cpu()[:num_samples] if gen_g is not None else None,
                                ref_tokens=ref_g.cpu()[:num_samples] if ref_g is not None else None,
                                prompt_len_tokens=len_g.cpu()[:num_samples] if len_g is not None else None,
                                stats=stats,
                            )
                    else:
                        results[tag] = GenerationBatch(
                            gen_tokens=gen_tokens.cpu(),
                            ref_tokens=ref_tokens.cpu(),
                            prompt_len_tokens=prompt_len_tokens.to(torch.long).cpu(),
                            stats=stats,
                        )

                    continue

                # ------------------------------------------------------------
                # Continuous bit branch
                # ------------------------------------------------------------
                if not is_cont_bits:
                    raise RuntimeError("Unsupported continuous generation mode")

                sampler, schedule = bundle.sampler, bundle.schedule
                ref_u8_local = (ref_local > 0.5).to(torch.uint8) if n_local > 0 else None

                if n_local > 0:
                    gen_bits_chunks = []

                    for i in range(0, n_local, local_bs):
                        end = min(i + local_bs, n_local)
                        show_prog = (rank == 0 and i == 0)

                        global_i = global_offset + i
                        chunk_seed = (int(seed) * 1_000_003 + tag_hash * 10_007 + int(global_i)) % MOD

                        try:
                            with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                torch.manual_seed(int(chunk_seed))
                                if device.type == "cuda":
                                    torch.cuda.manual_seed_all(int(chunk_seed))

                                with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                    _, probs_chunk = sampler.sample(
                                        end - i,
                                        int(ref_local.size(1)),
                                        schedule=schedule,
                                        num_steps=int(spec_steps),
                                        sigma_min_override=float(sigma),
                                        sigma_max_override=sigma_max,
                                        entropic_blend_alpha=float(entropic_blend_alpha),
                                        entropy_run_dir=entropy_run_dir,
                                        conditioning_prefix_full=ref_local[i:end],
                                        cond_prefix_mask=prefix_mask_full[i:end],
                                        guidance_scale=float(gs),
                                        sc_refresh_mode=sc_refresh_mode,
                                        ati_eta=float(ati_eta),
                                        return_probs=True,
                                        progress=show_prog,
                                    )

                        except TypeError as e:
                            if "sigma_max_override" in str(e):
                                with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                    torch.manual_seed(int(chunk_seed))
                                    if device.type == "cuda":
                                        torch.cuda.manual_seed_all(int(chunk_seed))

                                    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                        _, probs_chunk = sampler.sample(
                                            end - i,
                                            int(ref_local.size(1)),
                                            schedule=schedule,
                                            num_steps=int(spec_steps),
                                            sigma_min_override=float(sigma),
                                            entropic_blend_alpha=float(entropic_blend_alpha),
                                            entropy_run_dir=entropy_run_dir,
                                            conditioning_prefix_full=ref_local[i:end],
                                            cond_prefix_mask=prefix_mask_full[i:end],
                                            guidance_scale=float(gs),
                                            sc_refresh_mode=sc_refresh_mode,
                                            ati_eta=float(ati_eta),
                                            return_probs=True,
                                            progress=show_prog,
                                        )
                            else:
                                curr_bs = end - i
                                probs_chunk = torch.zeros(
                                    (curr_bs, int(ref_local.size(1))),
                                    device=device,
                                    dtype=torch.float32,
                                )
                                bpt_model = _get_bits_per_token_model(self.cfg)
                                prompt_len_bits_chunk = torch.clamp(
                                    prompt_len_tokens[i:end] * bpt_model,
                                    0,
                                    int(ref_local.size(1)),
                                )
                                uniq = torch.unique(prompt_len_bits_chunk).tolist()

                                for cL in uniq:
                                    idx = (prompt_len_bits_chunk == int(cL)).nonzero(as_tuple=False).view(-1)
                                    if idx.numel() == 0:
                                        continue

                                    ref_sub = ref_local[i:end].index_select(0, idx)
                                    sub_seed = (int(chunk_seed) + int(cL) * 1009) % MOD

                                    with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                                        torch.manual_seed(int(sub_seed))
                                        if device.type == "cuda":
                                            torch.cuda.manual_seed_all(int(sub_seed))

                                        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                                            try:
                                                _, p_sub = sampler.sample(
                                                    int(idx.numel()),
                                                    int(ref_local.size(1)),
                                                    schedule=schedule,
                                                    num_steps=int(spec_steps),
                                                    sigma_min_override=float(sigma),
                                                    sigma_max_override=sigma_max,
                                                    entropic_blend_alpha=float(entropic_blend_alpha),
                                                    entropy_run_dir=entropy_run_dir,
                                                    conditioning_prefix=ref_sub,
                                                    cond_len_bits=int(cL),
                                                    guidance_scale=float(gs),
                                                    sc_refresh_mode=sc_refresh_mode,
                                                    ati_eta=float(ati_eta),
                                                    return_probs=True,
                                                    progress=show_prog,
                                                )
                                            except TypeError:
                                                _, p_sub = sampler.sample(
                                                    int(idx.numel()),
                                                    int(ref_local.size(1)),
                                                    schedule=schedule,
                                                    num_steps=int(spec_steps),
                                                    sigma_min_override=float(sigma),
                                                    entropic_blend_alpha=float(entropic_blend_alpha),
                                                    entropy_run_dir=entropy_run_dir,
                                                    conditioning_prefix=ref_sub,
                                                    cond_len_bits=int(cL),
                                                    guidance_scale=float(gs),
                                                    sc_refresh_mode=sc_refresh_mode,
                                                    ati_eta=float(ati_eta),
                                                    return_probs=True,
                                                    progress=show_prog,
                                                )

                                    probs_chunk.index_copy_(0, idx, p_sub.to(dtype=torch.float32))

                        gen_chunk = (probs_chunk > 0.5).to(torch.uint8)
                        del probs_chunk

                        if prefix_mask_full.numel() > 0:
                            mask_chunk = prefix_mask_full[i:end]
                            gen_chunk[mask_chunk] = ref_u8_local[i:end][mask_chunk]

                        gen_bits_chunks.append(gen_chunk)

                    gen_bits = torch.cat(gen_bits_chunks, dim=0)
                    ref_u8 = ref_u8_local

                else:
                    S_bits = int(getattr(getattr(self.cfg, "data", None), "sequence_len", 0))
                    gen_bits = torch.empty((0, S_bits), device=device, dtype=torch.uint8)
                    ref_u8 = torch.empty((0, S_bits), device=device, dtype=torch.uint8)

                bpt_model = _get_bits_per_token_model(self.cfg)
                S_bits_for_clamp = (
                    int(ref_local.size(1))
                    if ref_local.numel()
                    else int(getattr(getattr(self.cfg, "data", None), "sequence_len", 0))
                )
                pl_bits_safe = torch.clamp(prompt_len_tokens * bpt_model, 0, S_bits_for_clamp)

                _sync_device(device)
                wall_time_local_sec = time.perf_counter() - t_gen_start

                print(
                    f"[rank {rank}] finished local generation for tag={tag} "
                    f"gen_bits_shape={tuple(gen_bits.shape)} gen_bits_dtype={gen_bits.dtype} "
                    f"ref_u8_shape={tuple(ref_u8.shape)} ref_u8_dtype={ref_u8.dtype} "
                    f"pl_bits_shape={tuple(pl_bits_safe.shape)} pl_bits_dtype={pl_bits_safe.dtype} "
                    f"device={gen_bits.device}",
                    flush=True,
                )

                stats = _build_speed_stats(
                    device=device,
                    wall_time_local_sec=wall_time_local_sec,
                    num_samples_local=n_local,
                    seq_len_tokens=S_tok,
                    seq_len_model=S_bits_for_clamp,
                    prompt_len_tokens=prompt_len_tokens,
                    prompt_len_model=pl_bits_safe,
                )

                if gather_to_rank0:
                    print(f"[rank {rank}] about to gather CONT-BIT gen_bits for tag={tag}", flush=True)
                    gen_g = gather_varlen_firstdim_to_rank0(gen_bits, dst=0)

                    print(f"[rank {rank}] about to gather CONT-BIT ref_u8 for tag={tag}", flush=True)
                    ref_g = gather_varlen_firstdim_to_rank0(ref_u8, dst=0)

                    print(f"[rank {rank}] about to gather CONT-BIT prompt_len_bits for tag={tag}", flush=True)
                    len_g = gather_varlen_firstdim_to_rank0(pl_bits_safe.to(torch.long), dst=0)

                    print(f"[rank {rank}] finished all CONT-BIT gathers for tag={tag}", flush=True)

                    if rank == 0:
                        results[tag] = GenerationBatch(
                            gen_bits=gen_g.cpu()[:num_samples] if gen_g is not None else None,
                            ref_bits=ref_g.cpu()[:num_samples] if ref_g is not None else None,
                            prompt_len_bits=len_g.cpu()[:num_samples] if len_g is not None else None,
                            stats=stats,
                        )
                else:
                    results[tag] = GenerationBatch(
                        gen_bits=gen_bits.cpu(),
                        ref_bits=ref_u8.cpu() if ref_u8 is not None else None,
                        prompt_len_bits=pl_bits_safe.to(torch.long).cpu(),
                        stats=stats,
                    )

            finally:
                for cfg_k, (old_exists, old_val) in st_backup.items():
                    if old_exists:
                        setattr(st_block, cfg_k, old_val)
                    else:
                        try:
                            delattr(st_block, cfg_k)
                        except AttributeError:
                            pass

                if not st_block_existed:
                    try:
                        delattr(ev, "stochastic")
                    except AttributeError:
                        pass

        return results if (rank == 0 or not gather_to_rank0) else {}