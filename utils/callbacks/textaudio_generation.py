from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.distributed as dist
except Exception:
    dist = None

from utils.textaudio_utils import _fixed_mask, _safe_decode, TextAudioEvaluator, _write_wav, load_ref_audio_cache
from utils.model_utils import unwrap_model

# -----------------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------------
def _ddp_is_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return int(dist.get_rank()) if _ddp_is_on() else 0

def _world() -> int:
    return int(dist.get_world_size()) if _ddp_is_on() else 1


def _rank0() -> bool:
    return (not _ddp_is_on()) or (_rank() == 0)


def _dbg(msg: str, *, all_ranks: bool = False) -> None:
    if all_ranks:
        print(f"[TextAudio][rank{_rank()}/{_world()}] {msg}", flush=True)
    elif _rank0():
        print(f"[TextAudio][rank{_rank()}/{_world()}] {msg}", flush=True)

def _tb(trainer) -> Optional[Any]:
    tb = getattr(trainer, "tb", None)
    if tb is not None:
        return tb
    tb = getattr(trainer, "tb_manager", None)
    if tb is not None:
        return tb
    return None

def _global_step(trainer, epoch: int) -> int:
    gs = getattr(trainer, "global_step", None)
    if gs is None:
        return int(epoch)
    try:
        return int(gs)
    except Exception:
        return int(epoch)

# -----------------------------------------------------------------------------
# Config resolution
# -----------------------------------------------------------------------------
@dataclass
class _SamplerSpec:
    tag: str              # unique key used for logging/results/save-to-disk
    sampler_name: str     # base algorithm, e.g. "ddim_entropic"
    num_steps: int
    target_nfe: int
    actual_nfe: int
    stochastic_enabled: bool
    s_churn: Optional[float]  # extract from config
    gamma: Optional[float]    # report
    guidance_scale: float


@dataclass
class _ResolvedTextAudio:
    enabled: bool
    every_k_epochs: int
    run_on_sanity: bool
    split: str

    num_samples: int
    samplers: List[_SamplerSpec]
    terminal_sigma: float
    entropic_blend_alpha: float
    entropy_run_dir: Optional[str]
    seed: int

    whisper_model: str
    sample_rate: int

    bits_per_token: int
    text_seq_len: int
    speaker_seq_len: int
    speech_seq_len: int
    speech_offset: int
    speech_vocab_size: int
    sequence_len: int

    use_amp: bool

def _first_scalar(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else default
    return x

def _build_sampler_specs(cfg: Any, c: Any) -> List[_SamplerSpec]:
    """
    Expands cfg.train.textaudio.sampling_sweep.specs (mirrors
    cfg.evaluation.sampling_sweep.specs, see configs/owt/eval_*.py) into one
    _SamplerSpec per (target_nfe x s_churn x guidance_scale) combination.
    """
    from evaluation.nfe import steps_for_target_nfe

    framework = str(getattr(cfg, 'framework', 'continuous_score'))
    self_condition = bool(getattr(getattr(cfg, 'model', None), 'self_condition', False))

    sweep = getattr(c, 'sampling_sweep', None) if c is not None else None
    raw_specs = list(getattr(sweep, 'specs', [])) if sweep is not None else []
    if not raw_specs:
        raise ValueError(
            "cfg.train.textaudio.sampling_sweep.specs must define at least one spec "
            "(sampler_name + target_nfes, optionally stochastic_enabled + s_churns, "
            "optionally guidance_scales)."
        )

    specs: List[_SamplerSpec] = []
    seen_tags = set()

    for base in raw_specs:
        sampler_name = str(base.sampler_name)
        target_nfes = [int(n) for n in base.target_nfes]
        stochastic_enabled = bool(getattr(base, 'stochastic_enabled', False))

        if stochastic_enabled:
            s_churns = [float(v) for v in getattr(base, 's_churns', [])]
            if not s_churns:
                raise ValueError(
                    f"stochastic_enabled=True but no s_churns given for sampler_name={sampler_name!r}"
                )
        else:
            s_churns = [None]

        guidance_scales = [float(v) for v in getattr(base, 'guidance_scales', [0.0])]
        if not guidance_scales:
            guidance_scales = [0.0]

        for target_nfe in target_nfes:
            num_steps, actual_nfe = steps_for_target_nfe(
                framework=framework,
                sampler_name=sampler_name,
                target_nfe=target_nfe,
                self_condition=self_condition,
                sc_refresh_mode="carry",
                return_probs=True,
            )
            for s_churn in s_churns:
                gamma = (
                    min(float(s_churn) / float(num_steps), math.sqrt(2.0) - 1.0)
                    if (stochastic_enabled and num_steps > 0) else None
                )
                for guidance_scale in guidance_scales:
                    tag = f"{sampler_name}_nfe{target_nfe}"
                    if stochastic_enabled:
                        tag += f"_stoch-g{gamma:.4g}"
                    if guidance_scale > 0.0:
                        tag += f"_gs{guidance_scale:g}"

                    if tag in seen_tags:
                        raise ValueError(f"Duplicate textaudio sampling spec tag: {tag!r}")
                    seen_tags.add(tag)

                    specs.append(_SamplerSpec(
                        tag=tag,
                        sampler_name=sampler_name,
                        num_steps=num_steps,
                        target_nfe=target_nfe,
                        actual_nfe=actual_nfe,
                        stochastic_enabled=stochastic_enabled,
                        s_churn=s_churn,
                        gamma=gamma,
                        guidance_scale=guidance_scale,
                    ))

    return specs

def _resolve_cfg(cfg: Any) -> _ResolvedTextAudio:
    train = getattr(cfg, 'train', None)
    data = getattr(cfg, 'data', None)

    c = getattr(train, 'textaudio', None)
    gen = getattr(train, 'generation', None)

    enabled = bool(getattr(c, 'enabled', False))

    run_on_sanity = False
    if c is not None and getattr(c, 'run_on_sanity', None) is not None:
        run_on_sanity = bool(getattr(c, 'run_on_sanity', None))

    k = None
    if c is not None:
        k = getattr(c, "every_k_epochs", None)
        if k is None:
            k = getattr(c, "every_epochs", None)
        if k is None:
            k = getattr(c, "every", None)
    if k is None:
        k = 10 if enabled else 1
    every_k_epochs = max(1, int(k))   

    split = 'val'
    if c is not None:
        s = getattr(c, 'split', None)
        if s is not None:
            split = str(s).lower()

    def pick(name:str, gen_name: str, default): 
        if c is not None and getattr(c, name, None) is not None:
            return getattr(c, name)
        if gen is not None and getattr(gen, gen_name, None) is not None:
            return getattr(gen, gen_name)
        return default
    
    num_samples = int(pick('num_samples', 'num_samples', 64))
    samplers = _build_sampler_specs(cfg, c)

    terminal_sigma = getattr(c, "terminal_sigma", None) if c is not None else None
    if terminal_sigma is None and c is not None:
        terminal_sigma = getattr(c, "terminal_sigmas", None)
    if terminal_sigma is None and gen is not None:
        terminal_sigma = getattr(gen, "terminal_sigmas", None)
    terminal_sigma = float(_first_scalar(terminal_sigma, 0.08))

    entropic_blend_alpha = float(pick("entropic_blend_alpha", "entropic_blend_alpha", 0.0))

    entropy_run_dir = None
    if c is not None:
        entropy_run_dir = getattr(c, "entropy_run_dir", None)
        if entropy_run_dir is None:
            entropy_run_dir = getattr(c, "entropy_ckpt_path", None)
    if entropy_run_dir is None and gen is not None:
        entropy_run_dir = getattr(gen, "entropy_ckpt_path", None)

    seed = int(pick("seed", "seed", 42))

    whisper_model = "openai/whisper-medium"
    if c is not None and getattr(c, "whisper_model", None) is not None:
        whisper_model = str(getattr(c, "whisper_model"))
 
    sample_rate = int(getattr(data, "sample_rate", 16000)) if data is not None else 16000    

    bits_per_token = int(getattr(data, "bits_per_token", 18)) if data is not None else 18
    text_seq_len = int(getattr(data, "text_seq_len", 168)) if data is not None else 168
    speaker_seq_len = int(getattr(data, "speaker_seq_len", 32)) if data is not None else 32
    speech_seq_len = int(getattr(data, "speech_seq_len", 800)) if data is not None else 800
    speech_offset = int(getattr(data, "speech_offset", 204115)) if data is not None else 204115
    speech_vocab_size = int(getattr(data, "speech_vocab_size", 46656)) if data is not None else 46656
    sequence_len = int(getattr(data, "sequence_len", 1000)) if data is not None else 1000    

    use_amp = bool(getattr(c, "use_amp", True)) if c is not None else True

    return _ResolvedTextAudio(
        enabled=enabled,
        every_k_epochs=every_k_epochs,
        run_on_sanity=run_on_sanity,
        split=split,
        num_samples=num_samples,
        samplers=samplers,
        terminal_sigma=terminal_sigma,
        entropic_blend_alpha=entropic_blend_alpha,
        entropy_run_dir=entropy_run_dir,
        seed=seed,
        whisper_model=whisper_model,
        sample_rate=sample_rate,
        bits_per_token=bits_per_token,
        text_seq_len=text_seq_len,
        speaker_seq_len=speaker_seq_len,
        speech_seq_len=speech_seq_len,
        speech_offset=speech_offset,
        speech_vocab_size=speech_vocab_size,
        sequence_len=sequence_len,
        use_amp=use_amp,
    )

TASKS = ['joint', 'tts', 'stt', 'cont']


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------
class TextAudioCallback:
    run_on_all_ranks=False

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._ddim_sampler = None
        self._evaluator = None
        self._last_run_key = None
        self._tts_ref_audio_cache: Optional[list] = None
        self._stt_ref_audio_cache: Optional[list] = None

    def _ensure_evaluator(self, r: _ResolvedTextAudio) -> None:
        if self._evaluator is not None:
            return
        self._evaluator = TextAudioEvaluator(
            whisper_model=r.whisper_model,
            sr=r.sample_rate,
            _dbg_func=_dbg,
        )

    def _ensure_tts_ref_audio_cache(self, data_cfg) -> list:
        if self._tts_ref_audio_cache is not None:
            return self._tts_ref_audio_cache
        root = getattr(data_cfg, 'root', 'datasets/') if data_cfg is not None else 'datasets/'
        self._tts_ref_audio_cache = load_ref_audio_cache(root, split='val', task='tts', dbg=_dbg)
        return self._tts_ref_audio_cache

    def _ensure_stt_ref_audio_cache(self, data_cfg) -> list:
        if self._stt_ref_audio_cache is not None:
            return self._stt_ref_audio_cache
        root = getattr(data_cfg, 'root', 'datasets/') if data_cfg is not None else 'datasets/'
        self._stt_ref_audio_cache = load_ref_audio_cache(root, split='val', task='asr', dbg=_dbg)
        return self._stt_ref_audio_cache

    def _should_run(self, epoch: int, r: _ResolvedTextAudio) -> bool:
        if not r.enabled:
            return False
        # epoch < 0 → sanity check run
        if int(epoch) < 0:
            return bool(r.run_on_sanity)
        k = max(1, int(r.every_k_epochs))
        return ((int(epoch) + 1) % k) == 0
    
    @torch.no_grad()
    def _sample_real_data_for_task(
        self, trainer, B: int, split: str, task: str
    ) -> Optional[torch.Tensor]:
        data_cfg = trainer.cfg.data
        text_seq_len = int(data_cfg.text_seq_len)
        bits_per_token = int(data_cfg.bits_per_token)
        sequence_len = int(data_cfg.sequence_len)
        
        if task == 'joint':
            return torch.zeros(B, sequence_len, device=trainer.device, dtype=torch.float32)

        # Use the appropriate conditioning data for each type of task
        if task == 'tts':
            loader = getattr(trainer, f'{split}_loader_tts', None)
            start = 0
        elif task == 'cont':
            loader = getattr(trainer, f'{split}_loader_cont', None)
            start = text_seq_len * bits_per_token
        else:
            loader = getattr(trainer, f'{split}_loader', None)
            start = 0

        if loader is None:
            return None

        dataset = loader.dataset
        if len(dataset) == 0:
            return None

        n = min(B, len(dataset))
        chunks = []
        for i in range(n):
            sample = dataset[i]
            if isinstance(sample, (tuple, list)):
                sample = sample[0]
            sample = sample.view(-1).to(device=trainer.device, dtype=torch.float32, non_blocking=True)
            chunks.append(sample.unsqueeze(0))

        partial = torch.cat(chunks, dim=0).contiguous()

        if task == 'stt':
            return partial

        full = torch.zeros(partial.size(0), sequence_len, device=trainer.device, dtype=torch.float32)
        full[:, start:start + partial.size(1)] = partial
        return full

    def _get_sampler(self, trainer, sampler_name: str):
        raw = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if "ddim" in sampler_name:
            from diffusion.continuous.samplers import DDIMSampler
            if self._ddim_sampler is None:
                self._ddim_sampler = DDIMSampler(raw, trainer.proc, self.cfg)
            return self._ddim_sampler
        return trainer.sampler

    @contextmanager
    def _stochastic_cfg_override(self, overrides: Dict[str, Any]):
        if not overrides:
            yield
            return

        ev = getattr(self.cfg, 'evaluation', None)
        if ev is None:
            yield
            return

        st = getattr(ev, 'stochastic', None)
        st_existed = st is not None
        if st is None:
            from ml_collections import config_dict
            ev.stochastic = config_dict.ConfigDict()
            st = ev.stochastic

        backup = {}
        for key, value in overrides.items():
            backup[key] = (hasattr(st, key), getattr(st, key, None))
            setattr(st, key, value)

        try:
            yield
        finally:
            for key, (existed, old_val) in backup.items():
                if existed:
                    setattr(st, key, old_val)
                else:
                    try:
                        delattr(st, key)
                    except AttributeError:
                        pass
            if not st_existed:
                try:
                    delattr(ev, 'stochastic')
                except AttributeError:
                    pass

    def _generate_task(
        self, trainer, x_full: torch.Tensor, task_id: int, B: int, r: _ResolvedTextAudio,
        spec: _SamplerSpec,
    ) -> torch.Tensor:
        cond_mask = _fixed_mask(
            self.cfg, B, r.sequence_len, task_id, device=trainer.device, bits_per_token=r.bits_per_token
        )
        schedule = 'entropic' if 'entropic' in spec.sampler_name else 'karras'
        sampler_obj = self._get_sampler(trainer, spec.sampler_name)

        stoch_overrides: Dict[str, Any] = {}
        if not spec.stochastic_enabled:
            stoch_overrides['enabled'] = False
        else:
            stoch_overrides['enabled'] = True
            if spec.s_churn is not None:
                stoch_overrides['s_churn'] = spec.s_churn

        entropy_run_dir = r.entropy_run_dir
        if entropy_run_dir is None:
            entropy_run_dir = str(getattr(trainer, 'run_dir', '.'))

        cond_kwargs = dict(
            conditioning_prefix_full=x_full,
            cond_prefix_mask=cond_mask,
            guidance_scale=spec.guidance_scale,
        )

        with self._stochastic_cfg_override(stoch_overrides):
            _, probs = sampler_obj.sample(
                B, r.sequence_len,
                schedule=schedule,
                num_steps=spec.num_steps,
                sc_refresh_mode="carry",
                entropic_blend_alpha=r.entropic_blend_alpha,
                entropy_run_dir=entropy_run_dir,
                sigma_min_override=r.terminal_sigma,
                return_probs=True,
                progress=True,
                **cond_kwargs
            )

        bits = (probs > 0.5).to(torch.long)
        bits[cond_mask] = (x_full[cond_mask] > 0.5).to(torch.long)
        return bits

    def _bits_to_token_ids(
        self, bits: torch.Tensor, r: _ResolvedTextAudio
    ) -> torch.Tensor:
        B = bits.size(0)
        tok_bits = bits.view(B, -1, r.bits_per_token)
        shifts = torch.arange(
            r.bits_per_token-1,-1,-1, device=bits.device
        )
        return (tok_bits * (2**shifts)).sum(dim=-1).long()
    
    def _decode_texts(
        self, trainer, token_ids: torch.Tensor, r: _ResolvedTextAudio
    ) -> List[str]:
        PAD_TEXT = r.speech_offset+r.speech_vocab_size
        ids = token_ids[:, :r.text_seq_len]
        pad = ids == PAD_TEXT
        ids_clean = ids.clone()
        ids_clean[pad] = 0

        texts = []
        for i in range(ids.size(0)):
            valid = ids_clean[i][~pad[i]].cpu().tolist()
            texts.append(_safe_decode(trainer.text_tok, valid))
        return texts
    
    def _decode_speech_wavs(
        self, trainer, token_ids: torch.Tensor, r: _ResolvedTextAudio
    ):
        PAD_SPEECH = r.speech_offset+r.speech_vocab_size+1
        global_ids = token_ids[:, r.text_seq_len+r.speaker_seq_len:]
        pad = global_ids == PAD_SPEECH
        local_ids = (global_ids-r.speech_offset).clamp(0, r.speech_vocab_size-1)

        wavs = []
        for i in range(local_ids.size(0)):
            valid = local_ids[i][~pad[i]]
            wav = trainer.speech_tok.decode([valid.unsqueeze(0).unsqueeze(-1)], posthoc_bottleneck=True)
            wav = wav.squeeze(0).cpu().float().squeeze(0).numpy()
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = wav / peak
            wavs.append(wav)
        return wavs
    
    def _log_scalars(
        self, trainer, metrics: Dict[str, float], tag: str, step: int
    ):
        for name, value in metrics.items():
            if math.isnan(value):
                continue
            key = f"textaudio/{tag}/{name}"
            w = getattr(trainer, 'writer', None)
            if w is not None:
                try:
                    w.add_scalar(key, value, step)
                except TypeError:
                    w.add_scalar(key, value)
            
            if getattr(trainer, 'use_wandb', False):
                try: 
                    trainer._log_wandb({key:value})
                except Exception:
                    pass

            _dbg(f'{key} = {value:.4f}')

    def _log_samples(
        self, trainer, task, tag: str, step: int,
        gen_texts, ref_texts, gen_wavs, ref_wavs, transcriptions,
        r: _ResolvedTextAudio
    ):
        has_gen_text = task in ('joint', 'stt', 'cont')
        has_ref_text = task in ('tts', 'stt')
        has_gen_audio = task in ('joint', 'tts', 'cont')
        has_ref_audio = task in ('tts', 'stt')
        has_whisper = transcriptions is not None

        w = getattr(trainer, 'writer', None)
        n = r.num_samples
        prefix = f'textaudio/{tag}'


        if w is not None:
            lines = []
            for i in range(n):
                lines.append(f'Sample {i}')
                if has_gen_text:
                    lines.append(f' gen_text:   {repr(gen_texts[i])}')
                if has_ref_text:
                    lines.append(f' ref_text:   {repr(ref_texts[i])}')
                if has_whisper:
                    lines.append(f' whisper:    {repr(transcriptions[i])}')
                lines.append('')
            body = '\n'.join(lines)
            try:
                w.add_text(f'{prefix}/samples', f'```\n{body}\n```', step)
            except Exception:
                pass

            if has_gen_audio:
                for i, wav in enumerate(gen_wavs):
                    if wav is None:
                        continue
                    w.add_audio(
                        f'{prefix}/gen_audio_{i}', wav, step, sample_rate = r.sample_rate 
                    )
            if has_ref_audio:
                for i, wav in enumerate(ref_wavs):
                    if wav is None:
                        continue
                    w.add_audio(
                        f'{prefix}/ref_audio_{i}', wav, step, sample_rate = r.sample_rate 
                    )                
            
        if getattr(trainer, 'use_wandb', False) and wandb is not None:
            try:
                columns = ['idx']
                if has_ref_text:
                    columns.append('ref_text')
                if has_gen_text:
                    columns.append('gen_text')
                if has_ref_audio:
                    columns.append('ref_audio')
                if has_gen_audio:
                    columns.append('gen_audio')
                if has_whisper:
                    columns.append('whisper')

                table = wandb.Table(columns=columns)

                for i in range(n):
                    row = [i]
                    if has_ref_text:
                        row.append(ref_texts[i])
                    if has_gen_text:
                        row.append(gen_texts[i])
                    if has_ref_audio:
                        row.append(wandb.Audio(ref_wavs[i], sample_rate=r.sample_rate))
                    if has_gen_audio:
                        row.append(wandb.Audio(gen_wavs[i], sample_rate=r.sample_rate))
                    if has_whisper:
                        row.append(transcriptions[i])
                        
                    table.add_data(*row)
                
                trainer._log_wandb({f'textaudio/{tag}/samples': table})
            except Exception as e:
                _dbg(f'W&B table log failed {e}')

    def _save_to_disk(
        self, trainer, all_results: Dict[str, Dict[str, dict]], r: _ResolvedTextAudio, step: int, epoch: int,
        save_dir_override: Optional[Path] = None,
    ) -> None:
        if save_dir_override is not None:
            save_dir = Path(save_dir_override)
        else:
            run_dir = Path(str(getattr(trainer, 'run_dir', 'runs/default')))
            base_dir = run_dir / 'textaudio'
            save_dir = base_dir / f'step_{step:09d}'
        save_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = getattr(self.cfg, 'data', None)

        report_data = {
            'header': {
                'title': 'Single-stream Joint Audio+Text Generation',
                'experiment': str(getattr(self.cfg, 'experiment', 'unknown')) if data_cfg else 'unknown',
                'text_tokenizer': str(
                    getattr(data_cfg, 'text_tokenizer', 'o200k_base')
                ) if data_cfg else 'o200k_base',
                'speaker_tokenizer': str(
                    getattr(data_cfg, 'speaker_tokenizer', 'bicodec')
                ) if data_cfg else 'bicodec',
                'speech_tokenizer': str(
                    getattr(data_cfg, 'speech_tokenizer', 'stabilityai/stable-codec-speech-16k')
                ) if data_cfg else 'stabilityai/stable-codec-speech-16k',
                'speech_tokenizer_bottleneck': str(
                    getattr(data_cfg,'speech_tokenizer_bottleneck', '1x46656_400bps')
                ) if data_cfg else '1x46656_400bps',
                'bits_per_token': r.bits_per_token,
                'sequence_layout': {
                    'text_tokens': r.text_seq_len,
                    'speaker_tokens': r.speaker_seq_len,
                    'speech_tokens': r.speech_seq_len,
                    'total_tokens': r.sequence_len
                },
                'split': r.split,
                'step': step,
                'epoch': epoch,
                'num_samples': r.num_samples,
                'samplers': [
                    {
                        'name': spec.tag,
                        'sampler_name': spec.sampler_name,
                        'num_steps': spec.num_steps,
                        'target_nfe': spec.target_nfe,
                        'actual_nfe': spec.actual_nfe,
                        'stochastic_enabled': spec.stochastic_enabled,
                        'gamma': spec.gamma,
                        'terminal_sigma': r.terminal_sigma,
                        'guidance_scale': spec.guidance_scale,
                    }
                    for spec in r.samplers
                ],
                'sample_rate': r.sample_rate,
            },
            'samplers': {},
        }

        written_ref_tasks: set = set()

        for sampler_tag, task_results in all_results.items():
            sampler_entry: Dict[str, dict] = {'tasks': {}}

            for task, res in task_results.items():
                task_dir = save_dir / sampler_tag / task
                task_dir.mkdir(parents=True, exist_ok=True)

                gen_texts = res.get('gen_texts', [])
                ref_texts = res.get('ref_texts', [])
                gen_wavs = res.get('gen_wavs', [])
                ref_wavs = res.get('ref_wavs')
                transcriptions = res.get('transcriptions')
                metrics = res.get('metrics', {})

                if task in ('tts', 'stt') and task not in written_ref_tasks and ref_wavs is not None:
                    ref_dir = save_dir / task
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(min(r.num_samples, len(ref_wavs))):
                        _write_wav(ref_dir / f'ref_{i:04d}.wav', ref_wavs[i], r.sample_rate)
                    written_ref_tasks.add(task)

                samples = []
                n = r.num_samples

                for i in range(n):
                    sample = {'idx': i}

                    if task in ('joint', 'stt', 'cont'):
                        sample['gen_text'] = gen_texts[i]
                    if task in ('tts', 'stt'):
                        sample['ref_text'] = ref_texts[i] if i < len(ref_texts) else None

                    if task in ('joint', 'tts', 'cont'):
                        wav_rel = f'{sampler_tag}/{task}/gen_{i:04d}.wav'
                        _write_wav(save_dir / wav_rel, gen_wavs[i], r.sample_rate)
                        sample['gen_wav'] = wav_rel

                    if task in ('tts', 'stt'):
                        sample['ref_wav'] = (
                            f'{task}/ref_{i:04d}.wav' if ref_wavs is not None and i < len(ref_wavs) else None
                        )

                    if task in ('joint', 'tts'):
                        sample['whisper'] = transcriptions[i]

                    samples.append(sample)

                clean_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, float) and math.isnan(v):
                        clean_metrics[k] = None
                    else:
                        clean_metrics[k] = v

                sampler_entry['tasks'][task] = {
                    'metrics': clean_metrics,
                    'samples': samples,
                }

            report_data['samplers'][sampler_tag] = sampler_entry

        with open(save_dir / 'data.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        _dbg(f"Saved to {save_dir}")

    @torch.compiler.disable
    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int, save_dir_override: Optional[Path] = None) -> None:
        r = _resolve_cfg(self.cfg)
        
        if not self._should_run(epoch, r):
            return
        if not _rank0():
            return  # generation output is master-only; no other rank needs to run this

        run_key = (int(epoch), _global_step(trainer, epoch))
        
        if self._last_run_key == run_key:
            return
        self._last_run_key = run_key

        if _rank0():
            t0 = time.perf_counter()

        B = r.num_samples
        raw = unwrap_model(trainer.model)

        # ── EMA, eval, generate (rank 0 only; enforced at function entry) ──
        task_x_full: Dict[str, torch.Tensor] = {}
        sampler_task_bits: Dict[str, Dict[str, torch.Tensor]] = {spec.tag: {} for spec in r.samplers}
        has_data = False

        try:
            trainer.model.eval()
            trainer.ema.apply(trainer.model)   # match _validate_epoch

            torch.manual_seed(r.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(r.seed)

            amp_dtype = getattr(trainer, "amp_dtype", torch.float16)

            with autocast(enabled=r.use_amp, dtype=amp_dtype):
                for task_id, task in enumerate(TASKS):
                    if task == 'joint':
                        # Skip joint (unconditional) generation for guided samplers
                        specs_for_task = [s for s in r.samplers if s.guidance_scale <= 0.0]
                        if not specs_for_task:
                            if _rank0():
                                _dbg(
                                    "Skipping joint (unconditional) generation entirely: every "
                                    "sampler in this sweep has guidance_scale>0."
                                )
                            continue
                    else:
                        specs_for_task = r.samplers

                    x_full = self._sample_real_data_for_task(trainer, B, r.split, task)
                    if x_full is None:
                        if _rank0():
                            _dbg(f"No data from {r.split} split for task={task}")
                        continue

                    has_data = True
                    task_B = x_full.size(0)
                    task_x_full[task] = x_full

                    if _rank0():
                        _dbg(
                            f"START epoch={epoch} split={r.split} task={task} "
                            f"B={task_B} samplers={[s.tag for s in specs_for_task]} "
                            f"sigma={r.terminal_sigma}"
                        )

                    for spec in specs_for_task:
                        if _rank0():
                            _dbg(f"  {r.split}_{task} [{spec.tag}]: generating {task_B} samples (steps={spec.num_steps})")
                        try:
                            bits = self._generate_task(trainer, x_full, task_id, task_B, r, spec)
                            sampler_task_bits[spec.tag][task] = bits
                        except Exception as e:
                            if _rank0():
                                _dbg(f"  {r.split}_{task} [{spec.tag}]: generation failed: {e}")

        finally:
            trainer.ema.restore(trainer.model)   # match _validate_epoch
            trainer.model.train()

        # ── decode, evaluate, save (rank 0 only; enforced at function entry) ──

        if has_data and any(sampler_task_bits.values()):
            step = _global_step(trainer, epoch)
            self._ensure_evaluator(r)
            all_results: Dict[str, Dict[str, dict]] = {}  # {sampler_tag: {task: result}}

            ref_texts_by_task: Dict[str, list] = {}
            ref_wavs_by_task: Dict[str, Optional[list]] = {}
            for task in ("tts", "stt"):
                if task not in task_x_full:
                    continue
                ref_token_ids = self._bits_to_token_ids(
                    (task_x_full[task] > 0.5).to(torch.long), r,
                )
                ref_texts_by_task[task] = self._decode_texts(trainer, ref_token_ids, r)
                if task == "stt":

                    stt_cache = self._ensure_stt_ref_audio_cache(getattr(trainer.cfg, "data", None))
                    n_stt = task_x_full[task].size(0)
                    if stt_cache and n_stt <= len(stt_cache):
                        ref_wavs_by_task[task] = list(stt_cache[:n_stt])
                    else:
                        ref_wavs_by_task[task] = self._decode_speech_wavs(trainer, ref_token_ids, r)
                else:

                    cache = self._ensure_tts_ref_audio_cache(getattr(trainer.cfg, "data", None))
                    n = task_x_full[task].size(0)
                    ref_wavs_by_task[task] = list(cache[:n]) if cache else None

            for sampler_tag, task_bits in sampler_task_bits.items():
                sampler_results: Dict[str, dict] = {}

                for task, bits in task_bits.items():
                    _dbg(f"  {r.split}_{task} [{sampler_tag}]: decoding+evaluating")
                    try:
                        needs_gen_text = task in ("joint", "stt", "cont")
                        needs_gen_wav  = task in ("joint", "tts", "cont")
                        needs_ref      = task in ("tts", "stt")

                        token_ids  = self._bits_to_token_ids(bits, r)
                        gen_texts  = self._decode_texts(trainer, token_ids, r) if needs_gen_text else []
                        gen_wavs   = self._decode_speech_wavs(trainer, token_ids, r) if needs_gen_wav else []

                        ref_texts = ref_texts_by_task.get(task, []) if needs_ref else []
                        ref_wavs  = ref_wavs_by_task.get(task) if needs_ref else None

                        metrics, transcriptions = self._evaluator.evaluate_task_brief(
                            task, gen_texts, ref_texts, gen_wavs, trainer.device,
                        )
                    except Exception as e:
                        _dbg(f"  {r.split}_{task} [{sampler_tag}]: eval failed: {e}")
                        continue

                    sampler_results[task] = {
                        "metrics": metrics, "gen_texts": gen_texts,
                        "ref_texts": ref_texts, "gen_wavs": gen_wavs,
                        "ref_wavs": ref_wavs, "transcriptions": transcriptions,
                    }

                    tag = f"{r.split}_{task}/{sampler_tag}"
                    self._log_scalars(trainer, metrics, tag, step)
                    self._log_samples(
                        trainer, task, tag, step,
                        gen_texts, ref_texts, gen_wavs, ref_wavs, transcriptions, r,
                    )

                all_results[sampler_tag] = sampler_results

            if all_results:
                try:
                    self._save_to_disk(trainer, all_results, r, step, epoch, save_dir_override=save_dir_override)
                except Exception as e:
                    _dbg(f"Local save failed: {e}")

        elapsed = time.perf_counter() - t0
        w = getattr(trainer, "writer", None)
        if w is not None:
            try:
                w.add_scalar("textaudio/timing_sec", elapsed, _global_step(trainer, epoch))
            except Exception:
                pass
        _dbg(f"END ({elapsed:.1f}s)")

