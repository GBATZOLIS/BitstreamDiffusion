import json
from pathlib import Path

import numpy as np

import torch
from ml_collections import config_dict

# Task IDs
UNCONDITIONAL = 0
TEXT_TO_SPEECH = 1
SPEECH_TO_TEXT = 2
SPEECH_CONTINUATION = 3
CONDITIONAL_TASKS = [TEXT_TO_SPEECH, SPEECH_TO_TEXT, SPEECH_CONTINUATION]

# Continuation samples are prefixed with this many seconds of reference
# (non-generated) audio -- matches scripts/text_audio/prepare_eval.py's
# CONT_TRIM_SECONDS used when building the continuation cache.
CONT_TRIM_SECONDS = 3.0


def _trim_prefix_seconds(wav, seconds: float, sr: int):
    """Drops the first `seconds` of a waveform (the reference prefix for
    continuation samples). Returns None if nothing is left afterward."""
    if wav is None:
        return None
    arr = np.asarray(wav, dtype=np.float32).flatten()
    skip = int(round(seconds * sr))
    trimmed = arr[skip:]
    return trimmed if trimmed.size > 0 else None

def _sample_tasks_and_cond_masks(
    cfg: config_dict.ConfigDict, B: int, S: int, device: torch.device, bits_per_token: int = 18
) -> tuple[torch.Tensor, torch.Tensor]:
    text_len = int(getattr(cfg.data, 'text_seq_len', 168))
    speaker_len = int(getattr(cfg.data, 'speaker_seq_len', 32))

    # Number of prefix tokens for speech continuation
    prefix_len = int(getattr(cfg.cond, 'continuation_prefix', 75))

    text_end = text_len
    speaker_end = text_len+speaker_len
    cont_end = speaker_end+prefix_len
    
    task_weights = torch.tensor([
        float(getattr(cfg.cond, 'unconditional_rate', 0.25)),
        float(getattr(cfg.cond, 'texttospeech_rate', 0.25)),
        float(getattr(cfg.cond, 'speechtotext_rate', 0.25)),
        float(getattr(cfg.cond, 'continuation_rate', 0.25))
    ])
    assert torch.sum(task_weights) == 1, 'Masking ratios must sum up to 1'
    task_ids = torch.multinomial(task_weights, B, replacement=True)

    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for task in CONDITIONAL_TASKS:
        sel = task_ids == task
        if not sel.any():
            continue
        
        if task == TEXT_TO_SPEECH:
            mask[sel, :speaker_end*bits_per_token] = True
        elif task == SPEECH_TO_TEXT:
            mask[sel, text_end*bits_per_token:] = True
        else: # Continuation
            mask[sel, text_end*bits_per_token:cont_end*bits_per_token] = True
    
    return task_ids, mask

def _fixed_mask(
    cfg: config_dict.ConfigDict, B: int, S: int, task: int, device: torch.device, bits_per_token: int = 18
) -> torch.Tensor:
    text_len = int(getattr(cfg.data, 'text_seq_len', 168))
    speaker_len = int(getattr(cfg.data, 'speaker_seq_len', 32))

    # Number of prefix tokens for speech continuation
    prefix_len = int(getattr(cfg.cond, 'continuation_prefix', 75))

    text_end = text_len
    speaker_end = text_len+speaker_len
    cont_end = speaker_end+prefix_len

    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    if task == TEXT_TO_SPEECH:
        mask[:, :speaker_end*bits_per_token] = True
    elif task == SPEECH_TO_TEXT:
        mask[:, text_end*bits_per_token:] = True
    elif task == SPEECH_CONTINUATION:
        mask[:, text_end*bits_per_token:cont_end*bits_per_token] = True
    
    return mask
    
def _safe_decode(enc, token_ids):
    parts = []
    for t in token_ids:
        try:
            parts.append(enc.decode_single_token_bytes(t))
        except KeyError:
            parts.append(b'<unk>')
    return b"".join(parts).decode("utf-8", errors="replace")

def _write_wav(path, arr, sr: int) -> None:
    import wave
    arr = np.asarray(arr, dtype=np.float32).flatten()
    pcm = (arr * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())

def _read_wav(path) -> np.ndarray:
    import wave
    with wave.open(str(path), 'rb') as wf:
        assert wf.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        pcm = wf.readframes(wf.getnframes())
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    return arr

def load_ref_audio_cache(root, split: str, task: str, partition: str = 'clean', dbg=None) -> list:
    assert task in ('tts', 'asr'), f"load_ref_audio_cache: unknown task={task!r}"
    root = Path(str(root)) if not isinstance(root, Path) else root
    if split == 'val':
        path = root / 'validation' / f'cache_val_{task}_632.ref_audio.npz'
    else:
        path = root / 'test_common' / f'cache_test_{partition}_{task}.ref_audio.npz'

    if not path.exists():
        msg = f"[{task} ref-audio] {path} not found -- reference audio will be skipped for now"
        if dbg:
            dbg(msg)
        else:
            print(msg)
        return []

    data = np.load(path, allow_pickle=True)
    return list(data['wavs'])

# Code taken from https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
# Pasted to bypass import issues

def _save_fsd_features(
    root: Path,
    *,
    extractor: str,
    model: str,
    layer,
    sample_rate: int,
    batch_size: int,
    num_samples: int,
    shards: list[tuple[int, int, np.ndarray, np.ndarray]],
    sample_keys=None,
    sample_key_name: str = 'sample_keys',
) -> None:
    """Write the shard/manifest/statistics layout used by external FSD extractors."""
    root.mkdir(parents=True, exist_ok=True)
    keys = list(sample_keys) if sample_keys is not None else None
    if keys is not None and len(keys) != num_samples:
        raise ValueError(
            f'FSD sample_keys has {len(keys)} entries for {num_samples} waveforms'
        )
    for start, stop, embeddings, offsets in shards:
        payload = {
            'embeddings': np.asarray(embeddings, dtype=np.float32),
            'offsets': np.asarray(offsets, dtype=np.int64),
            'sample_indices': np.arange(start, stop, dtype=np.int64),
        }
        np.savez(root / f'embeddings_{start:06d}_{stop - 1:06d}.npz', **payload)

    all_embeddings = np.concatenate([shard[2] for shard in shards], axis=0)
    np.savez(
        root / 'statistics.npz',
        mean=np.mean(all_embeddings, axis=0).astype(np.float32),
        cov=np.cov(all_embeddings, rowvar=False),
        count=np.int64(len(all_embeddings)),
        num_shards=np.int64(len(shards)),
    )
    manifest = {
        'format_version': 1,
        'extractor': extractor,
        'model': model,
        'layer': layer,
        'sample_rate': int(sample_rate),
        'batch_size': int(batch_size),
        'num_samples': int(num_samples),
    }
    if keys is not None:
        manifest[sample_key_name] = keys
    (root / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
