from __future__ import annotations

import math
import io
import os
import csv
import json
import shutil
import signal
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


import torch
import torch.multiprocessing as mp
from torch.amp import autocast
import torchaudio

from typing import Optional, Tuple


# The two cache layouts used by the text-audio experiments. Keeping this in
# one place prevents preparation scripts and model configs from drifting.
GEOMETRIES = {
    632: (100, 32, 500),
    1000: (168, 32, 800),
}

# -----------------------------------------------------------------------------
# Vocabulary layout
# -----------------------------------------------------------------------------

TEXT_VOCAB    = 200_019
SPEAKER_VOCAB = 4_096

TEXT_OFFSET    = 0
SPEAKER_OFFSET = TEXT_VOCAB          # 200_019
SPEECH_OFFSET  = TEXT_VOCAB + SPEAKER_VOCAB  # 204_115

# SPEECH_VOCAB, PAD_TOKEN, and TOTAL_VOCAB are bottleneck-dependent and
# computed at runtime in compute_speech_vocab / build_packed_cache.

MODEL_SR = 16_000  # both BiCodec and StableCodec operate at 16 kHz

# -----------------------------------------------------------------------------
# Accelerated volume normalization
# -----------------------------------------------------------------------------

def _biquad_coeffs_treble(sr: float, gain: float, central_freq: float, Q: float):
    w0 = 2 * math.pi * central_freq / sr
    alpha = math.sin(w0) / 2 / Q
    A = math.exp(gain / 40 * math.log(10))
    temp1 = 2 * math.sqrt(A) * alpha
    temp2 = (A - 1) * math.cos(w0)
    temp3 = (A + 1) * math.cos(w0)
    b0 = A * ((A + 1) + temp2 + temp1)
    b1 = -2 * A * ((A - 1) + temp3)
    b2 = A * ((A + 1) + temp2 - temp1)
    a0 = (A + 1) - temp2 + temp1
    a1 = 2 * ((A - 1) - temp3)
    a2 = (A + 1) - temp2 - temp1
    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def _biquad_coeffs_highpass(sr: float, cutoff_freq: float, Q: float):
    w0 = 2 * math.pi * cutoff_freq / sr
    alpha = math.sin(w0) / 2.0 / Q
    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return np.array([b0, b1, b2]), np.array([a0, a1, a2])

class FastVolumeNorm(torch.nn.Module):
    """Drop-in replacement for stable_audio_tools.data.utils.VolumeNorm.

    Usage matches the original: call with a (channels, time) tensor for a
    single clip, or a (batch, channels, time) tensor for a batch -- both
    return volume-normalized audio of the same shape.
    """
    
    def __init__(self, params=(-16, 2), sample_rate: int = 16_000, energy_threshold: float = 1e-6):
        super().__init__()
        self.value = params[0]
        self.gain_range = (-params[1], params[1])
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self._b_treble, self._a_treble = _biquad_coeffs_treble(sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
        self._b_highpass, self._a_highpass = _biquad_coeffs_highpass(sample_rate, 38.0, 0.5)

    def loudness(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (batch, channels=1, time). Returns (batch,) LKFS, same device/dtype as wav."""
        if wav.shape[-2] != 1:
            raise ValueError(f'FastVolumeNorm only supports mono audio, got {wav.shape[-2]} channels.')
        device, dtype = wav.device, wav.dtype
        x = wav[:, 0, :].detach().cpu().double().numpy()  # (B, T)

        x = lfilter(self._b_treble, self._a_treble, x, axis=-1)
        x = np.clip(x, -1.0, 1.0)
        x = lfilter(self._b_highpass, self._a_highpass, x, axis=-1)
        x = np.clip(x, -1.0, 1.0)

        gate_duration, overlap = 0.4, 0.75
        gamma_abs, kweight_bias = -70.0, -0.691
        gate_samples = int(round(gate_duration * self.sample_rate))
        step = int(round(gate_samples * (1 - overlap)))

        blocks = np.lib.stride_tricks.sliding_window_view(x, gate_samples, axis=-1)[..., ::step, :]
        energy = np.mean(blocks ** 2, axis=-1)  # (B, n_blocks)

        loudness_per_block = kweight_bias + 10 * np.log10(energy)
        gated = loudness_per_block > gamma_abs
        energy_filtered = np.sum(np.where(gated, energy, 0.0), axis=-1) / np.count_nonzero(gated, axis=-1)
        gamma_rel = kweight_bias + 10 * np.log10(energy_filtered) - 10
        gated2 = gated & (loudness_per_block > gamma_rel[..., None])
        energy_filtered2 = np.sum(np.where(gated2, energy, 0.0), axis=-1) / np.count_nonzero(gated2, axis=-1)
        result = kweight_bias + 10 * np.log10(energy_filtered2)  # (B,)

        return torch.as_tensor(result, dtype=dtype, device=device)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        single = signal.dim() == 2
        x = signal.unsqueeze(0) if single else signal
        if x.dim() != 3 or x.shape[-2] != 1:
            raise ValueError(
                f'expected (channels=1, time) or (batch, channels=1, time), got {tuple(signal.shape)}'
            )

        energy = torch.mean(x.reshape(x.shape[0], -1) ** 2, dim=-1)  # (B,)
        input_loudness = self.loudness(x)  # (B,)

        target_loudness = self.value + (
            torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
            * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
        )
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0).view(-1, 1, 1)

        output = gain * x
        silent = energy < self.energy_threshold
        output[silent] = x[silent]

        peak = output.abs().amax(dim=(-2, -1))
        clip_mask = peak >= 1.0
        if clip_mask.any():
            scale = torch.where(clip_mask, 0.95 / peak, torch.ones_like(peak)).view(-1, 1, 1)
            output = output * scale

        return output[0] if single else output

# -----------------------------------------------------------------------------
# Dataloader construction
# -----------------------------------------------------------------------------

class CodecDataset(torch.utils.data.Dataset):
    """Wraps an HF dataset; decoding and resampling happen in DataLoader workers."""

    def __init__(self, hf_dataset, text_field: str, target_sr: int = MODEL_SR):
        self.ds = hf_dataset
        self.text_field = text_field
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.ds[idx]
        wav, sr = torchaudio.load(io.BytesIO(row['audio']['bytes']))
        wav = wav.mean(dim=0)  # mono, shape (T,)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav, row[self.text_field]


class _CollateFn:
    """Pads a batch to its max length aligned to ds_ratio.

    A top-level class rather than a closure: DataLoader workers spawned from
    inside an already-mp.spawn'd GPU worker process must pickle the collate_fn,
    and closures aren't picklable.
    """

    def __init__(self, ds_ratio: int):
        self.ds_ratio = ds_ratio

    def __call__(
        self,
        batch: list[Tuple[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, list[str], list[int]]:
        wavs, texts = zip(*batch)
        true_lengths = [w.shape[-1] for w in wavs]
        max_len = math.ceil(max(true_lengths) / self.ds_ratio) * self.ds_ratio
        padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in wavs
        ]).unsqueeze(1)  # (B, 1, T_max)
        return padded, list(texts), true_lengths

# -----------------------------------------------------------------------------
# Tokenization logic
# Separate pad tokens are used for text and speech
# -----------------------------------------------------------------------------

def _tokenize_text_batch(
    text_tokenizer,
    texts: list[str],
    max_len: int,
    pad_token: int,
) -> Tuple[torch.Tensor, int]:
    """Returns ((B, max_len) int64 CPU PAD-filled, n_truncated)."""
    token_lists = text_tokenizer.encode_ordinary_batch(texts)
    batch = torch.full((len(texts), max_len), pad_token, dtype=torch.long)
    n_truncated = 0
    for i, toks in enumerate(token_lists):
        if len(toks) > max_len:
            n_truncated += 1
        toks = toks[:max_len]
        batch[i, :len(toks)] = torch.tensor(toks, dtype=torch.long) + TEXT_OFFSET
    return batch, n_truncated


@torch.no_grad()
def _tokenize_speaker_batch(
    bicodec,
    wav_batch: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    wav_batch: (B, 1, T) on device.
    Returns (B, seq_len) int64 on device. Always exactly seq_len tokens.
    """
    mel = bicodec.mel_transformer(wav_batch).squeeze(1)        # (B, n_mels, T_mel)
    with autocast('cuda', dtype=torch.float16):
        global_tokens = bicodec.speaker_encoder.tokenize(
            mel.transpose(1, 2)                                     # (B, T_mel, n_mels)
        )                                                           # (B, 1, seq_len)
    tokens = global_tokens.squeeze(1)                           # (B, seq_len)
    assert tokens.shape[1] == seq_len, (
        f'BiCodec speaker encoder produced {tokens.shape[1]} tokens, expected {seq_len}'
    )
    return tokens + SPEAKER_OFFSET


@torch.no_grad()
def _tokenize_speech_batch(
    stable_codec,
    wav_batch: torch.Tensor,
    true_wav_lengths: list[int],
    max_len: int,
    pad_token: int,
) -> Tuple[torch.Tensor, int]:
    """
    wav_batch: (B, 1, T) on device, already at MODEL_SR, zero-padded to batch max.
    true_wav_lengths: true waveform length in samples for each item, before padding.
    Returns ((B, max_len) int64 on device PAD-filled, n_truncated).

    pad_token is placed after each sample's true encoded length, not after the
    batch-padded length, so silence introduced by batch collation is never stored
    as real speech tokens.
    """
    normed = stable_codec.volume_norm(wav_batch)
    ds = stable_codec.model.downsampling_ratio
    T = normed.shape[-1]
    pad = (ds - T % ds) % ds
    if pad > 0:
        normed = torch.nn.functional.pad(normed, (0, pad))
    with autocast('cuda', dtype=torch.float16):
        _, tokens = stable_codec.encode(normed, posthoc_bottleneck=True)
    tokens = tokens[0]                                          # (B, S, 1) or (B, S)
    if tokens.dim() == 3:
        tokens = tokens.squeeze(-1)                             # (B, S)

    result = torch.full(
        (tokens.shape[0], max_len), pad_token, dtype=torch.long, device=tokens.device,
    )
    n_truncated = 0
    for i, true_len in enumerate(true_wav_lengths):
        true_token_len = math.ceil(true_len / ds)
        if true_token_len > max_len:
            n_truncated += 1
        true_token_len = min(true_token_len, max_len)
        result[i, :true_token_len] = tokens[i, :true_token_len] + SPEECH_OFFSET
    return result, n_truncated

# -----------------------------------------------------------------------------
# Tokenizer loading
# -----------------------------------------------------------------------------

def compute_speech_vocab(
    speech_bottleneck: str | None,
    speech_bottleneck_dims: list[int] | None,
) -> int:
    """Derive speech_vocab without loading any heavy model (cheap, used for sizing)."""
    if speech_bottleneck_dims is not None:
        return math.prod(speech_bottleneck_dims)
    return 46_656  # default for '1x46656_400bps'


def load_tokenizers(
    text_tokenizer_name: str,
    speaker_model_dir: str,
    speech_model: str,
    device: torch.device,
    speech_bottleneck: str | None = None,
    speech_bottleneck_dims: list[int] | None = None,
):
    """Load all three tokenizers and return (text_tok, bicodec, stable_codec, speech_vocab).

    Exactly one of speech_bottleneck (string preset) or speech_bottleneck_dims
    (list of per-codebook sizes, e.g. [8,8,8,8,8,8]) must be provided.
    speech_vocab is derived automatically: product of dims for the list form,
    or must be supplied explicitly for a string preset via speech_bottleneck_vocab.
    """
    if (speech_bottleneck is None) == (speech_bottleneck_dims is None):
        raise ValueError('Provide exactly one of speech_bottleneck or speech_bottleneck_dims')

    # sparktts isn't a pip-installed package -- it's the Spark-TTS/sparktts
    # submodule directory, only importable once Spark-TTS/ itself is on
    # sys.path. Do this here rather than relying on every caller's own
    # PYTHONPATH to remember it.
    import sys
    spark_tts_dir = Path(__file__).resolve().parents[2] / 'Spark-TTS'
    if str(spark_tts_dir) not in sys.path:
        sys.path.insert(0, str(spark_tts_dir))

    import tiktoken
    from sparktts.models.bicodec import BiCodec
    from stable_codec import StableCodec

    print(f'[textaudio-cache] loading text tokenizer: {text_tokenizer_name}')
    text_tok = tiktoken.get_encoding(text_tokenizer_name)

    print(f'[textaudio-cache] loading BiCodec from: {speaker_model_dir}')
    bicodec = BiCodec.load_from_checkpoint(model_dir=speaker_model_dir).to(device).eval()

    stable_codec = StableCodec(pretrained_model=speech_model, device=device)
    speech_vocab = compute_speech_vocab(speech_bottleneck, speech_bottleneck_dims)
    if speech_bottleneck_dims is not None:
        bottleneck_arg = [([speech_bottleneck_dims, 1.0])]
        print(f'[textaudio-cache] loading StableCodec: {speech_model}  '
              f'bottleneck: dims={speech_bottleneck_dims}  speech_vocab={speech_vocab}')
    else:
        bottleneck_arg = speech_bottleneck
        print(f'[textaudio-cache] loading StableCodec: {speech_model}  bottleneck: {speech_bottleneck}')
    stable_codec.set_posthoc_bottleneck(bottleneck_arg)

    # Substitute volume normalization (slow in this environment, ~20-40s for 20 second sample)
    # with fast alternative that produces identical results and also operates in batches
    orig_vn = stable_codec.volume_norm
    stable_codec.volume_norm = FastVolumeNorm(
        params=(orig_vn.value, orig_vn.gain_range[1]),
        sample_rate=MODEL_SR,
        energy_threshold=orig_vn.energy_threshold,
    )

    return text_tok, bicodec, stable_codec, speech_vocab

# -----------------------------------------------------------------------------
# Duration filtering / held-out speaker selection
# -----------------------------------------------------------------------------

def load_valid_ids(
    duration_file: str,
    max_duration: float,
    min_duration: float = 0.0,
    max_samples: int = -1,
) -> set[str]:
    """
    Reads (id, duration_seconds) rows, returns ids with duration in
    [min_duration, max_duration]. File order is preserved; if max_samples>0,
    only the first N qualifying ids are kept (used to cap a val set size).
    """
    valid: set[str] = set()
    with open(duration_file, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            sample_id, dur = row[0].strip(), float(row[1].strip())
            if min_duration <= dur <= max_duration:
                valid.add(sample_id)
                if max_samples > 0 and len(valid) >= max_samples:
                    break
    return valid


def load_heldout_map(heldout_file: str) -> dict[str, str]:
    """Reads (speaker_id, held_out_sample_id) rows -> dict."""
    out: dict[str, str] = {}
    with open(heldout_file, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            out[row[0].strip()] = row[1].strip()
    return out

# -----------------------------------------------------------------------------
# Reference-audio side caches (real, un-tokenized playback audio for eval)
# -----------------------------------------------------------------------------

def decode_wav(raw_bytes: bytes, target_sr: int = MODEL_SR) -> np.ndarray:
    """Decodes raw audio bytes to a mono float32 numpy waveform at target_sr."""
    wav, sr = torchaudio.load(io.BytesIO(raw_bytes))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.numpy().astype(np.float32)


def write_ref_audio_npz(out_path, wavs: list[np.ndarray], sample_rate: int = MODEL_SR) -> None:
    """wavs: list of variable-length mono float32 waveforms, in cache row order."""
    arr = np.empty(len(wavs), dtype=object)
    for i, w in enumerate(wavs):
        arr[i] = np.asarray(w, dtype=np.float32)
    np.savez(out_path, wavs=arr, sample_rate=sample_rate)

# -----------------------------------------------------------------------------
# Multi-GPU checkpointed packed-cache builder
#
# One process per GPU (torch.multiprocessing.spawn), each writing directly
# into ONE shared backing file via os.pwrite at disjoint, precomputed byte
# offsets -- safe because row-index shards are disjoint by construction
# (np.array_split over the not-yet-done row indices), so no two processes
# ever touch the same offset. A sidecar 1-byte-per-row ".done" bitmap file
# (same disjoint-pwrite pattern) tracks progress and makes the whole build
# resumable: killing the job and re-running only reprocesses undone rows.
#
# Every written row is read back and checked (non-zero, matches what was
# written) before being marked done -- guards against exactly the failure
# mode that silently corrupted ~8% of an earlier MLS build (a pathologically
# slow, since-replaced volume-normalization call stalling workers mid-shard,
# leaving zero-initialized rows marked incomplete forever). FastVolumeNorm
# above removes that root cause; this is the belt-and-suspenders check.
# -----------------------------------------------------------------------------

def _create_zeroed_file(path: Path, nbytes: int) -> None:
    with open(path, 'wb') as f:
        f.truncate(nbytes)


def _read_done_array(done_path: Path, n: int) -> np.ndarray:
    if (not done_path.exists()) or done_path.stat().st_size != n:
        return np.zeros(n, dtype=np.uint8)
    return np.fromfile(done_path, dtype=np.uint8, count=n)


def _pwrite_rows(fd: int, row_ids, rows: np.ndarray, row_nbytes: int) -> None:
    for i, r in enumerate(row_ids):
        os.pwrite(fd, rows[i].tobytes(), int(r) * row_nbytes)


def _pwrite_done(fd: int, row_ids) -> None:
    one = b'\x01'
    for r in row_ids:
        os.pwrite(fd, one, int(r))


def _verify_rows_written(fd: int, row_ids, rows_expected: np.ndarray, row_nbytes: int) -> None:
    for i, r in enumerate(row_ids):
        raw = os.pread(fd, row_nbytes, int(r) * row_nbytes)
        got = np.frombuffer(raw, dtype=np.uint32)
        if not np.array_equal(got, rows_expected[i]):
            raise RuntimeError(f'row {int(r)}: read-back mismatch right after write')
        if not got.any():
            raise RuntimeError(f'row {int(r)}: written row is all-zero (encoder produced nothing?)')


def _packed_cache_worker(
    local_rank: int,
    gpu_ids: list,
    index_chunks: list,
    dataset,
    text_field: str,
    text_tokenizer_name: str,
    speaker_model_dir: str,
    speech_model: str,
    speech_bottleneck,
    speech_bottleneck_dims,
    tmp_path: Path,
    done_path: Path,
    seq_len: int,
    text_seq_len: int,
    speaker_seq_len: int,
    speech_seq_len: int,
    pad_token_text: int,
    pad_token_speech: int,
    batch_size: int,
    num_workers: int,
    checkpoint_every: int,
    stats_dir: Path,
    split_name: str,
) -> None:
    gpu_id = gpu_ids[local_rank]
    device = torch.device('cpu') if gpu_id is None else torch.device(f'cuda:{gpu_id}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    text_tok, bicodec, stable_codec, _ = load_tokenizers(
        text_tokenizer_name=text_tokenizer_name,
        speaker_model_dir=speaker_model_dir,
        speech_model=speech_model,
        device=device,
        speech_bottleneck=speech_bottleneck,
        speech_bottleneck_dims=speech_bottleneck_dims,
    )

    indices = index_chunks[local_rank]
    subset = dataset.select(indices.tolist())
    ds_ratio = stable_codec.model.downsampling_ratio

    loader = torch.utils.data.DataLoader(
        CodecDataset(subset, text_field),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_CollateFn(ds_ratio),
        pin_memory=(device.type == 'cuda'),
    )

    arr_fd = os.open(tmp_path, os.O_RDWR)
    done_fd = os.open(done_path, os.O_RDWR)
    row_nbytes = seq_len * 4

    n_text_truncated = 0
    n_speech_truncated = 0
    stop = {'flag': False}

    def _handle_sigterm(signum, frame):
        stop['flag'] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    stats_path = Path(stats_dir) / f'rank{local_rank}.json'

    def _checkpoint():
        os.fsync(arr_fd)
        os.fsync(done_fd)
        stats_path.write_text(json.dumps({
            'n_text_truncated': n_text_truncated,
            'n_speech_truncated': n_speech_truncated,
        }))

    row_start = 0
    for batch_idx, (wav_batch, texts, true_lengths) in enumerate(loader):
        B = wav_batch.shape[0]
        row_ids = indices[row_start: row_start + B]
        row_start += B

        wav_batch = wav_batch.to(device, non_blocking=True)

        text_tokens, nt = _tokenize_text_batch(text_tok, texts, text_seq_len, pad_token_text)
        speaker_tokens = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu()
        speech_tokens, ns = _tokenize_speech_batch(
            stable_codec, wav_batch, true_lengths, speech_seq_len, pad_token_speech,
        )
        speech_tokens = speech_tokens.cpu()

        n_text_truncated += nt
        n_speech_truncated += ns

        rows = torch.cat([text_tokens, speaker_tokens, speech_tokens], dim=1).numpy().astype(np.uint32)

        _pwrite_rows(arr_fd, row_ids, rows, row_nbytes)
        _verify_rows_written(arr_fd, row_ids, rows, row_nbytes)
        _pwrite_done(done_fd, row_ids)

        if (batch_idx + 1) % checkpoint_every == 0:
            _checkpoint()
            print(f'[{split_name}] rank{local_rank}: checkpoint at batch {batch_idx + 1} '
                  f'({row_start}/{len(indices)} rows)', flush=True)

        if stop['flag']:
            _checkpoint()
            print(f'[{split_name}] rank{local_rank}: SIGTERM received, checkpointed at '
                  f'{row_start}/{len(indices)} rows, exiting', flush=True)
            os.close(arr_fd)
            os.close(done_fd)
            return

    _checkpoint()
    os.close(arr_fd)
    os.close(done_fd)


def build_packed_cache(
    *,
    dataset,
    cache_path: Path,
    meta_path: Path,
    hf_path: str,
    hf_config: Optional[str],
    hf_split: str,
    split_name: str,
    text_field: str,
    speaker_model_dir: str,
    speech_model: str,
    text_tokenizer_name: str = 'o200k_base',
    speech_bottleneck: Optional[str] = '1x46656_400bps',
    speech_bottleneck_dims: Optional[list] = None,
    text_seq_len: int,
    speaker_seq_len: int,
    speech_seq_len: int,
    gpu_ids: Optional[list] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    checkpoint_every: int = 50,
    max_duration: Optional[float] = None,
    row_durations: Optional[np.ndarray] = None,
) -> None:
    """
    Tokenizes `dataset` (already row-filtered by the caller, e.g. via
    load_valid_ids) into a packed [text|speaker|speech] uint32 cache.

    One process per entry in gpu_ids (mp.spawn), or a single in-process CPU
    worker if gpu_ids is None/empty. Resumable across re-runs via a sidecar
    .done bitmap; safe to Ctrl-C/SIGTERM and resume later.

    row_durations, if given, must be aligned 1:1 to `dataset`'s row order and
    is used only to sort the remaining-to-do rows before sharding, so each
    GPU's batches contain similarly-long clips (less padding waste). Leave
    None for sources with no natural per-row duration array.
    """
    n_samples = len(dataset)
    speech_vocab = compute_speech_vocab(speech_bottleneck, speech_bottleneck_dims)
    pad_token_text = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1
    seq_len = text_seq_len + speaker_seq_len + speech_seq_len
    row_nbytes = seq_len * 4

    cache_path = Path(cache_path)
    meta_path = Path(meta_path)
    tmp_path = cache_path.with_suffix('.tmp')
    done_path = cache_path.with_suffix('.done')
    stats_dir = cache_path.parent / f'.{cache_path.stem}_stats'

    if not (tmp_path.exists() and tmp_path.stat().st_size == n_samples * row_nbytes):
        print(f'[{split_name}] creating fresh {n_samples}x{seq_len} backing file at {tmp_path}')
        _create_zeroed_file(tmp_path, n_samples * row_nbytes)
        _create_zeroed_file(done_path, n_samples)
    stats_dir.mkdir(parents=True, exist_ok=True)

    done = _read_done_array(done_path, n_samples)
    undone = np.nonzero(done == 0)[0]

    if len(undone) == 0:
        print(f'[{split_name}] all {n_samples} rows already done, skipping tokenization')
    else:
        print(f'[{split_name}] {len(undone)}/{n_samples} rows remaining')

        if row_durations is not None:
            undone = undone[np.argsort(np.asarray(row_durations)[undone])]

        ids = list(gpu_ids) if gpu_ids else [None]
        nprocs = min(len(ids), len(undone))
        index_chunks = [c for c in np.array_split(undone, nprocs) if len(c) > 0]
        nprocs = len(index_chunks)
        active_gpu_ids = ids[:nprocs]

        worker_args = (
            active_gpu_ids, index_chunks, dataset, text_field,
            text_tokenizer_name, speaker_model_dir, speech_model,
            speech_bottleneck, speech_bottleneck_dims,
            tmp_path, done_path, seq_len,
            text_seq_len, speaker_seq_len, speech_seq_len,
            pad_token_text, pad_token_speech,
            batch_size, num_workers, checkpoint_every, stats_dir, split_name,
        )
        if nprocs == 1:
            _packed_cache_worker(0, *worker_args)
        else:
            mp.spawn(_packed_cache_worker, args=worker_args, nprocs=nprocs, join=True)

    done = _read_done_array(done_path, n_samples)
    if not done.all():
        missing = int((done == 0).sum())
        raise RuntimeError(
            f'[{split_name}] {missing}/{n_samples} rows still undone after build -- '
            f're-run the identical command to resume (checkpoint-safe).'
        )

    os.replace(tmp_path, cache_path)
    done_path.unlink(missing_ok=True)

    n_text_truncated = 0
    n_speech_truncated = 0
    for p in sorted(stats_dir.glob('rank*.json')):
        s = json.loads(p.read_text())
        n_text_truncated += s.get('n_text_truncated', 0)
        n_speech_truncated += s.get('n_speech_truncated', 0)
    shutil.rmtree(stats_dir, ignore_errors=True)

    meta = {
        'cache_format': 'packed_multimodal_blocks',
        'dtype': 'uint32',
        'hf_path': hf_path, 'hf_config': hf_config, 'hf_split': hf_split,
        'split_name': split_name,
        'n_sequences': n_samples,
        'seq_len_tokens': seq_len,
        'total_vocab': total_vocab,
        'pad_token_text': pad_token_text,
        'pad_token_speech': pad_token_speech,
        'text_seq_len': text_seq_len, 'text_offset': TEXT_OFFSET, 'text_vocab': TEXT_VOCAB,
        'n_text_truncated': n_text_truncated,
        'speaker_seq_len': speaker_seq_len, 'speaker_offset': SPEAKER_OFFSET, 'speaker_vocab': SPEAKER_VOCAB,
        'speech_seq_len': speech_seq_len, 'speech_offset': SPEECH_OFFSET, 'speech_vocab': speech_vocab,
        'n_speech_truncated': n_speech_truncated,
        'max_duration': max_duration,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{split_name}] wrote {cache_path} ({n_samples} rows, seq_len={seq_len})')

# -----------------------------------------------------------------------------
# Small (val/test-scale) ASR / TTS / continuation cache builders.
# Shared by prepare_validation.py and prepare_test.py. Single-process,
# batched, GPU if given -- meant for hundreds to a few thousand rows, not the
# multi-GPU training-set scale build_packed_cache handles. Each task selects
# its own rows independently (no shared id set across asr/tts/cont).
# -----------------------------------------------------------------------------

def _select_ids(dataset, valid_ids: set, size: Optional[int] = None) -> list:
    """Row indices in file order whose id is in valid_ids, capped at size if given."""
    ids = dataset['id']
    picked = []
    for i in range(len(dataset)):
        if ids[i] in valid_ids:
            picked.append(i)
            if size is not None and len(picked) >= size:
                break
    if size is not None and len(picked) < size:
        raise RuntimeError(f'only found {len(picked)} usable rows, requested {size}')
    return picked


def _pad_wavs(wavs: list, ds_ratio: int, device) -> Tuple[torch.Tensor, list]:
    max_len = max(w.shape[-1] for w in wavs)
    max_len = math.ceil(max_len / ds_ratio) * ds_ratio
    padded = np.stack([np.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
    batch = torch.from_numpy(padded).unsqueeze(1).to(device=device, dtype=torch.float32)
    true_lengths = [int(w.shape[-1]) for w in wavs]
    return batch, true_lengths


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _geometry_suffix(
    text_seq_len: int, speaker_seq_len: int, speech_seq_len: int, always_suffix: bool = False,
) -> str:
    # data/text_audio.py's _val_geometry_suffix/_test_geometry_suffix: applied to
    # asr/tts filenames, never to continuation's (its dataset class never
    # geometry-suffixes). validation/ caches (always_suffix=False) leave the
    # original 1000-token geometry unsuffixed, matching legacy filenames;
    # test/ caches (always_suffix=True) suffix every geometry, including 1000.
    total = text_seq_len + speaker_seq_len + speech_seq_len
    if always_suffix:
        return f'_{total}'
    return '' if total == 1000 else f'_{total}'


def build_asr_cache(
    *,
    dataset, out_dir: Path, stem: str,
    hf_path: str, hf_config: Optional[str], hf_split: str, split_name: str,
    text_field: str, valid_ids: set,
    text_tokenizer_name: str, speaker_model_dir: str, speech_model: str,
    text_seq_len: int, speaker_seq_len: int, speech_seq_len: int,
    device, batch_size: int = 32, size: Optional[int] = None,
    max_duration: Optional[float] = None, ref_audio_suffix: Optional[str] = None,
    always_suffix: bool = False,
) -> list:
    """
    text + speaker(own audio) + speech(own audio) -> packed_multimodal_blocks.
    Writes {stem}_asr{suffix}.uint32/.meta.json + {stem}_asr_{ref_audio_suffix}.ref_audio.npz.
    Returns the picked row indices.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = _select_ids(dataset, valid_ids, size)
    print(f'[{split_name}] asr: selected {len(picked)} rows')

    text_tok, bicodec, stable_codec, speech_vocab = load_tokenizers(
        text_tokenizer_name=text_tokenizer_name, speaker_model_dir=speaker_model_dir,
        speech_model=speech_model, device=device,
    )
    ds_ratio = stable_codec.model.downsampling_ratio
    pad_token_text = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1

    n = len(picked)
    rows_out = np.zeros((n, text_seq_len + speaker_seq_len + speech_seq_len), dtype=np.uint32)
    ref_wavs: list = [None] * n
    n_text_truncated = 0
    n_speech_truncated = 0

    for batch_idx_list in _chunks(list(range(n)), batch_size):
        rows = [dataset[picked[j]] for j in batch_idx_list]
        wavs = [decode_wav(r['audio']['bytes']) for r in rows]
        texts = [r[text_field] for r in rows]
        for j, w in zip(batch_idx_list, wavs):
            ref_wavs[j] = w

        wav_batch, true_lengths = _pad_wavs(wavs, ds_ratio, device)
        text_tokens, nt = _tokenize_text_batch(text_tok, texts, text_seq_len, pad_token_text)
        n_text_truncated += nt
        speaker_tokens = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu()
        speech_tokens, ns = _tokenize_speech_batch(
            stable_codec, wav_batch, true_lengths, speech_seq_len, pad_token_speech,
        )
        n_speech_truncated += ns
        speech_tokens = speech_tokens.cpu()

        batch_rows = torch.cat([text_tokens, speaker_tokens, speech_tokens], dim=1).numpy().astype(np.uint32)
        for k, j in enumerate(batch_idx_list):
            rows_out[j] = batch_rows[k]
        print(f'[{split_name}] asr: tokenized {min(batch_idx_list[-1] + 1, n)}/{n}', flush=True)

    suffix = _geometry_suffix(text_seq_len, speaker_seq_len, speech_seq_len, always_suffix)
    cache_path = out_dir / f'{stem}_asr{suffix}.uint32'
    meta_path = out_dir / f'{stem}_asr{suffix}.meta.json'
    rows_out.tofile(cache_path)
    meta = {
        'cache_format': 'packed_multimodal_blocks', 'dtype': 'uint32',
        'hf_path': hf_path, 'hf_config': hf_config, 'hf_split': hf_split, 'split_name': split_name,
        'n_sequences': n, 'seq_len_tokens': rows_out.shape[1], 'total_vocab': total_vocab,
        'pad_token_text': pad_token_text, 'pad_token_speech': pad_token_speech,
        'text_seq_len': text_seq_len, 'text_offset': TEXT_OFFSET, 'text_vocab': TEXT_VOCAB,
        'n_text_truncated': n_text_truncated,
        'speaker_seq_len': speaker_seq_len, 'speaker_offset': SPEAKER_OFFSET, 'speaker_vocab': SPEAKER_VOCAB,
        'speech_seq_len': speech_seq_len, 'speech_offset': SPEECH_OFFSET, 'speech_vocab': speech_vocab,
        'n_speech_truncated': n_speech_truncated, 'max_duration': max_duration,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    ref_suffix = f'_{ref_audio_suffix}' if ref_audio_suffix else ''
    write_ref_audio_npz(out_dir / f'{stem}_asr{ref_suffix}.ref_audio.npz', ref_wavs)
    print(f'[{split_name}] asr: wrote {cache_path} + ref_audio.npz ({n} rows)')
    return picked


def build_tts_cache(
    *,
    dataset, out_dir: Path, stem: str,
    hf_path: str, hf_config: Optional[str], hf_split: str, split_name: str,
    text_field: str, valid_ids: set, heldout_map: dict,
    text_tokenizer_name: str, speaker_model_dir: str, speech_model: str,
    text_seq_len: int, speaker_seq_len: int, speech_seq_len_for_suffix: int,
    device, batch_size: int = 32, size: Optional[int] = None,
    ref_audio_suffix: Optional[str] = None, always_suffix: bool = False,
) -> list:
    """
    text(own) + speaker(the speaker's held-out reference sample) -> packed_tts_blocks
    (no speech block). speech_seq_len_for_suffix only affects the geometry suffix in the
    filename (matches the ASR/joint row length this run trains at), not the
    actual row content, which never has a speech block.
    Writes {stem}_tts{suffix}.uint32/.meta.json + {stem}_tts_{ref_audio_suffix}.ref_audio.npz.
    Returns the picked row indices.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_ids = set(heldout_map.values())
    eligible = valid_ids - heldout_ids
    ids = dataset['id']
    speaker_ids = dataset['speaker_id']
    picked = []
    for i in range(len(dataset)):
        if ids[i] not in eligible:
            continue
        if speaker_ids[i] not in heldout_map:
            continue
        picked.append(i)
        if size is not None and len(picked) >= size:
            break
    if size is not None and len(picked) < size:
        raise RuntimeError(f'only found {len(picked)} usable tts rows, requested {size}')
    print(f'[{split_name}] tts: selected {len(picked)} rows')

    text_tok, bicodec, stable_codec, speech_vocab = load_tokenizers(
        text_tokenizer_name=text_tokenizer_name, speaker_model_dir=speaker_model_dir,
        speech_model=speech_model, device=device,
    )
    ds_ratio = stable_codec.model.downsampling_ratio
    pad_token_text = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1

    n = len(picked)
    rows_out = np.zeros((n, text_seq_len + speaker_seq_len), dtype=np.uint32)
    ref_wavs: list = [None] * n
    n_text_truncated = 0
    row_speaker_ids = [speaker_ids[picked[j]] for j in range(n)]

    for batch_idx_list in _chunks(list(range(n)), batch_size):
        rows = [dataset[picked[j]] for j in batch_idx_list]
        texts = [r[text_field] for r in rows]
        for j, r in zip(batch_idx_list, rows):
            ref_wavs[j] = decode_wav(r['audio']['bytes'])

        text_tokens, nt = _tokenize_text_batch(text_tok, texts, text_seq_len, pad_token_text)
        n_text_truncated += nt
        for k, j in enumerate(batch_idx_list):
            rows_out[j, :text_seq_len] = text_tokens[k].numpy()
        print(f'[{split_name}] tts: text-tokenized {min(batch_idx_list[-1] + 1, n)}/{n}', flush=True)

    id_to_index = {sid: i for i, sid in enumerate(ids)}
    unique_speakers = sorted(set(row_speaker_ids))
    heldout_indices = {spk: id_to_index[heldout_map[spk]] for spk in unique_speakers}

    speaker_tokens_by_spk: dict = {}
    for batch in _chunks(unique_speakers, batch_size):
        wavs = [decode_wav(dataset[heldout_indices[spk]]['audio']['bytes']) for spk in batch]
        wav_batch, _ = _pad_wavs(wavs, ds_ratio, device)
        tok = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu().numpy().astype(np.uint32)
        for k, spk in enumerate(batch):
            speaker_tokens_by_spk[spk] = tok[k]
    for j in range(n):
        rows_out[j, text_seq_len:] = speaker_tokens_by_spk[row_speaker_ids[j]]

    suffix = _geometry_suffix(text_seq_len, speaker_seq_len, speech_seq_len_for_suffix, always_suffix)
    cache_path = out_dir / f'{stem}_tts{suffix}.uint32'
    meta_path = out_dir / f'{stem}_tts{suffix}.meta.json'
    rows_out.tofile(cache_path)
    meta = {
        'cache_format': 'packed_tts_blocks', 'dtype': 'uint32',
        'hf_path': hf_path, 'hf_config': hf_config, 'hf_split': hf_split, 'split_name': split_name,
        'n_sequences': n, 'seq_len_tokens': rows_out.shape[1], 'total_vocab': total_vocab,
        'pad_token_text': pad_token_text, 'pad_token_speech': pad_token_speech,
        'text_seq_len': text_seq_len, 'text_offset': TEXT_OFFSET, 'text_vocab': TEXT_VOCAB,
        'n_text_truncated': n_text_truncated,
        'speaker_seq_len': speaker_seq_len, 'speaker_offset': SPEAKER_OFFSET, 'speaker_vocab': SPEAKER_VOCAB,
        'speech_offset': SPEECH_OFFSET, 'speech_vocab': speech_vocab,
        'n_speakers': len(unique_speakers),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    ref_suffix = f'_{ref_audio_suffix}' if ref_audio_suffix else ''
    write_ref_audio_npz(out_dir / f'{stem}_tts{ref_suffix}.ref_audio.npz', ref_wavs)
    print(f'[{split_name}] tts: wrote {cache_path} + ref_audio.npz ({n} rows, {len(unique_speakers)} speakers)')
    return picked


def build_continuation_cache(
    *,
    dataset, out_dir: Path, stem: str,
    hf_path: str, hf_config: Optional[str], hf_split: str, split_name: str,
    valid_ids: set,
    speaker_model_dir: str, speech_model: str,
    speaker_seq_len: int,
    device, batch_size: int = 32, size: Optional[int] = None,
    trim_seconds: float = 3.0, max_duration: Optional[float] = None,
) -> list:
    """
    speaker(3s prefix of own audio) + speech(same prefix) -> packed_continuation_prefix_blocks
    (no text block). valid_ids should already reflect min_duration=trim_seconds
    (via load_valid_ids(..., min_duration=trim_seconds)) so every selected row
    has a real, non-padded prefix. Never geometry-suffixed (matches
    data/text_audio.py's TextAudioContinuationDataset).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = _select_ids(dataset, valid_ids, size)
    print(f'[{split_name}] cont: selected {len(picked)} rows')

    _, bicodec, stable_codec, speech_vocab = load_tokenizers(
        text_tokenizer_name='o200k_base', speaker_model_dir=speaker_model_dir,
        speech_model=speech_model, device=device,
    )
    ds_ratio = stable_codec.model.downsampling_ratio
    pad_token_text = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1

    trim_samples = round(trim_seconds * MODEL_SR)
    trim_samples = (trim_samples // ds_ratio) * ds_ratio  # exact multiple, no pad needed
    speech_seq_len = trim_samples // ds_ratio

    n = len(picked)
    rows_out = np.zeros((n, speaker_seq_len + speech_seq_len), dtype=np.uint32)

    for batch_idx_list in _chunks(list(range(n)), batch_size):
        rows = [dataset[picked[j]] for j in batch_idx_list]
        wavs = [decode_wav(r['audio']['bytes'])[:trim_samples] for r in rows]
        wav_batch, true_lengths = _pad_wavs(wavs, ds_ratio, device)

        speaker_tokens = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu()
        speech_tokens, _ = _tokenize_speech_batch(
            stable_codec, wav_batch, true_lengths, speech_seq_len, pad_token_speech,
        )
        speech_tokens = speech_tokens.cpu()
        batch_rows = torch.cat([speaker_tokens, speech_tokens], dim=1).numpy().astype(np.uint32)
        for k, j in enumerate(batch_idx_list):
            rows_out[j] = batch_rows[k]
        print(f'[{split_name}] cont: tokenized {min(batch_idx_list[-1] + 1, n)}/{n}', flush=True)

    cache_path = out_dir / f'{stem}_cont.uint32'
    meta_path = out_dir / f'{stem}_cont.meta.json'
    rows_out.tofile(cache_path)
    meta = {
        'cache_format': 'packed_continuation_prefix_blocks', 'dtype': 'uint32',
        'hf_path': hf_path, 'hf_config': hf_config, 'hf_split': hf_split, 'split_name': split_name,
        'n_sequences': n, 'seq_len_tokens': rows_out.shape[1], 'total_vocab': total_vocab,
        'pad_token_speech': pad_token_speech,
        'speaker_seq_len': speaker_seq_len, 'speaker_offset': SPEAKER_OFFSET, 'speaker_vocab': SPEAKER_VOCAB,
        'speech_seq_len': speech_seq_len, 'speech_offset': SPEECH_OFFSET, 'speech_vocab': speech_vocab,
        'trim_seconds': trim_seconds, 'min_duration': trim_seconds, 'max_duration': max_duration,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'[{split_name}] cont: wrote {cache_path} ({n} rows)')
    return picked

# -----------------------------------------------------------------------------
# Shared preparation helpers
# -----------------------------------------------------------------------------

def extract_duration_rows(dataset) -> list[tuple[str, float]]:
    """Extract ``(sample id, seconds)`` rows in source-dataset order."""
    rows = []
    for n, sample in enumerate(dataset):
        info = torchaudio.info(io.BytesIO(sample['audio']['bytes']))
        rows.append((str(sample['id']), info.num_frames / info.sample_rate))
        if (n + 1) % 1000 == 0:
            print(f'[durations] {n + 1}/{len(dataset)}', flush=True)
    return rows


def write_csv_rows(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)


def select_heldout_samples(
    dataset, durations: dict[str, float], min_duration: float = 3.0,
) -> list[tuple[str, str]]:
    """Apply the canonical shortest-at-least-3s reference policy per speaker."""
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for speaker, sample_id in zip(dataset['speaker_id'], dataset['id']):
        by_speaker[str(speaker)].append(str(sample_id))
    result = []
    for speaker, ids in by_speaker.items():
        long_enough = [(sid, durations[sid]) for sid in ids if durations[sid] >= min_duration]
        sample_id, _ = min(long_enough, key=lambda x: x[1]) if long_enough else max(
            ((sid, durations[sid]) for sid in ids), key=lambda x: x[1],
        )
        result.append((speaker, sample_id))
    return result


def resize_packed_rows(
    rows: np.ndarray,
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    pad_token_text: int,
    pad_token_speech: int,
) -> np.ndarray:
    """Resize ``[text|speaker|speech]`` rows while preserving block boundaries."""
    if source[1] != target[1]:
        raise ValueError(f'speaker geometry cannot change: {source} -> {target}')
    out = np.full((len(rows), sum(target)), pad_token_speech, dtype=np.uint32)
    out[:, :target[0]] = pad_token_text
    text_n = min(source[0], target[0])
    speech_n = min(source[2], target[2])
    out[:, :text_n] = rows[:, :text_n]
    out[:, target[0]:target[0] + target[1]] = rows[:, source[0]:source[0] + source[1]]
    out[:, target[0] + target[1]:target[0] + target[1] + speech_n] = rows[
        :, source[0] + source[1]:source[0] + source[1] + speech_n
    ]
    return out


def resize_tts_rows(
    rows: np.ndarray, source_text_len: int, target_text_len: int, speaker_len: int = 32,
) -> np.ndarray:
    """Resize ``[text|speaker]`` rows while keeping speaker tokens in place."""
    if target_text_len > source_text_len:
        raise ValueError('TTS expansion requires an explicit text padding token')
    return np.concatenate(
        (rows[:, :target_text_len], rows[:, source_text_len:source_text_len + speaker_len]), axis=1,
    ).astype(np.uint32, copy=False)
