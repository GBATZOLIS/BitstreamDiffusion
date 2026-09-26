import math
import json
import os
import sys
from pathlib import Path

import numpy as np

import torch
from ml_collections import config_dict

from jiwer import wer, cer
from whisper_normalizer.english import EnglishTextNormalizer

from scipy.linalg import sqrtm

from typing import Any, Dict, List

from transformers import AutoFeatureExtractor, AutoTokenizer, WavLMModel, AutoModelForCausalLM

from utils.speaker_verification import ECAPA_TDNN_SMALL

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

class UTMOS:

    def __init__(self, sr=16000, use_gpu=True):
        """
        Args:
            sr (int): Sampling rate.
            use_gpu (bool): Whether to use GPU.
        """
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.predictor.eval()
        self.predictor.to(self.device)
        self.sr = sr
    
    def score(self, gen_wav):
        """
        Args:
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: UTMOS score.
        """
        gen_wav = torch.from_numpy(gen_wav).unsqueeze(0).to(self.device).float()
        score = self.predictor(gen_wav, self.sr)
        return score[0].item()
    
class TextAudioEvaluator:
    def __init__(
            self,
            whisper_model: str = 'openai/whisper-medium',
            sr=16_000,
            speaker_checkpoint: str = 'assets/text_audio/wavlm_large_finetune.pth',
            speech_extractor: str = 'microsoft/wavlm-large',
            text_model: str = 'meta-llama/Llama-3.2-1B',
            wlm_statistics_path: str = 'datasets/test_common/fsd_ref_stats_cont_common_wlm.npz',
            e2v_model: str = 'iic/emotion2vec_base',
            e2v_statistics_path: str = 'datasets/test_common/fsd_ref_stats_cont_common_e2v.npz',
            partition: str = 'clean',
            _dbg_func=None,
            speaker_ref_path: str = 'datasets/test_common/ref_speaker_embeddings_common.npz',
            continuation_speaker_ref_path: str = 'datasets/test_common/ref_speaker_embeddings_cont.npz',
            asr_num_workers: int = 4,
        ):
        self._whisper_name = whisper_model
        self._speaker_checkpoint_path = speaker_checkpoint
        self._feature_extractor_name = speech_extractor
        self._text_model_name = text_model
        self._wlm_statistics_path = wlm_statistics_path
        self._e2v_model_name = e2v_model
        self._e2v_statistics_path = e2v_statistics_path
        self._speaker_ref_path = speaker_ref_path
        self._continuation_speaker_ref_path = continuation_speaker_ref_path
        self._asr_num_workers = int(asr_num_workers)
        self._sr = sr
        self._asr = None
        self._speaker_model = None
        self._fsd_feature_extractor = None
        self._feature_extractor = None
        self._e2v_model = None
        self._text_extractor = None
        self._text_model = None
        self.reference_statistics = None
        self.e2v_reference_statistics = None
        self.partition = partition
        self._utmos = None
        self._dbg_func = _dbg_func
        self._text_normalizer = None

        self._ref_speaker_embeddings = {}

    def _dbg(self, msg: str):
        if self._dbg_func:
            self._dbg_func(msg)
        else:
            print(msg)

    def _ensure_asr(self, device):
        if self._asr is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            self._dbg(f'Loading {self._whisper_name} on {device}')
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._whisper_name, torch_dtype=torch.float16
            ).to(device)
            _proc = AutoProcessor.from_pretrained(self._whisper_name)
            self._asr = pipeline(
                'automatic-speech-recognition',
                model=_model,
                tokenizer = _proc.tokenizer,
                feature_extractor=_proc.feature_extractor,
                torch_dtype=torch.float16,
                device=device,
                chunk_length_s=30,
            )
        return self._asr


    def transcribe(self, wavs, device) -> list[str]:
        asr = self._ensure_asr(device)
        inputs, valid_idx = [], []
        for i, wav in enumerate(wavs):
            if wav is None:
                continue
            arr = np.asarray(wav, dtype=np.float32).flatten()
            if arr.ndim > 1:
                arr = arr[0]
            inputs.append({'array': arr, 'sampling_rate': self._sr})
            valid_idx.append(i)
        
        results = [""]*len(wavs)
        if inputs:
            try:
                preds = asr(inputs, batch_size=min(64, len(inputs)), num_workers=self._asr_num_workers)
                for idx, pred in zip(valid_idx, preds):
                    results[idx] = pred['text'].strip()
            except Exception as e:
                self._dbg(f'ASR batch failed: {e}')

        return results
    
    def _ensure_utmos(self):
        if self._utmos is None:
            self._dbg('Loading UTMOS')
            self._utmos = UTMOS(sr=self._sr, use_gpu=torch.cuda.is_available())

        return self._utmos
    
    def utmos_score(self, wavs) -> float:
        model = self._ensure_utmos()
        
        scores = []
        for wav in wavs:
            if wav is None:
                continue
            arr = np.asarray(wav, dtype=np.float32).flatten()
            if arr.ndim > 1:
                arr = arr[0]
            try:
                scores.append(float(model.score(arr)))
            except Exception as e:
                self._dbg(f'UTMOS score failed: {e}')

        return float(np.mean(scores)) if scores else float('nan')
    
    def _ensure_text_normalizer(self) -> EnglishTextNormalizer:
        if self._text_normalizer is None:
            self._text_normalizer = EnglishTextNormalizer()
        return self._text_normalizer

    def normalize_text(self, text: str) -> str:
        normalizer = self._ensure_text_normalizer()
        return normalizer(text or '')

    def _normalized_pairs(self, refs, hyps) -> tuple[list, list]:
        """Normalizes ref/hyp pairs and drops any whose normalized reference
        comes out empty -- jiwer can't score them and they carry no signal
        (same rule scripts/evaluation_drivers/recompute_wer_cer.py uses)."""
        norm_refs, norm_hyps = [], []
        for r, h in zip(refs, hyps):
            nr, nh = self.normalize_text(r), self.normalize_text(h)
            if not nr.strip():
                continue
            norm_refs.append(nr)
            norm_hyps.append(nh)
        return norm_refs, norm_hyps

    def word_error_rate(self, refs, hyps) -> float:
        norm_refs, norm_hyps = self._normalized_pairs(refs, hyps)
        if not norm_refs:
            return float('nan')
        return float(wer(norm_refs, norm_hyps))

    def character_error_rate(self, refs, hyps) -> float:
        norm_refs, norm_hyps = self._normalized_pairs(refs, hyps)
        if not norm_refs:
            return float('nan')
        return float(cer(norm_refs, norm_hyps))

    def _ensure_speaker_model(self, device):
        if self._speaker_model is None:
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            state_dict = torch.load(self._speaker_checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            model.eval()
            self._speaker_model = model.to(device)
        return self._speaker_model

    def _ensure_ref_speaker_embeddings(self, *, continuation: bool = False) -> torch.Tensor:
        path = self._continuation_speaker_ref_path if continuation else self._speaker_ref_path
        key = 'embeddings' if continuation else f'{self.partition}_embeddings'
        cache_key = (str(path), key)
        if cache_key not in self._ref_speaker_embeddings:
            with np.load(path) as data:
                if key not in data:
                    raise KeyError(f'{path}: missing speaker-embedding array {key!r}')
                embeddings = torch.from_numpy(data[key]).float()
            self._ref_speaker_embeddings[cache_key] = torch.nn.functional.normalize(
                embeddings, dim=-1
            )
        return self._ref_speaker_embeddings[cache_key]

    def spksim(
        self, gen_wavs, device, *, continuation: bool = False, reference_indices=None
    ) -> float:
        model = self._ensure_speaker_model(device)
        ref_embeddings = self._ensure_ref_speaker_embeddings(continuation=continuation)
        if reference_indices is not None:
            indices = torch.as_tensor(reference_indices, dtype=torch.long)
            if indices.ndim != 1 or len(indices) != len(gen_wavs):
                raise ValueError(
                    'SpkSim reference_indices must contain one row index per waveform'
                )
            if len(indices) and (indices.min() < 0 or indices.max() >= len(ref_embeddings)):
                raise IndexError('SpkSim reference index is outside the embedding archive')
            ref_embeddings = ref_embeddings[indices]

        if len(gen_wavs) != len(ref_embeddings):
            ref_kind = 'continuation' if continuation else self.partition
            raise ValueError(
                f'SpkSim row mismatch for {ref_kind}: {len(gen_wavs)} generated waveforms '
                f'but {len(ref_embeddings)} reference embeddings'
            )

        gen_embeddings = []
        with torch.no_grad():
            for i, wav in enumerate(gen_wavs):
                if wav is None:
                    raise ValueError(f'SpkSim generated waveform {i} is empty')
                arr = np.asarray(wav, dtype=np.float32).flatten()
                if arr.size == 0:
                    raise ValueError(f'SpkSim generated waveform {i} is empty')
                x = torch.from_numpy(arr).unsqueeze(0).to(device)
                emb = model(x)
                gen_embeddings.append(emb.squeeze(0).cpu())
        gen_embeddings = torch.nn.functional.normalize(torch.stack(gen_embeddings, dim=0), dim=-1)

        return (ref_embeddings * gen_embeddings).sum(dim=-1).mean().item()

    def _ensure_feature_extractor(self, device):
        if self._fsd_feature_extractor is None:
            self._fsd_feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._feature_extractor_name
            )
        if self._feature_extractor is None:
            self._feature_extractor = WavLMModel.from_pretrained(
                self._feature_extractor_name
            ).to(device)
            self._feature_extractor.eval()
        return self._fsd_feature_extractor, self._feature_extractor
    
    def _ensure_refernce_statistics(self):
        if self.reference_statistics is None:
            stats = np.load(self._wlm_statistics_path)
            self.reference_statistics = (stats['mean'], stats['cov'])
        return self.reference_statistics

    @staticmethod
    def _frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov) -> float:
        mean_diff = ref_mean - gen_mean
        tr, _ = sqrtm(ref_cov @ gen_cov, disp=False)
        if np.iscomplexobj(tr):
            tr = tr.real
        tr = np.trace(ref_cov + gen_cov - 2 * tr)
        return mean_diff @ mean_diff + tr

    def fsd(
        self, wavs, device, chunk_size: int = 32, *, save_dir=None, sample_keys=None,
        sample_key_name: str = 'sample_keys',
    ) -> float:
        extractor, model = self._ensure_feature_extractor(device)
        ref_mean, ref_cov = self._ensure_refernce_statistics()
        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in wavs]

        all_embs = []
        shards = []
        for start in range(0, len(wavs), chunk_size):
            stop = min(start + chunk_size, len(wavs))
            chunk = wavs[start:stop]
            inputs = extractor(chunk, return_tensors='pt', padding=True, sampling_rate=self._sr)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states[6]
            frames_per_sample = hidden.shape[1]
            embedding = hidden.reshape(-1, hidden.shape[-1]).cpu().float().numpy()
            offsets = np.arange(len(chunk) + 1, dtype=np.int64) * frames_per_sample
            all_embs.append(embedding)
            shards.append((start, stop, embedding, offsets))
        embedding = np.concatenate(all_embs, axis=0)
        if save_dir is not None:
            _save_fsd_features(
                Path(save_dir) / 'wlm', extractor='FSD-wlm',
                model=self._feature_extractor_name, layer=6,
                sample_rate=self._sr, batch_size=chunk_size,
                num_samples=len(wavs), shards=shards, sample_keys=sample_keys,
                sample_key_name=sample_key_name,
            )
        gen_mean = np.mean(embedding, axis=0)
        gen_cov = np.cov(embedding, rowvar=False)
        return self._frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)

    def _ensure_e2v_model(self, device):
        if self._e2v_model is None:
            from funasr import AutoModel
            am = AutoModel(model=self._e2v_model_name, device=str(device))
            self._e2v_model = am.model.to(device).eval()
        return self._e2v_model

    def _ensure_e2v_reference_statistics(self):
        if self.e2v_reference_statistics is None:
            stats = np.load(self._e2v_statistics_path)
            self.e2v_reference_statistics = (stats['mean'], stats['cov'])
        return self.e2v_reference_statistics

    @torch.no_grad()
    def _e2v_frame_embeddings(self, wavs, model, device, batch_size: int = 32) -> np.ndarray:
        out = []
        n = len(wavs)
        for start in range(0, n, batch_size):
            chunk = wavs[start:start + batch_size]
            lengths = [len(w) for w in chunk]
            max_len = max(lengths)
            batch = torch.zeros(len(chunk), max_len, dtype=torch.float32, device=device)
            pad_mask = torch.ones(len(chunk), max_len, dtype=torch.bool, device=device)  # True = pad
            for i, w in enumerate(chunk):
                t = torch.as_tensor(w, dtype=torch.float32)
                batch[i, :len(t)] = t.to(device)
                pad_mask[i, :len(t)] = False

            feats = model.extract_features(batch, padding_mask=pad_mask)
            hidden = feats['x']                 # (B, T', C)
            frame_mask = feats['padding_mask']  # (B, T') True = pad, or None if nothing was padded

            for i in range(hidden.size(0)):
                valid = hidden[i] if frame_mask is None else hidden[i][~frame_mask[i]]
                out.append(valid.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def fsd_e2v(
        self, wavs, device, chunk_size: int = 32, *, save_dir=None, sample_keys=None,
        sample_key_name: str = 'sample_keys',
    ) -> float:
        model = self._ensure_e2v_model(device)
        ref_mean, ref_cov = self._ensure_e2v_reference_statistics()
        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in wavs]

        all_embeddings = []
        shards = []
        for start in range(0, len(wavs), chunk_size):
            stop = min(start + chunk_size, len(wavs))
            chunk = wavs[start:stop]
            lengths = [len(wav) for wav in chunk]
            batch = torch.zeros(len(chunk), max(lengths), dtype=torch.float32, device=device)
            pad_mask = torch.ones(len(chunk), max(lengths), dtype=torch.bool, device=device)
            for index, wav in enumerate(chunk):
                batch[index, :len(wav)] = torch.as_tensor(wav, dtype=torch.float32, device=device)
                pad_mask[index, :len(wav)] = False
            with torch.no_grad():
                features = model.extract_features(batch, padding_mask=pad_mask)
            hidden = features['x']
            frame_mask = features['padding_mask']
            per_sample = [
                hidden[index] if frame_mask is None else hidden[index][~frame_mask[index]]
                for index in range(hidden.shape[0])
            ]
            offsets = np.concatenate(([0], np.cumsum([len(value) for value in per_sample])))
            embedding = torch.cat(per_sample, dim=0).cpu().float().numpy()
            all_embeddings.append(embedding)
            shards.append((start, stop, embedding, offsets))
        embedding = np.concatenate(all_embeddings, axis=0)
        if save_dir is not None:
            _save_fsd_features(
                Path(save_dir) / 'e2v', extractor='FSD-e2v',
                model=self._e2v_model_name, layer='final',
                sample_rate=self._sr, batch_size=chunk_size,
                num_samples=len(wavs), shards=shards, sample_keys=sample_keys,
                sample_key_name=sample_key_name,
            )
        gen_mean = np.mean(embedding, axis=0)
        gen_cov = np.cov(embedding, rowvar=False)
        return self._frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)

    def _ensure_text_model(self, device):
        if self._text_extractor is None:
            self._text_extractor = AutoTokenizer.from_pretrained(self._text_model_name)
            self._text_extractor.pad_token = self._text_extractor.eos_token
        if self._text_model is None:
            self._text_model = AutoModelForCausalLM.from_pretrained(
                self._text_model_name, torch_dtype=torch.float16
            ).to(device).eval()
        return self._text_extractor, self._text_model

    def gen_ppl(self, hyps, device, chunk_size: int = 32) -> float:
        extractor, model = self._ensure_text_model(device)

        all_perplexities: List[float] = []
        for i in range(0, len(hyps), chunk_size):
            chunk = hyps[i : i + chunk_size]
            inputs = extractor(chunk, return_tensors='pt', padding=True).to(model.device)
            labels = inputs['input_ids'].clone()
            labels[inputs['attention_mask'] == 0] = -100

            with torch.no_grad():
                logits = model(**inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            mask = shift_labels != -100
            mean_loss_per_sample = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
            all_perplexities.extend(torch.exp(mean_loss_per_sample).tolist())

        finite = [p for p in all_perplexities if math.isfinite(p)]
        n_dropped = len(all_perplexities) - len(finite)
        if n_dropped:
            self._dbg(
                f'gen_ppl: {n_dropped}/{len(all_perplexities)} samples had non-finite perplexity'
            )
        if not finite:
            return float('nan')
        return sum(finite) / len(finite)

    def evaluate_task_brief(
        self, task:str, gen_texts:List[str], ref_texts: List[str], gen_wavs, device
    ) -> Dict[str, float]:
        metrics = {}
        transcriptions = None

        if task == 'joint':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['Cross-Modal WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['Cross-Modal CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
        elif task == 'tts':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['ASR-WER'] = self.word_error_rate(ref_texts, transcriptions)
            metrics['ASR-CER'] = self.character_error_rate(ref_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
        elif task == 'stt':
            metrics['WER'] = self.word_error_rate(ref_texts, gen_texts)
            metrics['CER'] = self.character_error_rate(ref_texts, gen_texts)
        elif task == 'cont':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['GenPPL-speech'] = self.gen_ppl(transcriptions, device)

        return metrics, transcriptions

    def evaluate_task_extensive(
        self, task: str, gen_texts: List[str], ref_texts: List[str], gen_wavs, device,
        *, fsd_output_dir=None, fsd_sample_keys=None,
        fsd_sample_key_name: str = 'sample_keys', fsd_chunk_size: int = 32,
    ) -> Dict[str, float]:
        metrics = {}
        transcriptions = None

        if task == 'joint':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['Cross-Modal WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['Cross-Modal CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['GenPPL-text'] = self.gen_ppl(gen_texts, device)
            metrics['GenPPL-speech'] = self.gen_ppl(transcriptions, device)
        elif task == 'tts':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['ASR-WER'] = self.word_error_rate(ref_texts, transcriptions)
            metrics['ASR-CER'] = self.character_error_rate(ref_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['SpkSim'] = self.spksim(gen_wavs, device)
        elif task == 'stt':
            metrics[f'{self.partition}-WER'] = self.word_error_rate(ref_texts, gen_texts)
            metrics[f'{self.partition}-CER'] = self.character_error_rate(ref_texts, gen_texts)
        elif task == 'cont_taste':
            trimmed_wavs = [_trim_prefix_seconds(w, CONT_TRIM_SECONDS, self._sr) for w in gen_wavs]
            transcriptions = self.transcribe(trimmed_wavs, device)
            original_transcriptions = self.transcribe(gen_wavs, device)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['SpkSim'] = self.spksim(trimmed_wavs, device, continuation=True)
            metrics['GenPPL-speech'] = self.gen_ppl(original_transcriptions, device)
            metrics['FSD-wlm'] = self.fsd(
                gen_wavs, device, fsd_chunk_size, save_dir=fsd_output_dir,
                sample_keys=fsd_sample_keys, sample_key_name=fsd_sample_key_name,
            )
            metrics['FSD-e2v'] = self.fsd_e2v(
                gen_wavs, device, fsd_chunk_size, save_dir=fsd_output_dir,
                sample_keys=fsd_sample_keys, sample_key_name=fsd_sample_key_name,
            )

        return metrics, transcriptions
 
    def evaluate_stt(self, ref_texts: list[str], gen_wavs=None, transcriptions: list[str] = None, device='cpu'):
        if not transcriptions:
            transcriptions = self.transcribe(gen_wavs, device)
        wer = self.word_error_rate(ref_texts, transcriptions)
        cer = self.character_error_rate(ref_texts, transcriptions)
        return wer, cer

SALMON_JUDGES = [
    "nvidia/speakerverification_en_titanet_large", 
    "ALM/hubert-large-audioset",
    "ALM/wav2vec2-large-audioset"
]

SALMON_JUDGE_PER_TASK = {
    'sentiment_consistency': "nvidia/speakerverification_en_titanet_large",
    'speaker_consistency': "nvidia/speakerverification_en_titanet_large",
    'gender_consistency': "nvidia/speakerverification_en_titanet_large",
    'bg_domain_consistency': "ALM/hubert-large-audioset",
    'bg_all_consistency': "ALM/hubert-large-audioset",
    'rir_consistency': "ALM/wav2vec2-large-audioset"
}

class SALMONEvaluator:
    def __init__(
            self, 
            sr: int = 16_000, 
            max_len: float = 5.0,
            ref_dir: str = 'datasets/test_common/salmon',
            _dbg_func=None
        ):
        self._sr = sr
        self._max_len = max_len
        self._dbg_func = _dbg_func
        self.judges = None
        self.embeddings_per_task = None
        self.ref_dir = ref_dir

    def _dbg(self, msg: str):
        if self._dbg_func:
            self._dbg_func(msg)
        else:
            print(msg)

    def _ensure_judges(self, device):
        from utils.judge_models import JudgeModel
        if not self.judges:
            self.judges = {}
            for judge_name in SALMON_JUDGES:
                self.judges[judge_name] = JudgeModel(judge_name, self._sr, device, self._max_len)
        return self.judges
    
    def _ensure_embeddings(self, device):
        if not self.embeddings_per_task:
            self.embeddings_per_task = {}
            for task in SALMON_JUDGE_PER_TASK.keys():
                embeddings_path = f'{self.ref_dir}/judge_embeddings_{task}.npz'
                embeddings = np.load(embeddings_path)
                self.embeddings_per_task[task] = {
                    'pos': torch.from_numpy(embeddings['positive']).to(device),
                    'neg': torch.from_numpy(embeddings['negative']).to(device)
                }
        return self.embeddings_per_task
    
    def evaluate_SALMON(self, gen_wavs_per_task: dict, device):
        metrics = {}

        judges = self._ensure_judges(device)
        ref_embeddings = self._ensure_embeddings(device)

        for task, judge in SALMON_JUDGE_PER_TASK.items():
            gen_embeddings = judges[judge].embed_batch(gen_wavs_per_task[task])
            metrics[task] = judges[judge].score(
                gen_embeddings, ref_embeddings[task]['pos'], ref_embeddings[task]['neg']
            )

        return metrics

# -----------------------------------------------------------------------------
# Joint-generation speaker evaluation
# -----------------------------------------------------------------------------

class JointSpeakerEvaluator:
    """Evaluate sampled speaker codes against their waveform realization.

    This consolidates the CAM++ extraction, BiCodec re-tokenization, and the
    token/latent diversity analyses previously kept in standalone scripts and
    ``joint.ipynb``. Models are loaded lazily and reused across sampler tags.
    """

    SPEAKER_OFFSET = 200_019
    SPEAKER_VOCAB_SIZE = 4_096
    SPEAKER_SEQ_LEN = 32

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 3.0,
        campp_checkpoint: str | None = None,
        bicodec_model_dir: str = 'Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec',
        campp_batch_size: int = 128,
        bicodec_batch_size: int = 8,
        pad_alignment: int = 640,
        shuffle_trials: int = 100,
        shuffle_seed: int = 42,
        _dbg_func=None,
    ):
        if sample_rate != 16_000:
            raise ValueError('CAM++ and BiCodec speaker evaluation requires 16 kHz audio.')
        for name, value in (
            ('campp_batch_size', campp_batch_size),
            ('bicodec_batch_size', bicodec_batch_size),
            ('pad_alignment', pad_alignment),
            ('shuffle_trials', shuffle_trials),
        ):
            if value < 1:
                raise ValueError(f'{name} must be positive.')
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(sample_rate * segment_seconds))
        self.campp_checkpoint = campp_checkpoint
        self.bicodec_model_dir = str(bicodec_model_dir)
        self.campp_batch_size = int(campp_batch_size)
        self.bicodec_batch_size = int(bicodec_batch_size)
        self.pad_alignment = int(pad_alignment)
        self.shuffle_trials = int(shuffle_trials)
        self.shuffle_seed = int(shuffle_seed)
        self._dbg_func = _dbg_func
        self._campp_session = None
        self._campp_input_name = None
        self._bicodec = None

    def _dbg(self, message: str) -> None:
        if self._dbg_func:
            self._dbg_func(message)
        else:
            print(message, flush=True)

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump(value, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_npz(path: Path, **arrays) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as handle:
                np.savez(handle, **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_torch_save(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_campp(self):
        if self._campp_session is None:
            import onnxruntime
            from huggingface_hub import hf_hub_download

            checkpoint = self.campp_checkpoint
            if checkpoint is None:
                checkpoint = hf_hub_download(
                    repo_id='MediaTek-Research/Llama-1B-TASTE-V0',
                    filename='cosyvoice/speaker_embed.onnx',
                )
            self._dbg(f'Loading CAM++ from {checkpoint}')
            self._campp_session = onnxruntime.InferenceSession(
                str(checkpoint), providers=['CPUExecutionProvider']
            )
            self._campp_input_name = self._campp_session.get_inputs()[0].name
        return self._campp_session, self._campp_input_name

    def _ensure_bicodec(self, device):
        if self._bicodec is None:
            spark_tts_root = Path('Spark-TTS').resolve()
            if str(spark_tts_root) not in sys.path:
                sys.path.insert(0, str(spark_tts_root))
            from sparktts.models.bicodec import BiCodec

            self._dbg(f'Loading BiCodec from {self.bicodec_model_dir}')
            self._bicodec = BiCodec.load_from_checkpoint(
                model_dir=self.bicodec_model_dir
            ).to(device).eval()
        return self._bicodec

    @torch.inference_mode()
    def _extract_campp(self, wavs, wav_names, output_path: Path):
        import torchaudio.compliance.kaldi as kaldi

        session, input_name = self._ensure_campp()
        durations = np.asarray(
            [len(np.asarray(wav).reshape(-1)) / self.sample_rate for wav in wavs],
            dtype=np.float32,
        )
        valid_indices = np.flatnonzero(durations >= self.segment_seconds).astype(np.int64)
        if not len(valid_indices):
            raise ValueError(
                f'No joint waveform is at least {self.segment_seconds:g} seconds long.'
            )

        embedding_batches = []
        for start in range(0, len(valid_indices), self.campp_batch_size):
            indices = valid_indices[start:start + self.campp_batch_size]
            features = []
            for index in indices:
                waveform = torch.as_tensor(
                    np.asarray(wavs[int(index)], dtype=np.float32).reshape(-1)[
                        :self.segment_samples
                    ]
                ).unsqueeze(0)
                feature = kaldi.fbank(
                    waveform,
                    num_mel_bins=80,
                    sample_frequency=self.sample_rate,
                    dither=0.0,
                )
                feature = feature - feature.mean(dim=0, keepdim=True)
                features.append(feature.cpu().numpy().astype(np.float32))
            embeddings = session.run(
                None, {input_name: np.stack(features, axis=0)}
            )[0].reshape(len(features), -1)
            embeddings = embeddings / np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
            )
            embedding_batches.append(embeddings.astype(np.float32))
            self._dbg(f'CAM++ embedded {min(start + len(indices), len(valid_indices))}/{len(valid_indices)}')

        embeddings = np.concatenate(embedding_batches, axis=0)
        valid_paths = np.asarray([str(wav_names[index]) for index in valid_indices])
        self._atomic_npz(
            output_path,
            embeddings=embeddings,
            indices=valid_indices,
            paths=valid_paths,
            durations=durations,
            segment_seconds=np.float32(self.segment_seconds),
            sample_rate=np.int32(self.sample_rate),
        )
        return embeddings, valid_indices, durations

    @torch.inference_mode()
    def _extract_bicodec_tokens(self, wavs, device) -> torch.Tensor:
        bicodec = self._ensure_bicodec(device)
        result = torch.empty(
            (len(wavs), self.SPEAKER_SEQ_LEN), dtype=torch.long
        )
        for start in range(0, len(wavs), self.bicodec_batch_size):
            stop = min(start + self.bicodec_batch_size, len(wavs))
            chunk = [np.asarray(wav, dtype=np.float32).reshape(-1) for wav in wavs[start:stop]]
            if any(not len(wav) or not np.isfinite(wav).all() for wav in chunk):
                raise ValueError(f'Invalid joint waveform in rows {start}:{stop}.')
            maximum = max(len(wav) for wav in chunk)
            padded_length = math.ceil(maximum / self.pad_alignment) * self.pad_alignment
            padded = np.stack([
                np.pad(wav, (0, padded_length - len(wav))) for wav in chunk
            ])
            waveform_batch = torch.from_numpy(padded).unsqueeze(1).to(
                device=device, dtype=torch.float32
            )
            mel = bicodec.mel_transformer(waveform_batch).squeeze(1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == 'cuda',
            ):
                tokens = bicodec.speaker_encoder.tokenize(
                    mel.transpose(1, 2)
                ).squeeze(1)
            expected = (stop - start, self.SPEAKER_SEQ_LEN)
            if tuple(tokens.shape) != expected:
                raise ValueError(f'BiCodec returned {tuple(tokens.shape)}, expected {expected}.')
            tokens = tokens.to(device='cpu', dtype=torch.long)
            if tokens.min() < 0 or tokens.max() >= self.SPEAKER_VOCAB_SIZE:
                raise ValueError('BiCodec returned a token outside its speaker vocabulary.')
            result[start:stop] = tokens
            self._dbg(f'BiCodec re-tokenized {stop}/{len(wavs)}')
        return result

    def _save_bicodec_archive(
        self, path: Path, tokens: torch.Tensor, wav_names, task_dir: Path,
        sampler_tag: str, header: dict,
    ) -> None:
        archive = {
            'tokens': tokens,
            'global_token_ids': tokens + self.SPEAKER_OFFSET,
            'token_space': 'bicodec_speaker_code',
            'global_token_space': 'global_tokenizer_id',
            'speaker_offset': self.SPEAKER_OFFSET,
            'speaker_vocab_size': self.SPEAKER_VOCAB_SIZE,
            'speaker_seq_len': self.SPEAKER_SEQ_LEN,
            'num_samples': len(tokens),
            'sample_indices': torch.arange(len(tokens), dtype=torch.long),
            'wav_files': list(wav_names),
            'source_samples_json': str(task_dir / 'samples.json'),
            'sampler_tag': sampler_tag,
            'task': 'joint',
            'sampler': header.get('sampler'),
            'checkpoint': header.get('checkpoint'),
            'sample_rate': self.sample_rate,
            'speaker_model_dir': self.bicodec_model_dir,
            'batch_size': self.bicodec_batch_size,
            'pad_alignment': self.pad_alignment,
        }
        self._atomic_torch_save(path, archive)
        written = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(written["tokens"], tokens):
            raise IOError(f"Post-write verification failed for {path}.")

    @staticmethod
    def _unit_latent_metrics(lhs, rhs, valid_positions):
        import torch.nn.functional as F

        mask = valid_positions.bool().unsqueeze(-1)
        lhs = F.normalize(
            torch.where(mask, torch.nan_to_num(lhs), torch.zeros_like(lhs)),
            p=2, dim=-1, eps=1e-12,
        )
        rhs = F.normalize(
            torch.where(mask, torch.nan_to_num(rhs), torch.zeros_like(rhs)),
            p=2, dim=-1, eps=1e-12,
        )
        squared_error = (lhs - rhs).square()
        position_rmse = squared_error.mean(dim=-1).sqrt().masked_fill(
            ~valid_positions, torch.nan
        )
        expanded_mask = mask.expand_as(squared_error)
        component_count = expanded_mask.sum(dim=(1, 2))
        sequence_rmse = (
            squared_error.masked_fill(~expanded_mask, 0.0).sum(dim=(1, 2))
            / component_count.clamp_min(1)
        ).sqrt().masked_fill(component_count == 0, torch.nan)
        position_cosine = (lhs * rhs).sum(dim=-1).masked_fill(
            ~valid_positions, torch.nan
        )
        position_count = valid_positions.sum(dim=-1)
        sequence_cosine = (
            position_cosine.nan_to_num(0.0).sum(dim=-1)
            / position_count.clamp_min(1)
        ).masked_fill(position_count == 0, torch.nan)
        return position_rmse, sequence_rmse, position_cosine, sequence_cosine

    @torch.inference_mode()
    def _tokens_to_latents(self, tokens: torch.Tensor, device):
        quantizer = self._ensure_bicodec(device).speaker_encoder.quantizer
        tokens = tokens.detach().to(device='cpu', dtype=torch.long)
        valid = (tokens >= 0) & (tokens < int(quantizer.codebook_size))
        flat_tokens = tokens.reshape(-1)
        flat_valid = valid.reshape(-1)
        indices = flat_tokens[flat_valid].to(device).reshape(-1, 1, 1)
        valid_latents = quantizer.get_output_from_indices(indices).squeeze(1).cpu().float()
        latents = torch.full(
            (flat_tokens.numel(), valid_latents.shape[-1]), float('nan')
        )
        latents[flat_valid] = valid_latents
        return latents.reshape(*tokens.shape, -1), valid

    @staticmethod
    def _derangements(num_samples: int, trials: int, seed: int) -> torch.Tensor:
        if num_samples < 2:
            raise ValueError('Joint speaker metrics require at least two samples.')
        generator = torch.Generator().manual_seed(seed)
        identity = torch.arange(num_samples)
        permutations = []
        for _ in range(trials):
            while True:
                candidate = torch.randperm(num_samples, generator=generator)
                if not candidate.eq(identity).any():
                    permutations.append(candidate)
                    break
        return torch.stack(permutations)

    @staticmethod
    def _token_diversity(tokens: torch.Tensor, vocab_size: int) -> dict:
        valid = (tokens >= 0) & (tokens < vocab_size)
        entropies = torch.full((tokens.shape[1],), torch.nan, dtype=torch.float64)
        valid_per_position = valid.sum(dim=0)
        for position in range(tokens.shape[1]):
            values = tokens[valid[:, position], position]
            if values.numel():
                counts = torch.bincount(values, minlength=vocab_size).double()
                probabilities = counts[counts > 0] / values.numel()
                entropies[position] = -(probabilities * probabilities.log2()).sum()

        distances = []
        for left in range(tokens.shape[0] - 1):
            jointly_valid = valid[left] & valid[left + 1:]
            comparable = jointly_valid.sum(dim=1)
            usable = comparable > 0
            mismatches = (
                tokens[left].ne(tokens[left + 1:]) & jointly_valid
            ).sum(dim=1)
            distances.append(mismatches[usable].double() / comparable[usable])
        pairwise = torch.cat(distances)

        fully_valid = valid.all(dim=1)
        valid_sequences = tokens[fully_valid]
        _, counts = torch.unique(valid_sequences, dim=0, return_counts=True)
        duplicate_counts = counts[counts > 1]
        return {
            'num_samples': int(tokens.shape[0]),
            'num_positions': int(tokens.shape[1]),
            'invalid_token_positions_excluded': int((~valid).sum()),
            'mean_per_position_entropy_bits': float(torch.nanmean(entropies)),
            'mean_pairwise_normalized_hamming_distance': float(pairwise.mean()),
            'number_of_unordered_pairs': int(pairwise.numel()),
            'fully_valid_sequences': int(fully_valid.sum()),
            'excluded_invalid_sequences': int((~fully_valid).sum()),
            'unique_sequences': int(counts.numel()),
            'duplicate_sequence_groups': int(duplicate_counts.numel()),
            'samples_in_repeated_groups': int(duplicate_counts.sum()),
            'repeated_copies_beyond_first': int((duplicate_counts - 1).sum()),
            'identical_unordered_pairs': int(
                (duplicate_counts * (duplicate_counts - 1) // 2).sum()
            ),
            'maximum_sequence_multiplicity': int(counts.max()) if counts.numel() else 0,
            'per_position_entropy_bits': entropies.tolist(),
            'valid_samples_per_position': valid_per_position.tolist(),
        }

    @staticmethod
    def _latent_diversity(latents: torch.Tensor, valid: torch.Tensor) -> tuple[dict, dict]:
        import torch.nn.functional as F

        retained = valid.all(dim=1) & torch.isfinite(latents).all(dim=(1, 2))
        values = latents[retained].double()
        if len(values) < 2:
            raise ValueError('At least two fully valid sampled speaker sequences are required.')
        position_unit = F.normalize(values, p=2, dim=-1, eps=1e-12)
        sequence_unit = F.normalize(position_unit.flatten(start_dim=1), p=2, dim=1)
        pair_indices = torch.triu_indices(len(sequence_unit), len(sequence_unit), offset=1)
        similarities = (sequence_unit @ sequence_unit.T)[
            pair_indices[0], pair_indices[1]
        ].clamp(-1.0, 1.0)
        distances = 1.0 - similarities
        euclidean = (2.0 * distances).clamp_min(0.0).sqrt()
        rmse = euclidean / sequence_unit.shape[1] ** 0.5
        centered = sequence_unit - sequence_unit.mean(dim=0)
        component_variance = centered.square().mean(dim=0)
        gram = centered @ centered.T / len(sequence_unit)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        proportions = eigenvalues / eigenvalues.sum()
        nonzero = proportions > 0
        effective_rank = torch.exp(
            -(proportions[nonzero] * proportions[nonzero].log()).sum()
        )
        flattened = {
            'source_samples': int(latents.shape[0]),
            'valid_sequences': int(retained.sum()),
            'excluded_invalid_sequences': int((~retained).sum()),
            'positions_per_sequence': int(latents.shape[1]),
            'latent_dimension_per_position': int(latents.shape[2]),
            'flattened_sequence_dimension': int(sequence_unit.shape[1]),
            'unordered_pairs': int(similarities.numel()),
            'mean_pairwise_unit_vector_rmse': float(rmse.mean()),
            'mean_pairwise_cosine_similarity': float(similarities.mean()),
            'mean_pairwise_cosine_distance': float(distances.mean()),
            'mean_pairwise_euclidean_distance': float(euclidean.mean()),
            'total_component_variance': float(component_variance.sum()),
            'mean_component_variance': float(component_variance.mean()),
            'covariance_effective_rank': float(effective_rank),
        }

        aligned_unit = F.normalize(values.float(), p=2, dim=-1, eps=1e-12)
        pair_count = len(aligned_unit) * (len(aligned_unit) - 1) // 2
        vector_sum = aligned_unit.sum(dim=0)
        cosine_by_position = (
            vector_sum.square().sum(dim=-1)
            - aligned_unit.square().sum(dim=-1).sum(dim=0)
        ) / (2.0 * pair_count)
        position_aligned = {
            'num_sequences': int(len(position_unit)),
            'sequence_length': int(position_unit.shape[1]),
            'latent_dim': int(position_unit.shape[2]),
            'number_of_unordered_pairs': int(pair_count),
            'mean_pairwise_cosine_similarity': float(cosine_by_position.mean()),
            'mean_pairwise_cosine_distance': float((1.0 - cosine_by_position).mean()),
            'mean_cosine_similarity_by_position': cosine_by_position.tolist(),
            'mean_cosine_distance_by_position': (1.0 - cosine_by_position).tolist(),
        }
        return flattened, position_aligned

    @staticmethod
    def _campp_diversity(embeddings: np.ndarray, source_count: int) -> dict:
        import torch.nn.functional as F

        values = torch.from_numpy(np.asarray(embeddings)).double()
        valid = torch.isfinite(values).all(dim=1) & (values.norm(dim=1) > 0)
        unit = F.normalize(values[valid], p=2, dim=1, eps=1e-12)
        if len(unit) < 2:
            raise ValueError('At least two valid CAM++ embeddings are required.')
        indices = torch.triu_indices(len(unit), len(unit), offset=1)
        similarities = (unit @ unit.T)[indices[0], indices[1]].clamp(-1.0, 1.0)
        distances = 1.0 - similarities
        euclidean = (2.0 * distances).clamp_min(0.0).sqrt()
        rmse = euclidean / unit.shape[1] ** 0.5
        centered = unit - unit.mean(dim=0)
        component_variance = centered.square().mean(dim=0)
        covariance = centered.T @ centered / len(unit)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        proportions = eigenvalues / eigenvalues.sum()
        nonzero = proportions > 0
        effective_rank = torch.exp(
            -(proportions[nonzero] * proportions[nonzero].log()).sum()
        )
        return {
            'source_samples': int(source_count),
            'excluded_by_extractor': int(source_count - len(values)),
            'archive_embeddings': int(len(values)),
            'valid_embeddings': int(valid.sum()),
            'invalid_embeddings': int((~valid).sum()),
            'embedding_dimension': int(values.shape[1]),
            'unordered_pairs': int(similarities.numel()),
            'mean_pairwise_unit_vector_rmse': float(rmse.mean()),
            'mean_pairwise_cosine_similarity': float(similarities.mean()),
            'mean_pairwise_cosine_distance': float(distances.mean()),
            'mean_pairwise_euclidean_distance': float(euclidean.mean()),
            'total_component_variance': float(component_variance.sum()),
            'mean_component_variance': float(component_variance.mean()),
            'covariance_effective_rank': float(effective_rank),
        }

    @staticmethod
    def _flatten_scalars(report: dict) -> dict[str, float]:
        flattened = {}
        for section, values in report.items():
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flattened[f'Speaker/{section}/{name}'] = float(value)
        return flattened

    @torch.inference_mode()
    def evaluate(
        self,
        *,
        task_dir: Path,
        samples: list[dict],
        wavs,
        device,
        sampler_tag: str,
        manifest_header: dict,
    ) -> dict[str, float]:
        task_dir = Path(task_dir)
        if len(samples) != len(wavs) or not samples:
            raise ValueError('Joint samples and waveforms must be non-empty and row-aligned.')
        if any(wav is None for wav in wavs):
            raise ValueError('Every joint sample must have a generated waveform.')
        wav_names = [sample.get('gen_wav') for sample in samples]
        if any(not isinstance(name, str) or not name for name in wav_names):
            raise ValueError('Every joint samples.json row must contain gen_wav.')
        if len(set(wav_names)) != len(wav_names):
            raise ValueError('Joint samples.json contains duplicate gen_wav entries.')

        sampled_path = task_dir / 'generated_tokens.pt'
        if not sampled_path.is_file():
            raise FileNotFoundError(
                f'{sampled_path} is required for --speaker; generate with '
                'save_generated_tokens enabled.'
            )
        sampled_archive = torch.load(sampled_path, map_location='cpu', weights_only=False)
        sampled_all = sampled_archive['tokens'].long()
        if len(sampled_all) != len(samples) or int(sampled_archive['num_samples']) != len(samples):
            raise ValueError('generated_tokens.pt is not row-aligned with samples.json.')
        speaker_start, speaker_stop = sampled_archive['segments']['speaker']
        sampled_tokens = (
            sampled_all[:, int(speaker_start):int(speaker_stop)]
            - int(sampled_archive['speaker_offset'])
        ).long()
        vocab_size = int(sampled_archive['speaker_vocab_size'])
        if sampled_tokens.shape != (len(samples), self.SPEAKER_SEQ_LEN):
            raise ValueError(
                f'Expected sampled speaker tokens {(len(samples), self.SPEAKER_SEQ_LEN)}, '
                f'found {tuple(sampled_tokens.shape)}.'
            )
        if int(sampled_archive["speaker_offset"]) != self.SPEAKER_OFFSET:
            raise ValueError(
                f"Expected speaker offset {self.SPEAKER_OFFSET}, found "
                f"{sampled_archive['speaker_offset']}."
            )
        if vocab_size != self.SPEAKER_VOCAB_SIZE:
            raise ValueError(
                f'Expected BiCodec speaker vocabulary {self.SPEAKER_VOCAB_SIZE}, found {vocab_size}.'
            )

        campp_paths = [str(task_dir / name) for name in wav_names]
        campp_embeddings, campp_indices, durations = self._extract_campp(
            wavs, campp_paths, task_dir / 'sv_embeddings.npz'
        )
        realized_tokens = self._extract_bicodec_tokens(wavs, device)
        self._save_bicodec_archive(
            task_dir / 'bicodec_global_tokens.pt', realized_tokens, wav_names,
            task_dir, sampler_tag, manifest_header,
        )

        sampled_valid = (sampled_tokens >= 0) & (sampled_tokens < vocab_size)
        realized_valid = (realized_tokens >= 0) & (realized_tokens < vocab_size)
        matches = sampled_tokens.eq(realized_tokens)
        exact = matches.all(dim=1)
        token_agreement = {
            'mean_per_sample_token_agreement': float(matches.float().mean(dim=1).mean()),
            'mean_per_position_agreement': float(matches.float().mean(dim=0).mean()),
            'exact_sequence_matches': int(exact.sum()),
            'num_samples': int(len(exact)),
            'exact_sequence_agreement': float(exact.float().mean()),
        }

        permutations = self._derangements(
            len(samples), self.shuffle_trials, self.shuffle_seed
        )
        shuffled_token_means = []
        shuffled_exact_counts = []
        for permutation in permutations:
            shuffled_matches = sampled_tokens.eq(realized_tokens[permutation])
            shuffled_token_means.append(shuffled_matches.float().mean())
            shuffled_exact_counts.append(shuffled_matches.all(dim=1).sum())
        shuffled_token_means = torch.stack(shuffled_token_means)
        shuffled_exact_counts = torch.stack(shuffled_exact_counts).float()
        shuffled_agreement = {
            'num_shuffle_trials': self.shuffle_trials,
            'mean_per_sample_token_agreement': float(shuffled_token_means.mean()),
            'mean_per_position_agreement': float(shuffled_token_means.mean()),
            'token_agreement_std_across_trials': float(
                shuffled_token_means.std(unbiased=self.shuffle_trials > 1)
            ),
            'mean_exact_sequence_matches_per_trial': float(shuffled_exact_counts.mean()),
            'exact_sequence_agreement': float(
                shuffled_exact_counts.mean() / len(samples)
            ),
            'fixed_points_across_all_trials': int(
                permutations.eq(torch.arange(len(samples))).sum()
            ),
        }

        sampled_latents, sampled_latent_valid = self._tokens_to_latents(
            sampled_tokens, device
        )
        realized_latents, realized_latent_valid = self._tokens_to_latents(
            realized_tokens, device
        )
        paired_valid = sampled_latent_valid & realized_latent_valid
        paired = self._unit_latent_metrics(
            sampled_latents, realized_latents, paired_valid
        )
        shuffled_position_rmse = []
        shuffled_sequence_rmse = []
        shuffled_position_cosine = []
        shuffled_sequence_cosine = []
        shuffled_valid_counts = []
        for permutation in permutations:
            valid = sampled_latent_valid & realized_latent_valid[permutation]
            values = self._unit_latent_metrics(
                sampled_latents, realized_latents[permutation], valid
            )
            shuffled_position_rmse.append(torch.nanmean(values[0]))
            shuffled_sequence_rmse.append(torch.nanmean(values[1]))
            shuffled_position_cosine.append(torch.nanmean(values[2]))
            shuffled_sequence_cosine.append(torch.nanmean(values[3]))
            shuffled_valid_counts.append(valid.sum())
        shuffled_position_rmse = torch.stack(shuffled_position_rmse)
        shuffled_sequence_rmse = torch.stack(shuffled_sequence_rmse)
        shuffled_position_cosine = torch.stack(shuffled_position_cosine)
        shuffled_sequence_cosine = torch.stack(shuffled_sequence_cosine)
        shuffled_valid_counts = torch.stack(shuffled_valid_counts).float()
        paired_position_rmse = float(torch.nanmean(paired[0]))
        paired_sequence_rmse = float(torch.nanmean(paired[1]))
        shuffled_position_rmse_mean = float(shuffled_position_rmse.mean())
        shuffled_sequence_rmse_mean = float(shuffled_sequence_rmse.mean())
        latent_similarity = {
            'paired_position_unit_vector_rmse': paired_position_rmse,
            'shuffled_position_unit_vector_rmse': shuffled_position_rmse_mean,
            'shuffled_position_unit_vector_rmse_std': float(
                shuffled_position_rmse.std(unbiased=self.shuffle_trials > 1)
            ),
            'position_rmse_relative_reduction': (
                1.0 - paired_position_rmse / shuffled_position_rmse_mean
                if shuffled_position_rmse_mean else float('nan')
            ),
            'paired_sequence_unit_vector_rmse': paired_sequence_rmse,
            'shuffled_sequence_unit_vector_rmse': shuffled_sequence_rmse_mean,
            'shuffled_sequence_unit_vector_rmse_std': float(
                shuffled_sequence_rmse.std(unbiased=self.shuffle_trials > 1)
            ),
            'sequence_rmse_relative_reduction': (
                1.0 - paired_sequence_rmse / shuffled_sequence_rmse_mean
                if shuffled_sequence_rmse_mean else float('nan')
            ),
            'paired_position_cosine_similarity': float(torch.nanmean(paired[2])),
            'shuffled_position_cosine_similarity': float(shuffled_position_cosine.mean()),
            'shuffled_position_cosine_similarity_std': float(
                shuffled_position_cosine.std(unbiased=self.shuffle_trials > 1)
            ),
            'paired_sequence_mean_cosine_similarity': float(torch.nanmean(paired[3])),
            'shuffled_sequence_mean_cosine_similarity': float(shuffled_sequence_cosine.mean()),
            'shuffled_sequence_mean_cosine_similarity_std': float(
                shuffled_sequence_cosine.std(unbiased=self.shuffle_trials > 1)
            ),
            'valid_paired_positions': int(paired_valid.sum()),
            'skipped_paired_positions': int((~paired_valid).sum()),
            'mean_valid_shuffled_positions': float(shuffled_valid_counts.mean()),
            'shuffle_trials': self.shuffle_trials,
        }

        flattened_diversity, aligned_diversity = self._latent_diversity(
            sampled_latents, sampled_latent_valid
        )
        report = {
            'diagnostics': {
                'num_samples': len(samples),
                'sampled_invalid_tokens': int((~sampled_valid).sum()),
                'sampled_rows_with_invalid_tokens': int((~sampled_valid).any(dim=1).sum()),
                'realized_invalid_tokens': int((~realized_valid).sum()),
                'campp_retained_embeddings': int(len(campp_indices)),
                'campp_excluded_short_samples': int(len(samples) - len(campp_indices)),
            },
            'token_agreement': token_agreement,
            'shuffled_token_agreement': shuffled_agreement,
            'latent_similarity': latent_similarity,
            'sampled_token_diversity': self._token_diversity(
                sampled_tokens, vocab_size
            ),
            'sampled_bicodec_latent_diversity': flattened_diversity,
            'sampled_bicodec_position_aligned_diversity': aligned_diversity,
            'generated_campp_diversity': self._campp_diversity(
                campp_embeddings, len(durations)
            ),
            'artifacts': {
                'sampled_tokens': str(sampled_path),
                'realized_tokens': str(task_dir / 'bicodec_global_tokens.pt'),
                'campp_embeddings': str(task_dir / 'sv_embeddings.npz'),
            },
        }
        report_path = task_dir / 'speaker_metrics.json'
        self._atomic_json(report_path, report)
        self._dbg(f'Wrote joint speaker report to {report_path}')
        return self._flatten_scalars(report)
