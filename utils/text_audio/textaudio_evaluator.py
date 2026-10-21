import math
from pathlib import Path

import numpy as np

import torch

from jiwer import wer, cer
from whisper_normalizer.english import EnglishTextNormalizer

from scipy.linalg import sqrtm

from typing import Dict, List

from transformers import AutoFeatureExtractor, AutoTokenizer, WavLMModel, AutoModelForCausalLM

from utils.text_audio.speaker_verification import ECAPA_TDNN_SMALL
from utils.text_audio.utils import CONT_TRIM_SECONDS, _save_fsd_features, _trim_prefix_seconds
from utils.text_audio.utmos import UTMOS


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

    def _ensure_ref_speaker_embeddings(
        self, *, continuation: bool = False, reference_path: str | None = None,
    ) -> torch.Tensor:
        path = reference_path or (
            self._continuation_speaker_ref_path if continuation else self._speaker_ref_path
        )
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
        self, gen_wavs, device, *, continuation: bool = False, reference_indices=None,
        reference_path: str | None = None,
    ) -> float:
        model = self._ensure_speaker_model(device)
        ref_embeddings = self._ensure_ref_speaker_embeddings(
            continuation=continuation, reference_path=reference_path,
        )
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
        speaker_ref_path: str | None = None,
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
            metrics['SpkSim'] = self.spksim(
                gen_wavs, device, reference_path=speaker_ref_path,
            )
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
