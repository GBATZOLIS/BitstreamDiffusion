"""Prepare validation and common test assets for text-audio evaluation.

By default both sets are rebuilt.  Use ``--target validation`` or
``--target test_common`` to run either half independently.  Both 632- and
1000-token geometries are produced from a single 1000-token encoding pass.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, concatenate_datasets, load_dataset

from cache_utils import (
    extract_duration_rows, resize_packed_rows, resize_tts_rows,
    select_heldout_samples, write_csv_rows,
    MODEL_SR, SPEAKER_OFFSET, SPEAKER_VOCAB, SPEECH_OFFSET,
    TEXT_OFFSET, TEXT_VOCAB, _pad_wavs, _tokenize_speaker_batch,
    _tokenize_speech_batch, _tokenize_text_batch, compute_speech_vocab,
    decode_wav, load_tokenizers, write_ref_audio_npz,
)

LIBRITTS_PATH = 'mythicinfinity/libritts'
LIBRITTS_CONFIG = 'all'
TEST_PATH = 'mythicinfinity/librispeech-pc-44khz-opus'
CONT_SECONDS = 3.0
FSD_MODEL = 'microsoft/wavlm-large'
FSD_LAYER = 6
E2V_MODEL = 'iic/emotion2vec_base'
SALMON_TASKS = {
    'sentiment_consistency': 'nvidia/speakerverification_en_titanet_large',
    'speaker_consistency': 'nvidia/speakerverification_en_titanet_large',
    'gender_consistency': 'nvidia/speakerverification_en_titanet_large',
    'bg_domain_consistency': 'ALM/hubert-large-audioset',
    'bg_all_consistency': 'ALM/hubert-large-audioset',
    'rir_consistency': 'ALM/wav2vec2-large-audioset',
}


def _duration_rows(dataset):
    return extract_duration_rows(dataset)


def _write_csv(path: Path, rows):
    write_csv_rows(path, rows)


def _select_heldout(dataset, durations):
    return select_heldout_samples(dataset, durations, CONT_SECONDS)


def _base_meta(*, hf_path, hf_config, hf_split, split_name, n, fmt, text_len=None,
               speaker_len=32, speech_len=None, ids=None, max_duration=32.0) -> dict:
    speech_vocab = compute_speech_vocab('1x46656_400bps', None)
    pad_text = SPEECH_OFFSET + speech_vocab
    pad_speech = pad_text + 1
    meta = {
        'cache_format': fmt, 'dtype': 'uint32', 'hf_path': hf_path,
        'hf_config': hf_config, 'hf_split': hf_split, 'split_name': split_name,
        'n_sequences': n, 'total_vocab': pad_speech + 1,
        'pad_token_text': pad_text, 'pad_token_speech': pad_speech,
        'text_offset': TEXT_OFFSET, 'text_vocab': TEXT_VOCAB,
        'speaker_seq_len': speaker_len, 'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB, 'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab, 'max_duration': max_duration,
    }
    if text_len is not None:
        meta.update(text_seq_len=text_len, n_text_truncated=0)
    if speech_len is not None:
        meta.update(speech_seq_len=speech_len, n_speech_truncated=0)
    meta['seq_len_tokens'] = (text_len or 0) + speaker_len + (speech_len or 0)
    if ids is not None:
        meta['ids'] = list(ids)
    return meta


def _write_cache(data_path: Path, meta_path: Path, rows: np.ndarray, meta: dict) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    rows.astype(np.uint32, copy=False).tofile(data_path)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'[cache] wrote {data_path} {rows.shape}')


def _joint_geometry(rows, source=(168, 32, 800), target=(100, 32, 500),
                    pad_text=250771, pad_speech=250772):
    return resize_packed_rows(rows, source, target, pad_text, pad_speech)


def _tts_geometry(rows, source_text=168, target_text=100):
    return resize_tts_rows(rows, source_text, target_text)


class EvaluationEncoder:
    """One shared model load for all caches in one requested evaluation set."""
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.text_tok, self.bicodec, self.codec, self.speech_vocab = load_tokenizers(
            args.text_tokenizer, args.speaker_model_dir, args.speech_model, self.device,
        )
        self.ds_ratio = self.codec.model.downsampling_ratio
        self.pad_text = SPEECH_OFFSET + self.speech_vocab
        self.pad_speech = self.pad_text + 1

    def _rows(self, dataset, ids: list[str]):
        index = {str(sid): i for i, sid in enumerate(dataset['id'])}
        missing = [sid for sid in ids if sid not in index]
        if missing:
            raise RuntimeError(f'{len(missing)} selected ids absent from dataset: {missing[:5]}')
        return [index[sid] for sid in ids]

    @torch.inference_mode()
    def joint(self, dataset, ids: list[str], text_field: str) -> tuple[np.ndarray, list[np.ndarray]]:
        picked = self._rows(dataset, ids)
        out = np.zeros((len(ids), 1000), dtype=np.uint32)
        refs = [None] * len(ids)
        for start in range(0, len(ids), self.batch_size):
            stop = min(start + self.batch_size, len(ids))
            records = [dataset[picked[i]] for i in range(start, stop)]
            wavs = [decode_wav(row['audio']['bytes']) for row in records]
            refs[start:stop] = wavs
            batch, lengths = _pad_wavs(wavs, self.ds_ratio, self.device)
            text, _ = _tokenize_text_batch(self.text_tok, [r[text_field] for r in records], 168, self.pad_text)
            speaker = _tokenize_speaker_batch(self.bicodec, batch, 32).cpu()
            speech, _ = _tokenize_speech_batch(self.codec, batch, lengths, 800, self.pad_speech)
            out[start:stop] = torch.cat((text, speaker, speech.cpu()), 1).numpy()
            print(f'[joint] {stop}/{len(ids)}', flush=True)
        return out, refs

    @torch.inference_mode()
    def tts(self, dataset, ids: list[str], text_field: str, heldout: dict[str, str],
            own_speaker: bool = False, batch_size: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        picked = self._rows(dataset, ids)
        out = np.zeros((len(ids), 200), dtype=np.uint32)
        refs = [None] * len(ids)
        bs = batch_size or self.batch_size
        speakers = [str(dataset[i]['speaker_id']) for i in picked]
        for start in range(0, len(ids), bs):
            stop = min(start + bs, len(ids))
            records = [dataset[picked[i]] for i in range(start, stop)]
            texts, _ = _tokenize_text_batch(self.text_tok, [r[text_field] for r in records], 168, self.pad_text)
            out[start:stop, :168] = texts.numpy()
            refs[start:stop] = [decode_wav(r['audio']['bytes']) for r in records]
        if own_speaker:
            token_ids = ids
            destination = list(range(len(ids)))
        else:
            unique = list(dict.fromkeys(speakers))
            token_ids = [heldout[speaker] for speaker in unique]
            speaker_row = {speaker: i for i, speaker in enumerate(unique)}
            destination = [speaker_row[speaker] for speaker in speakers]
        token_picked = self._rows(dataset, token_ids)
        token_rows = np.zeros((len(token_ids), 32), dtype=np.uint32)
        for start in range(0, len(token_ids), bs):
            stop = min(start + bs, len(token_ids))
            wavs = [decode_wav(dataset[token_picked[i]]['audio']['bytes']) for i in range(start, stop)]
            batch, _ = _pad_wavs(wavs, self.ds_ratio, self.device)
            token_rows[start:stop] = _tokenize_speaker_batch(self.bicodec, batch, 32).cpu().numpy()
        out[:, 168:] = token_rows[destination]
        return out, refs

    @torch.inference_mode()
    def continuation(self, dataset, ids: list[str]) -> np.ndarray:
        picked = self._rows(dataset, ids)
        trim = round(CONT_SECONDS * MODEL_SR)
        trim = trim // self.ds_ratio * self.ds_ratio
        speech_len = trim // self.ds_ratio
        out = np.zeros((len(ids), 32 + speech_len), dtype=np.uint32)
        for start in range(0, len(ids), self.batch_size):
            stop = min(start + self.batch_size, len(ids))
            wavs = [decode_wav(dataset[picked[i]]['audio']['bytes'])[:trim] for i in range(start, stop)]
            batch, lengths = _pad_wavs(wavs, self.ds_ratio, self.device)
            speaker = _tokenize_speaker_batch(self.bicodec, batch, 32).cpu()
            speech, _ = _tokenize_speech_batch(self.codec, batch, lengths, speech_len, self.pad_speech)
            out[start:stop] = torch.cat((speaker, speech.cpu()), 1).numpy()
            print(f'[continuation] {stop}/{len(ids)}', flush=True)
        return out


def _write_joint_pair(root, stem, rows1000, meta1000, refs, ref_name):
    rows632 = _joint_geometry(rows1000, pad_text=meta1000['pad_token_text'], pad_speech=meta1000['pad_token_speech'])
    meta632 = dict(meta1000)
    meta632.update(seq_len_tokens=632, text_seq_len=100, speech_seq_len=500,
                   derived_from_seq_len_tokens=1000)
    _write_cache(root / f'{stem}_632.uint32', root / f'{stem}_632.meta.json', rows632, meta632)
    _write_cache(root / f'{stem}_1000.uint32', root / f'{stem}_1000.meta.json', rows1000, meta1000)
    write_ref_audio_npz(root / ref_name, refs)


def _write_tts_pair(root, stem, rows1000, meta1000, data_suffix='', meta_suffix=''):
    rows632 = _tts_geometry(rows1000)
    meta632 = dict(meta1000)
    meta632.update(seq_len_tokens=132, text_seq_len=100, derived_from_seq_len_tokens=200)
    _write_cache(root / f'{stem}_632{data_suffix}.uint32', root / f'{stem}_632{meta_suffix}.meta.json', rows632, meta632)
    _write_cache(root / f'{stem}_1000{data_suffix}.uint32', root / f'{stem}_1000{meta_suffix}.meta.json', rows1000, meta1000)


def _duration_map(rows):
    return {sample_id: float(seconds) for sample_id, seconds in rows}


def prepare_validation(args) -> None:
    root = Path(args.validation_out)
    assets = Path(args.assets_dir)
    print('[validation] loading LibriTTS dev.clean and dev.other')
    all_ds = load_dataset(LIBRITTS_PATH, LIBRITTS_CONFIG)
    clean = all_ds['dev.clean'].cast_column('audio', Audio(decode=False))
    other = all_ds['dev.other'].cast_column('audio', Audio(decode=False))
    durations_rows = _duration_rows(concatenate_datasets([clean, other]))
    _write_csv(assets / 'durations/val.csv', durations_rows)
    durations = _duration_map(durations_rows)
    heldout_rows = _select_heldout(clean, durations)
    _write_csv(assets / 'heldout/val.csv', heldout_rows)
    heldout = dict(heldout_rows)

    dataset_ids = [str(x) for x in clean['id']]
    first = [sid for sid in dataset_ids if durations[sid] <= args.max_duration][:args.validation_size]
    asr_ids = [sid for sid in first if durations[sid] <= 20.0]
    asr_ids.extend(sid for sid in dataset_ids if durations[sid] <= 20.0 and sid not in set(asr_ids))
    asr_ids = asr_ids[:args.validation_size]
    tts_ids = [sid for sid in dataset_ids if durations[sid] <= args.max_duration and sid not in set(heldout.values())]
    cont_ids = [sid for sid in dataset_ids if CONT_SECONDS <= durations[sid] <= args.max_duration]

    enc = EvaluationEncoder(args)
    joint, refs = enc.joint(clean, asr_ids, 'text_normalized')
    joint_meta = _base_meta(hf_path=LIBRITTS_PATH, hf_config=LIBRITTS_CONFIG,
        hf_split='dev.clean', split_name='val', n=len(asr_ids), fmt='packed_multimodal_blocks',
        text_len=168, speech_len=800, ids=asr_ids)
    _write_joint_pair(root, 'cache_val_asr', joint, joint_meta, refs, 'cache_val_asr_632.ref_audio.npz')
    shutil.copyfile(root / 'cache_val_asr_1000.uint32', root / 'cache_val_asr.uint32')
    shutil.copyfile(root / 'cache_val_asr_1000.meta.json', root / 'cache_val_asr.meta.json')

    tts, tts_refs = enc.tts(clean, tts_ids, 'text_normalized', heldout)
    tts_meta = _base_meta(hf_path=LIBRITTS_PATH, hf_config='clean', hf_split='dev.clean',
        split_name='libritts_dev_clean', n=len(tts_ids), fmt='packed_tts_blocks',
        text_len=168, ids=tts_ids)
    tts_meta['n_speakers'] = len(set(str(clean[enc._rows(clean, tts_ids)[i]]['speaker_id']) for i in range(len(tts_ids))))
    _write_tts_pair(root, 'cache_val_tts', tts, tts_meta)
    write_ref_audio_npz(root / 'cache_val_tts_632.ref_audio.npz', tts_refs)
    shutil.copyfile(root / 'cache_val_tts_1000.uint32', root / 'cache_val_tts.uint32')
    shutil.copyfile(root / 'cache_val_tts_1000.meta.json', root / 'cache_val_tts.meta.json')

    cont = enc.continuation(clean, cont_ids)
    cont_meta = _base_meta(hf_path=LIBRITTS_PATH, hf_config='clean', hf_split='dev.clean',
        split_name='libritts_dev_clean', n=len(cont_ids), fmt='packed_continuation_prefix_blocks',
        speaker_len=32, speech_len=75, ids=cont_ids)
    cont_meta.update(trim_seconds=CONT_SECONDS, min_duration=CONT_SECONDS)
    _write_cache(root / 'cache_val_cont.uint32', root / 'cache_val_cont.meta.json', cont, cont_meta)


def _embed_speakers(dataset, ids, output, checkpoint, device, prefix_seconds=None):
    from utils.speaker_verification import ECAPA_TDNN_SMALL
    index = {str(sid): i for i, sid in enumerate(dataset['id'])}
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)
    model.to(device).eval()
    embeddings = []
    with torch.inference_mode():
        for n, sid in enumerate(ids):
            wav = decode_wav(dataset[index[sid]]['audio']['bytes'])
            if prefix_seconds is not None:
                wav = wav[:round(prefix_seconds * MODEL_SR)]
            embeddings.append(model(torch.from_numpy(wav).unsqueeze(0).to(device)).squeeze(0).cpu())
            if (n + 1) % 200 == 0:
                print(f'[speaker embeddings] {n + 1}/{len(ids)}', flush=True)
    values = F.normalize(torch.stack(embeddings), dim=-1).numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    if prefix_seconds is None:
        np.savez(output, clean_ids=np.asarray(ids), clean_embeddings=values)
    else:
        np.savez(output, ids=np.asarray(ids), embeddings=values,
                 reference_seconds=np.float32(prefix_seconds), sample_rate=np.int32(MODEL_SR))


def _wavlm_embeddings(wavs, device, batch_size):
    from transformers import AutoFeatureExtractor, WavLMModel
    extractor = AutoFeatureExtractor.from_pretrained(FSD_MODEL)
    model = WavLMModel.from_pretrained(FSD_MODEL).to(device).eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(wavs), batch_size):
            chunk = wavs[start:start + batch_size]
            inputs = extractor(chunk, sampling_rate=MODEL_SR, return_tensors='pt', padding=True)
            hidden = model(**{k: v.to(device) for k, v in inputs.items()},
                           output_hidden_states=True).hidden_states[FSD_LAYER]
            pieces.append(hidden.reshape(-1, hidden.shape[-1]).cpu().float().numpy())
    return np.concatenate(pieces)


def _e2v_embeddings(wavs, device, batch_size, model_name):
    from funasr import AutoModel
    model = AutoModel(model=model_name, device=str(device)).model.to(device).eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(wavs), batch_size):
            chunk = wavs[start:start + batch_size]
            lengths = [len(x) for x in chunk]
            batch = torch.zeros(len(chunk), max(lengths), device=device)
            mask = torch.ones_like(batch, dtype=torch.bool)
            for i, wav in enumerate(chunk):
                batch[i, :len(wav)] = torch.from_numpy(wav).to(device)
                mask[i, :len(wav)] = False
            result = model.extract_features(batch, padding_mask=mask)
            hidden, frame_mask = result['x'], result['padding_mask']
            pieces.extend((hidden[i] if frame_mask is None else hidden[i][~frame_mask[i]]).cpu().float().numpy()
                          for i in range(len(chunk)))
    return np.concatenate(pieces)


def _statistics(x):
    return np.mean(x, axis=0).astype(np.float32), np.cov(x, rowvar=False)


def _prepare_test_auxiliary(args, clean, other, tts_ids, cont_ids, other_ids, cont_refs):
    root = Path(args.test_out)
    device = torch.device(args.device)
    if not args.skip_embeddings:
        _embed_speakers(clean, tts_ids, root / 'ref_speaker_embeddings_common.npz',
                        args.speaker_checkpoint, device)
        _embed_speakers(clean, cont_ids, root / 'ref_speaker_embeddings_cont.npz',
                        args.speaker_checkpoint, device, CONT_SECONDS)
    if not args.skip_transcriptions:
        from utils.textaudio_utils import TextAudioEvaluator
        evaluator = TextAudioEvaluator(whisper_model=args.whisper_model, sr=MODEL_SR, _dbg_func=print)
        transcripts = evaluator.transcribe([x[:round(CONT_SECONDS * MODEL_SR)] for x in cont_refs], device)
        rows = [{'id': sid, 'transcription': text} for sid, text in zip(cont_ids, transcripts)]
        (root / 'cache_test_cont.ref_transcriptions.json').write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not args.skip_statistics:
        clean_index = {str(x): i for i, x in enumerate(clean['id'])}
        other_index = {str(x): i for i, x in enumerate(other['id'])}
        clean_wavs = [decode_wav(clean[clean_index[x]]['audio']['bytes']) for x in cont_ids]
        other_wavs = [decode_wav(other[other_index[x]]['audio']['bytes']) for x in other_ids]
        clean_wlm = _wavlm_embeddings(clean_wavs, device, args.statistics_batch_size)
        clean_mean, clean_cov = _statistics(clean_wlm)
        np.savez(root / 'fsd_ref_stats_cont_common_wlm.npz', mean=clean_mean, cov=clean_cov, layer=FSD_LAYER)
        e2v = _e2v_embeddings(clean_wavs, device, args.statistics_batch_size, args.e2v_model)
        e2v_mean, e2v_cov = _statistics(e2v)
        np.savez(root / 'fsd_ref_stats_cont_common_e2v.npz', mean=e2v_mean, cov=e2v_cov, layer='final')
        other_mean, other_cov = _statistics(_wavlm_embeddings(other_wavs, device, args.statistics_batch_size))
        stats_path = Path(args.assets_dir) / 'fsd_statistics.npz'
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, clean_mean=clean_mean, clean_cov=clean_cov,
                 other_mean=other_mean, other_cov=other_cov, layer=FSD_LAYER)
        print(f'[statistics] wrote {stats_path}')


def prepare_test_common(args) -> None:
    root = Path(args.test_out)
    assets = Path(args.assets_dir)
    clean = load_dataset(TEST_PATH, 'clean', split='test').cast_column('audio', Audio(decode=False))
    other = load_dataset(TEST_PATH, 'other', split='test').cast_column('audio', Audio(decode=False))
    clean_rows, other_rows = _duration_rows(clean), _duration_rows(other)
    _write_csv(assets / 'durations/test_clean.csv', clean_rows)
    _write_csv(assets / 'durations/test_other.csv', other_rows)
    clean_durations, other_durations = _duration_map(clean_rows), _duration_map(other_rows)
    heldout_rows = _select_heldout(clean, clean_durations)
    _write_csv(assets / 'heldout/test_clean.csv', heldout_rows)
    _write_csv(root / 'heldout.csv', heldout_rows)
    heldout = dict(heldout_rows)
    heldout_ids = set(heldout.values())

    # Geometry 632 carries 20 seconds of speech.  Selecting this common set up
    # front makes both geometries row-identical and reproduces the current IDs.
    clean_ids = sorted(sid for sid, seconds in clean_rows if seconds <= 20.0 and sid not in heldout_ids)
    cont_ids = [sid for sid in clean_ids if clean_durations[sid] >= CONT_SECONDS]
    other_ids = [str(sid) for sid in other['id'] if other_durations[str(sid)] <= 20.0]

    enc = EvaluationEncoder(args)
    clean_joint, clean_refs = enc.joint(clean, clean_ids, 'text')
    clean_meta = _base_meta(hf_path=TEST_PATH, hf_config='clean', hf_split='test',
        split_name='librispeech_test_clean', n=len(clean_ids), fmt='packed_multimodal_blocks',
        text_len=168, speech_len=800, ids=clean_ids)
    _write_joint_pair(root, 'cache_test_clean_asr', clean_joint, clean_meta, clean_refs,
                      'cache_test_clean_asr.ref_audio.npz')

    other_joint, other_refs = enc.joint(other, other_ids, 'text')
    other_meta = _base_meta(hf_path=TEST_PATH, hf_config='other', hf_split='test',
        split_name='test_other', n=len(other_ids), fmt='packed_multimodal_blocks',
        text_len=168, speech_len=800, ids=other_ids)
    _write_joint_pair(root, 'cache_test_other_asr', other_joint, other_meta, other_refs,
                      'cache_test_other_asr.ref_audio.npz')

    tts, tts_refs = enc.tts(clean, clean_ids, 'text', heldout)
    tts_meta = _base_meta(hf_path=TEST_PATH, hf_config='clean', hf_split='test',
        split_name='librispeech_test_clean', n=len(clean_ids), fmt='packed_tts_blocks',
        text_len=168, ids=clean_ids)
    tts_meta.update(n_speakers=len(set(str(clean[i]['speaker_id']) for i in enc._rows(clean, clean_ids))),
                    speaker_token_source='heldout_reference_utterance')
    _write_tts_pair(root, 'cache_test_clean_tts', tts, tts_meta, data_suffix='_common')
    write_ref_audio_npz(root / 'cache_test_clean_tts.ref_audio.npz', tts_refs)

    # Reuse the own-utterance BiCodec globals already extracted for ASR.
    # This makes the alternate TTS cache free of a duplicate model pass.
    actual = tts.copy()
    actual[:, 168:] = clean_joint[:, 168:200]
    actual_meta = dict(tts_meta)
    actual_meta['speaker_token_source'] = 'actual_test_utterance'
    # Only the 632 actual-global cache is part of the established test contract.
    actual632 = _tts_geometry(actual)
    actual_meta.update(seq_len_tokens=132, text_seq_len=100, derived_from_seq_len_tokens=200)
    _write_cache(root / 'cache_test_clean_tts_632_actual_global_common.uint32',
                 root / 'cache_test_clean_tts_632_actual_global.meta.json', actual632, actual_meta)

    cont = enc.continuation(clean, cont_ids)
    cont_meta = _base_meta(hf_path=TEST_PATH, hf_config='clean', hf_split='test',
        split_name='librispeech_test_clean', n=len(cont_ids), fmt='packed_continuation_prefix_blocks',
        speaker_len=32, speech_len=75, ids=cont_ids)
    cont_meta.update(trim_seconds=CONT_SECONDS, min_duration=CONT_SECONDS)
    _write_cache(root / 'cache_test_cont.uint32', root / 'cache_test_cont.meta.json', cont, cont_meta)
    cont_index = {sid: i for i, sid in enumerate(clean_ids)}
    cont_refs = [clean_refs[cont_index[sid]] for sid in cont_ids]
    _prepare_test_auxiliary(args, clean, other, clean_ids, cont_ids, other_ids, cont_refs)
    if not args.skip_salmon:
        _prepare_salmon(args, enc)



def _prepare_salmon(args, encoder: EvaluationEncoder) -> None:
    """Build all SALMON prompt caches and row-aligned judge embeddings."""
    from utils.judge_models import JudgeModel

    out_dir = Path(args.test_out) / 'salmon'
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, judge_id in SALMON_TASKS.items():
        print(f'[salmon] loading {task}')
        dataset = load_dataset('SpeechPPL/SALMon_with_meta', task, split='train')
        for column in ('prompt_audio', 'continuation_audio_positive',
                       'continuation_audio_negative'):
            dataset = dataset.cast_column(column, Audio(decode=False))

        tokens = [None] * len(dataset)
        for start in range(0, len(dataset), args.salmon_batch_size):
            stop = min(start + args.salmon_batch_size, len(dataset))
            wavs = [decode_wav(dataset[i]['prompt_audio']['bytes']) for i in range(start, stop)]
            batch, lengths = _pad_wavs(wavs, encoder.ds_ratio, encoder.device)
            speaker = _tokenize_speaker_batch(encoder.bicodec, batch, 32).cpu()
            max_speech = max(math.ceil(length / encoder.ds_ratio) for length in lengths)
            speech, _ = _tokenize_speech_batch(
                encoder.codec, batch, lengths, max_speech, encoder.pad_speech,
            )
            speech = speech.cpu()
            for offset, row in enumerate(range(start, stop)):
                true_len = math.ceil(lengths[offset] / encoder.ds_ratio)
                tokens[row] = torch.cat((speaker[offset], speech[offset, :true_len])).to(torch.int64)
            print(f'[salmon] {task}: prompts {stop}/{len(dataset)}', flush=True)
        torch.save({
            'tokens': tokens, 'n_sequences': len(dataset), 'partition': task,
            'speaker_seq_len': 32, 'speaker_offset': SPEAKER_OFFSET,
            'speaker_vocab': SPEAKER_VOCAB, 'speech_offset': SPEECH_OFFSET,
            'speech_vocab': encoder.speech_vocab,
        }, out_dir / f'cache_salmon_{task}_prompt.pt')

        judge = JudgeModel(judge_id, MODEL_SR, encoder.device)
        embedded = {}
        for output_key, column in (
            ('positive', 'continuation_audio_positive'),
            ('negative', 'continuation_audio_negative'),
        ):
            pieces = []
            for start in range(0, len(dataset), args.salmon_batch_size):
                stop = min(start + args.salmon_batch_size, len(dataset))
                wavs = [
                    torch.from_numpy(decode_wav(dataset[i][column]['bytes']))
                    for i in range(start, stop)
                ]
                pieces.append(judge.embed_batch(wavs).detach().cpu())
            embedded[output_key] = torch.cat(pieces).numpy()
        np.savez(out_dir / f'judge_embeddings_{task}.npz', **embedded)

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', choices=('all', 'validation', 'test_common'), default='all')
    ap.add_argument('--validation_out', default='datasets/validation')
    ap.add_argument('--test_out', default='datasets/test_common')
    ap.add_argument('--assets_dir', default='assets/text_audio')
    ap.add_argument('--validation_size', type=int, default=512)
    ap.add_argument('--max_duration', type=float, default=32.0)
    ap.add_argument('--text_tokenizer', default='o200k_base')
    ap.add_argument('--speaker_model_dir', default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speech_model', default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speaker_checkpoint', default='assets/text_audio/wavlm_large_finetune.pth')
    ap.add_argument('--whisper_model', default='openai/whisper-large')
    ap.add_argument('--e2v_model', default=E2V_MODEL)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--statistics_batch_size', type=int, default=32)
    ap.add_argument('--salmon_batch_size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip_embeddings', action='store_true')
    ap.add_argument('--skip_transcriptions', action='store_true')
    ap.add_argument('--skip_statistics', action='store_true')
    ap.add_argument('--skip_salmon', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.target in {'all', 'validation'}:
        prepare_validation(args)
    if args.target in {'all', 'test_common'}:
        prepare_test_common(args)


if __name__ == '__main__':
    main()
