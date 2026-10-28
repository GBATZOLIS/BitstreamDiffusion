"""Build the text-audio training caches.

The three stages can be run independently with ``--stage libritts``,
``--stage mls`` and ``--stage combine``.  ``--stage all`` runs them in that
order.  Geometry 632 is the default.  ``--geometry both`` tokenizes only the
larger 1000-token layout and derives the 632-token cache by block slicing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cache_utils import GEOMETRIES, build_packed_cache, load_valid_ids, resize_packed_rows

LIBRITTS_PATH = 'mythicinfinity/libritts'
LIBRITTS_CONFIG = 'all'
LIBRITTS_SPLITS = ('train.clean.100', 'train.clean.360', 'train.other.500')
MLS_PATH = 'parler-tts/mls_eng'
MLS_SPLIT = 'train'
CHUNK_ROWS = 20_000


def _cache_paths(root: Path, stem: str, total: int, multi: bool) -> tuple[Path, Path]:
    # The 632 cache stays at the historical unsuffixed path used by configs.
    suffix = '_1000' if multi and total == 1000 else ''
    return root / f'{stem}{suffix}.uint32', root / f'{stem}{suffix}.meta.json'


def _load_meta(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def resize_packed_cache(
    source_cache: Path, source_meta_path: Path,
    output_cache: Path, output_meta_path: Path, target_total: int,
) -> None:
    """Resize text and speech blocks without re-running any tokenizer."""
    target = GEOMETRIES[target_total]
    meta = _load_meta(source_meta_path)
    source = (int(meta['text_seq_len']), int(meta['speaker_seq_len']), int(meta['speech_seq_len']))
    if source[1] != target[1]:
        raise ValueError(f'speaker geometry cannot be resized: {source} -> {target}')
    n = int(meta['n_sequences'])
    src = np.memmap(source_cache, np.uint32, 'r', shape=(n, sum(source)))
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_cache.with_suffix(output_cache.suffix + '.tmp')
    dst = np.memmap(tmp, np.uint32, 'w+', shape=(n, sum(target)))
    for start in range(0, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        new = resize_packed_rows(
            src[start:stop], source, target,
            int(meta['pad_token_text']), int(meta['pad_token_speech']),
        )
        dst[start:stop] = new
    dst.flush()
    del dst
    tmp.replace(output_cache)

    out_meta = dict(meta)
    out_meta.update({
        'seq_len_tokens': sum(target),
        'text_seq_len': target[0],
        'speaker_seq_len': target[1],
        'speech_seq_len': target[2],
        'derived_from_seq_len_tokens': sum(source),
        'derived_from_cache': source_cache.name,
    })
    output_meta_path.write_text(json.dumps(out_meta, indent=2) + '\n', encoding='utf-8')
    print(f'[geometry] {source_cache} -> {output_cache} ({sum(source)} -> {sum(target)})')


def _gpu_ids(requested: list[int] | None) -> list[int | None]:
    if requested is not None:
        return requested or [None]
    import torch
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]


def _build_one(
    *, dataset, root: Path, stem: str, split_name: str, hf_path: str,
    hf_config: str | None, hf_split: str, text_field: str,
    geometry: str, args, row_durations=None,
) -> None:
    totals = [632, 1000] if geometry == 'both' else [int(geometry)]
    encode_total = max(totals)
    encode_cache, encode_meta = _cache_paths(root, stem, encode_total, geometry == 'both')
    text_len, speaker_len, speech_len = GEOMETRIES[encode_total]
    build_packed_cache(
        dataset=dataset, cache_path=encode_cache, meta_path=encode_meta,
        hf_path=hf_path, hf_config=hf_config, hf_split=hf_split,
        split_name=split_name, text_field=text_field,
        text_tokenizer_name=args.text_tokenizer,
        speaker_model_dir=args.speaker_model_dir, speech_model=args.speech_model,
        text_seq_len=text_len, speaker_seq_len=speaker_len, speech_seq_len=speech_len,
        gpu_ids=_gpu_ids(args.gpus), batch_size=args.batch_size,
        num_workers=args.num_workers, checkpoint_every=args.checkpoint_every,
        max_duration=args.max_duration if split_name == 'libri_train' else None,
        row_durations=row_durations,
    )
    if geometry == 'both':
        small_cache, small_meta = _cache_paths(root, stem, 632, True)
        resize_packed_cache(encode_cache, encode_meta, small_cache, small_meta, 632)


def build_libritts(args) -> None:
    from datasets import Audio, concatenate_datasets, load_dataset
    root = Path(args.libritts_out)
    root.mkdir(parents=True, exist_ok=True)
    valid_ids = load_valid_ids(args.train_duration_file, args.max_duration)
    ds = load_dataset(LIBRITTS_PATH, LIBRITTS_CONFIG)
    dataset = concatenate_datasets([ds[x] for x in LIBRITTS_SPLITS]).cast_column(
        'audio', Audio(decode=False),
    )
    dataset = dataset.filter(lambda row: row['id'] in valid_ids)
    _build_one(
        dataset=dataset, root=root, stem='cache_libri_train', split_name='libri_train',
        hf_path=LIBRITTS_PATH, hf_config=LIBRITTS_CONFIG,
        hf_split='+'.join(LIBRITTS_SPLITS), text_field='text_normalized',
        geometry=args.geometry, args=args,
    )



def _resolve_mls_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        downloaded = Path(hf_home) / 'mls_eng'
        if downloaded.exists():
            return str(downloaded)
    return MLS_PATH

def build_mls(args) -> None:
    from datasets import Audio, load_dataset
    root = Path(args.mls_out)
    root.mkdir(parents=True, exist_ok=True)
    mls_path = _resolve_mls_path(args.mls_path)
    dataset = load_dataset(mls_path, split=MLS_SPLIT).cast_column('audio', Audio(decode=False))
    durations = np.asarray(dataset['audio_duration'], dtype=np.float64)
    _build_one(
        dataset=dataset, root=root, stem='cache_mls_train', split_name='mls_train',
        hf_path=mls_path, hf_config=None, hf_split=MLS_SPLIT, text_field='transcript',
        geometry=args.geometry, args=args, row_durations=durations,
    )


def _combine_geometry(args, total: int) -> None:
    multi = args.geometry == 'both'
    sources = []
    for name, root, stem in (
        ('libri', Path(args.libritts_out), 'cache_libri_train'),
        ('mls', Path(args.mls_out), 'cache_mls_train'),
    ):
        cache, meta_path = _cache_paths(root, stem, total, multi)
        meta = _load_meta(meta_path)
        if int(meta['seq_len_tokens']) != total:
            raise ValueError(f'{meta_path}: expected geometry {total}')
        sources.append((name, cache, meta))

    out_root = Path(args.combined_out)
    out_root.mkdir(parents=True, exist_ok=True)
    out_cache, out_meta = _cache_paths(out_root, 'cache_textaudio_train', total, multi)
    with out_cache.open('wb') as output:
        for name, path, meta in sources:
            n = int(meta['n_sequences'])
            mm = np.memmap(path, np.uint32, 'r', shape=(n, total))
            for start in range(0, n, CHUNK_ROWS):
                output.write(np.asarray(mm[start:start + CHUNK_ROWS]).tobytes())
            print(f'[combine:{total}] {name}: {n} rows')

    reference = sources[0][2]
    combined = dict(reference)
    combined.update({
        'hf_path': 'combined:' + '+'.join(x[2]['hf_path'] for x in sources),
        'hf_config': None, 'hf_split': 'train', 'split_name': 'textaudio_train',
        'n_sequences': sum(int(x[2]['n_sequences']) for x in sources),
        'source_datasets': [x[0] for x in sources],
        'source_n_sequences': {x[0]: int(x[2]['n_sequences']) for x in sources},
    })
    out_meta.write_text(json.dumps(combined, indent=2) + '\n', encoding='utf-8')
    print(f'[combine:{total}] wrote {out_cache}')


def combine(args) -> None:
    for total in ([632, 1000] if args.geometry == 'both' else [int(args.geometry)]):
        _combine_geometry(args, total)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=('all', 'libritts', 'mls', 'combine'), default='all')
    ap.add_argument('--geometry', choices=('632', '1000', 'both'), default='632')
    ap.add_argument('--libritts_out', default='datasets/libri')
    ap.add_argument('--mls_out', default='datasets/mls')
    ap.add_argument(
        '--mls_path', default=None,
        help='MLS path/repository; defaults to $HF_HOME/mls_eng when present, otherwise parler-tts/mls_eng.',
    )
    ap.add_argument('--combined_out', default='datasets/textaudio')
    ap.add_argument('--train_duration_file', default='assets/text_audio/durations/train.csv')
    ap.add_argument('--max_duration', type=float, default=32.0)
    ap.add_argument('--text_tokenizer', default='o200k_base')
    ap.add_argument('--speaker_model_dir', default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speech_model', default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=32)
    ap.add_argument('--gpus', type=int, nargs='*', default=None)
    ap.add_argument('--checkpoint_every', type=int, default=25)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {'all', 'libritts'}:
        build_libritts(args)
    if args.stage in {'all', 'mls'}:
        build_mls(args)
    if args.stage in {'all', 'combine'}:
        combine(args)


if __name__ == '__main__':
    main()
