"""Build the held-out-speaker voice-conversion TTS evaluation cache.

The output is speaker-major: each held-out speaker receives the same randomly
selected text prompts and its held-out-reference BiCodec tokens. The companion
ECAPA archive follows exactly the same row order for SpkSim evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_heldout(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [(str(row[0]), str(row[1])) for row in csv.reader(stream) if row]
    if not rows or len({speaker for speaker, _ in rows}) != len(rows):
        raise ValueError(f"{path}: expected a non-empty, unique speaker mapping")
    return rows


def build(args: argparse.Namespace) -> None:
    source_meta = json.loads(args.source_meta.read_text(encoding="utf-8"))
    n_source = int(source_meta["n_sequences"])
    seq_len = int(source_meta["seq_len_tokens"])
    text_len = int(source_meta["text_seq_len"])
    speaker_len = int(source_meta["speaker_seq_len"])
    if seq_len != text_len + speaker_len:
        raise ValueError("source TTS cache must contain only text and speaker blocks")

    source_ids = [str(value) for value in source_meta["ids"]]
    if len(source_ids) != n_source:
        raise ValueError("source metadata IDs do not match n_sequences")
    source = np.memmap(
        args.source_cache, dtype=np.uint32, mode="r", shape=(n_source, seq_len)
    )

    heldout = _read_heldout(args.heldout)
    first_row_by_speaker: dict[str, int] = {}
    for row, sample_id in enumerate(source_ids):
        first_row_by_speaker.setdefault(sample_id.split("-", 1)[0], row)
    missing = [speaker for speaker, _ in heldout if speaker not in first_row_by_speaker]
    if missing:
        raise ValueError(f"source cache has no held-out speaker-token row for: {missing}")

    if not 1 <= args.num_prompts <= n_source:
        raise ValueError(f"--num-prompts must be between 1 and {n_source}")
    prompt_rows = np.random.default_rng(args.seed).choice(
        n_source, size=args.num_prompts, replace=False
    )
    prompt_rows.sort()

    speakers = [speaker for speaker, _ in heldout]
    n_output = len(speakers) * len(prompt_rows)
    output = np.empty((n_output, seq_len), dtype=np.uint32)
    row_speakers: list[str] = []
    row_heldout_ids: list[str] = []
    row_prompt_ids: list[str] = []
    row = 0
    for speaker, heldout_id in heldout:
        speaker_tokens = source[first_row_by_speaker[speaker], text_len:].copy()
        for prompt_row in prompt_rows:
            output[row, :text_len] = source[prompt_row, :text_len]
            output[row, text_len:] = speaker_tokens
            row_speakers.append(speaker)
            row_heldout_ids.append(heldout_id)
            row_prompt_ids.append(source_ids[prompt_row])
            row += 1

    with np.load(args.heldout_embeddings) as archive:
        unique_speakers = [str(value) for value in archive["unique_speaker_ids"]]
        unique_ids = [str(value) for value in archive["unique_heldout_ids"]]
        unique_embeddings = np.asarray(
            archive["unique_heldout_embeddings"], dtype=np.float32
        )
    embedding_by_speaker = {
        speaker: (sample_id, embedding)
        for speaker, sample_id, embedding
        in zip(unique_speakers, unique_ids, unique_embeddings)
    }
    missing = [speaker for speaker in speakers if speaker not in embedding_by_speaker]
    if missing:
        raise ValueError(f"held-out embedding archive is missing speakers: {missing}")
    for speaker, expected_id in heldout:
        actual_id = embedding_by_speaker[speaker][0]
        if actual_id != expected_id:
            raise ValueError(
                f"held-out ID mismatch for speaker {speaker}: {actual_id} != {expected_id}"
            )
    ref_embeddings = np.stack([
        embedding_by_speaker[speaker][1] for speaker in row_speakers
    ]).astype(np.float32)

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output_cache)
    out_meta = dict(source_meta)
    out_meta.update(
        n_sequences=n_output,
        ids=[
            f"{speaker}::{prompt_id}"
            for speaker, prompt_id in zip(row_speakers, row_prompt_ids)
        ],
        n_speakers=len(speakers),
        task="tts_vc",
        row_order="speaker_major_then_prompt",
        speaker_token_source="heldout_reference_utterance",
        heldout_map=str(args.heldout),
        source_cache=str(args.source_cache),
        source_metadata=str(args.source_meta),
        prompt_seed=args.seed,
        num_prompts=len(prompt_rows),
        prompt_source_rows=[int(value) for value in prompt_rows],
        prompt_source_ids=[source_ids[value] for value in prompt_rows],
        row_speaker_ids=row_speakers,
        row_heldout_ids=row_heldout_ids,
        row_prompt_source_ids=row_prompt_ids,
    )
    args.output_meta.write_text(
        json.dumps(out_meta, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_embeddings,
        clean_ids=np.asarray(out_meta["ids"]),
        clean_embeddings=ref_embeddings,
        speaker_ids=np.asarray(row_speakers),
        heldout_ids=np.asarray(row_heldout_ids),
        prompt_source_ids=np.asarray(row_prompt_ids),
    )
    print(
        f"wrote {n_output} rows ({len(speakers)} speakers x {len(prompt_rows)} prompts)\n"
        f"  cache: {args.output_cache}\n  metadata: {args.output_meta}\n"
        f"  SpkSim references: {args.output_embeddings}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("datasets/test_common")
    parser.add_argument(
        "--source-cache", type=Path,
        default=root / "cache_test_clean_tts_632_common.uint32",
    )
    parser.add_argument(
        "--source-meta", type=Path,
        default=root / "cache_test_clean_tts_632.meta.json",
    )
    parser.add_argument("--heldout", type=Path, default=root / "heldout.csv")
    parser.add_argument(
        "--heldout-embeddings", type=Path,
        default=root / "ground_truth/tts/tts_heldout_spksim.npz",
    )
    parser.add_argument(
        "--output-cache", type=Path,
        default=root / "cache_test_clean_tts_vc_632_common.uint32",
    )
    parser.add_argument(
        "--output-meta", type=Path,
        default=root / "cache_test_clean_tts_vc_632.meta.json",
    )
    parser.add_argument(
        "--output-embeddings", type=Path,
        default=root / "ref_speaker_embeddings_tts_vc.npz",
    )
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
