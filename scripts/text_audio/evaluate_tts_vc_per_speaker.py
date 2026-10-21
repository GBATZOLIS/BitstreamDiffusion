#!/usr/bin/env python3
"""Evaluate saved TTS-VC generations separately for every held-out speaker.

Whisper is never loaded: ASR metrics reuse the ``whisper`` strings already
stored in samples.json. UTMOS and SpkSim are recomputed on GPU per speaker.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from evaluation.utils import load_config
from utils.textaudio_utils import TextAudioEvaluator, _read_wav

DEFAULT_CONFIG = ROOT / "configs/text_audio/mls/eval_sweeps/tts_vc.py"
DEFAULT_RUN_DIR = ROOT / "runs/cobit_mls/evaluation/textaudio_eval/last"
DEFAULT_TAG = "ddim_entropic_nfe512_stoch-g0.27_gs5"


def _load_inputs(cfg: Any, run_dir: Path, tag: str):
    task_dir = run_dir / tag / "tts_vc"
    samples_path = task_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    meta_path = Path(str(cfg.data.tts_vc_meta_path))
    if not meta_path.is_absolute():
        meta_path = ROOT / meta_path
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n = int(meta["n_sequences"])
    speaker_ids = [str(value) for value in meta["row_speaker_ids"]]
    heldout_ids = [str(value) for value in meta["row_heldout_ids"]]
    if not (len(samples) == len(speaker_ids) == len(heldout_ids) == n):
        raise ValueError(
            f"row mismatch: samples={len(samples)}, speakers={len(speaker_ids)}, "
            f"heldout={len(heldout_ids)}, metadata={n}"
        )
    missing = [i for i, sample in enumerate(samples) if not sample.get("whisper")]
    if missing:
        raise ValueError(
            f"{samples_path}: {len(missing)} rows lack saved Whisper transcriptions; "
            f"first missing rows: {missing[:10]}"
        )

    wavs = []
    for index, sample in enumerate(samples):
        wav_path = task_dir / str(sample["gen_wav"])
        if not wav_path.is_file():
            raise FileNotFoundError(f"generated waveform {index} not found: {wav_path}")
        wavs.append(_read_wav(wav_path))
    return task_dir, samples, speaker_ids, heldout_ids, wavs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(str(args.config))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    task_dir, samples, speaker_ids, heldout_ids, wavs = _load_inputs(
        cfg, args.run_dir, args.tag
    )
    textaudio_cfg = cfg.evaluation.multimodal_text_audio
    reference_path = Path(str(textaudio_cfg.tts_vc_speaker_ref_path))
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    speaker_checkpoint = Path(str(getattr(
        textaudio_cfg, "speaker_checkpoint",
        "assets/text_audio/wavlm_large_finetune.pth",
    )))
    if not speaker_checkpoint.is_absolute():
        speaker_checkpoint = ROOT / speaker_checkpoint

    evaluator = TextAudioEvaluator(
        sr=int(getattr(cfg.data, "sample_rate", 16_000)),
        speaker_checkpoint=str(speaker_checkpoint),
        partition=str(getattr(cfg.data, "partition", "clean")),
        speaker_ref_path=str(reference_path),
        _dbg_func=lambda message: print(
            f"[TTS-VC per speaker] {message}", flush=True
        ),
    )

    ordered_speakers = list(dict.fromkeys(speaker_ids))
    per_speaker: dict[str, Any] = {}
    for number, speaker in enumerate(ordered_speakers, start=1):
        indices = [i for i, value in enumerate(speaker_ids) if value == speaker]
        speaker_heldout_ids = {heldout_ids[i] for i in indices}
        if len(speaker_heldout_ids) != 1:
            raise ValueError(f"speaker {speaker} maps to multiple held-out IDs")
        refs = [str(samples[i]["ref_text"]) for i in indices]
        hyps = [str(samples[i]["whisper"]) for i in indices]
        speaker_wavs = [wavs[i] for i in indices]
        metrics = {
            "ASR-WER": evaluator.word_error_rate(refs, hyps),
            "ASR-CER": evaluator.character_error_rate(refs, hyps),
            "UTMOS": evaluator.utmos_score(speaker_wavs),
            "SpkSim": evaluator.spksim(
                speaker_wavs,
                device,
                reference_indices=indices,
                reference_path=str(reference_path),
            ),
        }
        per_speaker[speaker] = {
            "heldout_id": next(iter(speaker_heldout_ids)),
            "n_samples": len(indices),
            **metrics,
        }
        values = " ".join(
            f"{name}={value:.4f}" for name, value in metrics.items()
        )
        print(
            f"[{number:02d}/{len(ordered_speakers)}] "
            f"speaker={speaker} {values}",
            flush=True,
        )

    metric_names = ("ASR-WER", "ASR-CER", "UTMOS", "SpkSim")
    macro = {
        name: float(np.mean([row[name] for row in per_speaker.values()]))
        for name in metric_names
    }
    result = {
        "header": {
            "config": str(args.config),
            "run_dir": str(args.run_dir),
            "tag": args.tag,
            "task_dir": str(task_dir),
            "device": str(device),
            "used_saved_whisper_transcriptions": True,
            "n_speakers": len(per_speaker),
            "n_samples": len(samples),
        },
        "macro_average": macro,
        "per_speaker": per_speaker,
    }
    if evaluator._asr is not None:
        raise AssertionError("Whisper was unexpectedly loaded")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.run_dir = args.run_dir.resolve()
    output = args.output or (
        args.run_dir / args.tag / "tts_vc" / "per_speaker_metrics.json"
    )
    output = output.resolve()

    result = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "speaker_id", "heldout_id", "n_samples",
            "ASR-WER", "ASR-CER", "UTMOS", "SpkSim",
        ))
        for speaker, row in result["per_speaker"].items():
            writer.writerow((
                speaker, row["heldout_id"], row["n_samples"],
                row["ASR-WER"], row["ASR-CER"], row["UTMOS"], row["SpkSim"],
            ))
    print(f"Macro average: {json.dumps(result['macro_average'])}", flush=True)
    print(f"Wrote {output}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
