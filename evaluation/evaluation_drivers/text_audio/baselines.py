from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

import importlib.util
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTILS_PATH = _REPO_ROOT / "baselines" / "text_audio" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("text_audio_baseline_utils", _UTILS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load baseline utils from {_UTILS_PATH}")
bu = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bu
_SPEC.loader.exec_module(bu)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("prepare", "train", "generate", "eval"), required=True)
    ap.add_argument("--baseline", choices=tuple(bu.BASELINE_TASKS), required=True)
    ap.add_argument("--tasks", nargs="+", default=None, help="Task tags for the selected baseline")
    ap.add_argument("--work-dir", type=Path, default=bu.DEFAULT_WORK_DIR)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--num-samples", type=bu.positive_int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")

    prep = ap.add_argument_group("data preparation")
    prep.add_argument("--prompt-dir", type=Path, default=None)
    prep.add_argument("--continuation-meta", type=Path, default=bu.REPO_ROOT / "datasets/test_common/cache_test_cont.meta.json")
    prep.add_argument("--speaker-cache", type=Path, default=bu.BASELINE_ROOT / "cascade/data/speaker_tokens_train.uint16")
    prep.add_argument("--train-cache", type=Path, default=bu.REPO_ROOT / "datasets/textaudio/cache_textaudio_train.uint32")
    prep.add_argument("--train-cache-meta", type=Path, default=bu.REPO_ROOT / "datasets/textaudio/cache_textaudio_train.meta.json")

    train = ap.add_argument_group("training")
    train.add_argument("--train-extra-args", nargs=argparse.REMAINDER, default=None)

    gen = ap.add_argument_group("generation")
    gen.add_argument("--generate-extra-args", nargs=argparse.REMAINDER, default=None)
    gen.add_argument("--model", default=None, help="Baseline model/checkpoint name/path")
    gen.add_argument("--codec-model", default="kyutai/mimi")
    gen.add_argument("--audio-tokenizer-path", default=None, help="GPA Spark tokenizer path")
    gen.add_argument("--attn-impl", default="sdpa", choices=("auto", "flash_attention_2", "eager", "sdpa"))
    gen.add_argument("--max-new-tokens", type=bu.positive_int, default=200)
    gen.add_argument("--max-length", type=bu.positive_int, default=1024)
    gen.add_argument("--temperature", type=bu.positive_float, default=None)
    gen.add_argument("--top-p", type=bu.probability, default=0.95)
    gen.add_argument("--top-k", type=bu.positive_int, default=30)
    gen.add_argument("--max-continuation-seconds", type=bu.positive_float, default=None)
    gen.add_argument("--continuation-only", action="store_true")

    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--config", help="Text-audio config used to resolve evaluator settings")
    ev.add_argument("--device", default=None)
    ev.add_argument("--save-fsd-embeddings", action="store_true")
    ev.add_argument("--fsd-embedding-batch-size", type=bu.positive_int, default=32)
    return ap


def _run_dir(args: argparse.Namespace) -> Path:
    return args.run_dir or (Path(args.work_dir) / "evaluation")


def _prompt_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    prompt_dir = args.prompt_dir or (Path(args.work_dir) / "prompts" / "cont_taste")
    return prompt_dir / "wavs", prompt_dir / "prompts.csv"


def _prepare(args: argparse.Namespace, tasks: Sequence[str]) -> None:
    if args.baseline == "cascade" or "joint" in tasks:
        if args.baseline == "cascade":
            bu.prepare_cascade_speaker_cache(
                output=args.speaker_cache,
                source=args.train_cache,
                source_meta=args.train_cache_meta,
                overwrite=args.overwrite,
            )
    if args.baseline == "gpa":
        manifest_dir = Path(args.work_dir) / "manifests" / "gpa"
        manifests = bu.prepare_gpa_manifests(
            output_dir=manifest_dir,
            tasks=tasks,
            limit=args.num_samples,
            overwrite=args.overwrite,
        )
        print(f"Prepared GPA manifests: {manifests}", flush=True)
    if "cont_taste" in tasks:
        prompt_root = args.prompt_dir or (Path(args.work_dir) / "prompts" / "cont_taste")
        csv_path = bu.prepare_continuation_prompts(
            output_dir=prompt_root,
            cache_meta=args.continuation_meta,
            limit=args.num_samples,
            overwrite=args.overwrite,
        )
        print(f"Prepared continuation prompts: {csv_path}", flush=True)


def _train(args: argparse.Namespace) -> None:
    if args.baseline != "cascade":
        raise SystemExit("--mode train is only available for --baseline cascade")
    bu.train_cascade_speaker_transformer(args.train_extra_args or [])


def _generate(args: argparse.Namespace, tasks: Sequence[str]) -> None:
    run_dir = _run_dir(args)
    layout = bu.BaselineLayout(args.baseline, run_dir)
    prompt_dir, prompt_csv = _prompt_paths(args)

    for task in tasks:
        task_dir = layout.task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        if args.baseline == "cascade" and task == "joint":
            bu.generate_cascade(
                output_dir=task_dir,
                num_samples=args.num_samples,
                stages=None,
                overwrite=args.overwrite,
                extra_args=args.generate_extra_args or [],
            )
            bu.normalize_joint_samples(task_dir)
        elif args.baseline == "taste" and task == "joint":
            bu.generate_taste_joint(
                output_dir=task_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                model_id=args.model or "MediaTek-Research/Llama-1B-TASTE-V0",
                resume=args.resume,
            )
        elif args.baseline == "taste" and task == "cont_taste":
            bu.generate_taste_continuations(
                prompt_dir=prompt_dir,
                prompt_csv=prompt_csv,
                output_dir=task_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                model_id=args.model or "MediaTek-Research/Llama-1B-TASTE-V0",
                resume=args.resume,
                max_continuation_seconds=args.max_continuation_seconds,
                generated_part_only=args.continuation_only,
            )
        elif args.baseline == "spiritlm" and task == "cont_taste":
            if not args.model:
                raise SystemExit("SpiritLM generation requires --model")
            bu.generate_spiritlm_continuations(
                prompt_dir=prompt_dir,
                prompt_csv=prompt_csv,
                output_dir=task_dir,
                model_id=args.model,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature is not None else 0.9,
                top_p=args.top_p,
                max_continuation_seconds=args.max_continuation_seconds,
                seed=args.seed,
                continuation_only=args.continuation_only,
            )
        elif args.baseline == "llama_mimi" and task == "cont_taste":
            bu.generate_llama_mimi_continuations(
                prompt_dir=prompt_dir,
                prompt_csv=prompt_csv,
                output_dir=task_dir,
                num_samples=args.num_samples,
                model_id=args.model or "llm-jp/Llama-Mimi-1.3B",
                codec_model=args.codec_model,
                max_length=args.max_length,
                temperature=args.temperature if args.temperature is not None else 0.8,
                top_k=args.top_k,
                max_continuation_seconds=args.max_continuation_seconds,
                seed=args.seed,
                continuation_only=args.continuation_only,
                resume=args.resume,
            )
        elif args.baseline == "gpa" and task == "tts":
            manifest = Path(args.work_dir) / "manifests" / "gpa" / "tts.csv"
            if not manifest.exists():
                bu.prepare_gpa_manifests(
                    output_dir=manifest.parent, tasks=["tts"],
                    limit=args.num_samples, overwrite=args.overwrite,
                )
            bu.generate_gpa(
                task="tts",
                manifest=manifest,
                output_dir=task_dir,
                model_path=args.model,
                audio_tokenizer_path=args.audio_tokenizer_path,
                num_samples=args.num_samples,
                seed=args.seed,
                resume=args.resume,
                device=args.device,
                attn_impl=args.attn_impl,
            )
        elif args.baseline == "gpa" and task == "stt":
            for partition in ("clean", "other"):
                manifest = Path(args.work_dir) / "manifests" / "gpa" / f"stt_{partition}.csv"
                if not manifest.exists():
                    bu.prepare_gpa_manifests(
                        output_dir=manifest.parent, tasks=["stt"],
                        limit=args.num_samples, overwrite=args.overwrite,
                    )
                bu.generate_gpa(
                    task="asr",
                    manifest=manifest,
                    output_dir=layout.task_dir(f"stt_{partition}"),
                    model_path=args.model,
                    audio_tokenizer_path=args.audio_tokenizer_path,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    resume=args.resume,
                    device=args.device,
                    attn_impl=args.attn_impl,
                )
        else:
            raise SystemExit(f"Generation not implemented for {args.baseline}/{task}")

    bu.write_manifest(run_dir, args.baseline, tasks)
    print(f"Generated baseline outputs under {layout.tag_dir}", flush=True)


def _eval(args: argparse.Namespace, tasks: Sequence[str]) -> None:
    if not args.config:
        raise SystemExit("--mode eval requires --config")
    from evaluation.evaluation_drivers.text_audio.eval import evaluate_textaudio_from_disk
    from evaluation.utils import load_config

    run_dir = _run_dir(args)
    for task in tasks:
        if task == "stt":
            for partition in ("clean", "other"):
                task_dir = run_dir / args.baseline / f"stt_{partition}"
                if task_dir.exists():
                    bu.normalize_task_outputs(task_dir, f"stt_{partition}")
            continue
        task_dir = run_dir / args.baseline / task
        if task_dir.exists():
            bu.normalize_task_outputs(task_dir, task)
    bu.write_manifest(run_dir, args.baseline, tasks)

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    results = evaluate_textaudio_from_disk(
        cfg,
        run_dir,
        device,
        tags_override=[args.baseline],
        tasks_override=list(tasks),
        save_fsd_embeddings=args.save_fsd_embeddings,
        fsd_embedding_batch_size=args.fsd_embedding_batch_size,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = _parser()
    args = ap.parse_args()
    tasks = bu.canonical_tasks(args.baseline, args.tasks)
    if args.mode == "prepare":
        _prepare(args, tasks)
    elif args.mode == "train":
        _train(args)
    elif args.mode == "generate":
        _generate(args, tasks)
    elif args.mode == "eval":
        _eval(args, tasks)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
