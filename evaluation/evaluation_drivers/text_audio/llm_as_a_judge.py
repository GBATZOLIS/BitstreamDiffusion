"""
Make sure to export OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

from evaluation.evaluation_drivers.text_audio.eval import (
    _discover_tag_manifests,
    _load_samples,
    _update_tag_results,
)
from evaluation.evaluation_drivers.utils import _resolve_eval_dirs
from evaluation.utils import load_config

PROMPTS_PATH_DEFAULT = 'datasets/test_common/cache_test_cont.ref_transcriptions.json'
TASK_DEFAULT = 'cont_taste'

PROMPT_TEMPLATE = """
The task is evaluating the relevance and likelihood of the predicted text continuation, given the text prompt. You should also consider whether the meaning of the text continuation is making sense. The text prompt is:

"{prompt}"
, and the text continuation is :
"{content}"

You must give an overall rating from 1 to 5. The rating guideline is as below:

1: The text continuation is very unlikely and irrelevant to the text prompt.
2: The text continuation is unlikely and marginally relevant to the text prompt.
3: The text continuation is moderately likely and relevant to the text prompt.
4: The text continuation is likely and relevant to the text.
5: The text continuation is very likely and highly relevant.

You should take the following steps to provide the score:
First: briefly analyze the sample with the above definition.
Second: MUST follow the output format as: I would rate the score as _
"""


def _dbg(msg: str) -> None:
    print(f"[LLMJudge] {msg}", flush=True)


def _load_prompt_transcriptions(prompts_path: str) -> List[str]:
    with open(prompts_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    return [row['transcription'] for row in rows]


def _build_dataframe(
    prompts_path: str,
    samples: List[Dict[str, Any]],
    prompt_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    prompts = _load_prompt_transcriptions(prompts_path)
    if prompt_indices is not None:
        prompts = [prompts[index] for index in prompt_indices]
    if len(prompts) != len(samples):
        raise ValueError(
            f"{len(prompts)} reference prompts (from {prompts_path}) vs {len(samples)} "
            f"generated samples -- row-count mismatch, can't pair them up."
        )
    continuations = [s.get('whisper') for s in samples]
    missing = [i for i, c in enumerate(continuations) if c is None]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(samples)} samples have no 'whisper' transcription -- "
            f"run python -m evaluation.evaluation_drivers.text_audio.eval against this "
            f"tag/task first (it merges Whisper "
            f"transcriptions into samples.json)."
        )
    return pd.DataFrame({'prompt': prompts, 'continuation': continuations})


# -----------------------------------------------------------------------------
# OpenAI Batch API judging
# -----------------------------------------------------------------------------
def _judge_batch(
    df: pd.DataFrame, *, model: str, batch_size: int, output_dir: Path, poll_seconds: float,
) -> pd.DataFrame:
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = []
    num_samples = len(df)
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        filename = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
        with open(filename, 'w', encoding='utf-8') as f:
            for idx in range(batch_start, batch_end):
                row = df.iloc[idx]
                prompt = PROMPT_TEMPLATE.format(prompt=row['prompt'], content=row['continuation'])
                request = {
                    "custom_id": f"sample-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                }
                f.write(json.dumps(request) + "\n")
        batch_files.append(filename)
    _dbg(f'created {len(batch_files)} batch files under {output_dir}')

    all_results = []
    for batch_number, filename in enumerate(batch_files):
        _dbg(f'submitting batch {batch_number + 1}/{len(batch_files)}: {filename}')
        uploaded = client.files.create(file=open(filename, 'rb'), purpose='batch')
        batch = client.batches.create(
            input_file_id=uploaded.id, endpoint='/v1/responses', completion_window='24h',
        )
        _dbg(f'batch id: {batch.id}')

        while True:
            batch_status = client.batches.retrieve(batch.id)
            _dbg(f'status: {batch_status.status}')
            if batch_status.status == 'completed':
                break
            if batch_status.status in ('failed', 'cancelled', 'expired'):
                raise RuntimeError(f'batch failure: {batch_status}')
            time.sleep(poll_seconds)

        output = client.files.content(batch_status.output_file_id)
        results_file = output_dir / f'results_{batch_number}.jsonl'
        with open(results_file, 'wb') as f:
            f.write(output.read())
        _dbg(f'saved {results_file}')

        with open(results_file, encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                custom_id = result['custom_id']
                response_text = result['response']['body']['output'][0]['content'][0]['text']
                match = re.search(r"I would rate the score as\s*(\d)", response_text)
                score = int(match.group(1)) if match else None
                all_results.append({'custom_id': custom_id, 'response': response_text, 'score': score})

    results_df = pd.DataFrame(all_results)
    results_df['idx'] = results_df['custom_id'].str.replace('sample-', '', regex=False).astype(int)
    results_df = results_df.sort_values('idx').reset_index(drop=True)

    final_df = df.copy()
    final_df['judge_response'] = results_df['response'].values
    final_df['judge_score'] = results_df['score'].values
    return final_df


def build_csv_only(run_dir: Path, tag: str, task: str, prompts_path: str) -> Optional[Path]:
    """Writes the prompt/continuation pairs _build_dataframe() would hand to
    the judge, without calling OpenAI at all -- no OPENAI_API_KEY needed."""
    task_dir = run_dir / tag / task
    samples = _load_samples(task_dir)
    if not samples:
        _dbg(f'[{tag}] no {task} samples found under {task_dir}; skipping.')
        return None

    df = _build_dataframe(prompts_path, samples)
    out_path = task_dir / 'llm_judge_input.csv'
    df.to_csv(out_path, index=False)
    _dbg(f'[{tag}] wrote {out_path} ({len(df)} rows)')
    return out_path


# -----------------------------------------------------------------------------
# Per-tag driver
# -----------------------------------------------------------------------------
def judge_tag(
    run_dir: Path, tag: str, task: str, prompts_path: str, *,
    model: str, batch_size: int, poll_seconds: float, save_per_sample: bool,
) -> Optional[float]:
    task_dir = run_dir / tag / task
    samples = _load_samples(task_dir)
    if not samples:
        _dbg(f'[{tag}] no {task} samples found under {task_dir}; skipping.')
        return None

    df = _build_dataframe(prompts_path, samples)
    output_dir = task_dir / 'llm_judge_batches'
    final_df = _judge_batch(
        df, model=model, batch_size=batch_size, output_dir=output_dir, poll_seconds=poll_seconds,
    )

    avg_score = float(final_df['judge_score'].mean())
    _dbg(f'[{tag}] LLM-judge score ({model}) = {avg_score:.4f}')

    _update_tag_results(run_dir, tag, task, {'LLM-judge-score': avg_score})
    if save_per_sample:
        final_df.to_csv(task_dir / 'llm_judge_samples.csv', index=False)
        _dbg(f'[{tag}] wrote {task_dir / "llm_judge_samples.csv"}')

    return avg_score


def main() -> None:
    ap = argparse.ArgumentParser("LLM-as-a-Judge scoring for cont_taste continuations")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--run_dir", type=str, default=None, help="Override the auto-resolved shard directory")
    ap.add_argument(
        "--tags", nargs="+", default=None,
        help="Only judge these sampler tags (run_dir subdirectory names), instead of every "
             "tag with a manifest.json under run_dir.",
    )
    ap.add_argument(
        "--task", type=str, default=TASK_DEFAULT,
        help="Task subdirectory to judge (default: cont_taste).",
    )
    ap.add_argument("--prompts_path", type=str, default=PROMPTS_PATH_DEFAULT)
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--batch_size", type=int, default=300)
    ap.add_argument("--poll_seconds", type=float, default=30.0)
    ap.add_argument(
        "--save_per_sample", action="store_true",
        help="Also write a per-sample prompt/continuation/judge_response/judge_score CSV "
             "under run_dir/<tag>/<task>/llm_judge_samples.csv.",
    )
    ap.add_argument(
        "--csv_only", action="store_true",
        help="Just write the prompt/continuation pairs to "
             "run_dir/<tag>/<task>/llm_judge_input.csv and exit -- no OpenAI call, "
             "no OPENAI_API_KEY required.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        out_dir, _, _ = _resolve_eval_dirs(cfg)
        ckpt_tag = Path(str(getattr(cfg.evaluation, "checkpoint_path", "checkpoint"))).stem
        run_dir = out_dir / "textaudio_eval" / ckpt_tag

    tag_headers = _discover_tag_manifests(run_dir)
    if not tag_headers:
        raise FileNotFoundError(
            f"No {run_dir}/*/manifest.json found -- run the textaudio_generate metric and "
            "python -m evaluation.evaluation_drivers.text_audio.eval first."
        )

    tags = args.tags if args.tags else sorted(tag_headers.keys())
    missing = [t for t in tags if t not in tag_headers]
    if missing:
        raise ValueError(f"--tags {missing} not found under {run_dir} (available: {sorted(tag_headers)})")

    _dbg(f"run_dir: {run_dir}")
    _dbg(f"tags: {tags}")

    if args.csv_only:
        for tag in tags:
            build_csv_only(run_dir, tag, args.task, args.prompts_path)
        return

    scores: Dict[str, float] = {}
    for tag in tags:
        score = judge_tag(
            run_dir, tag, args.task, args.prompts_path,
            model=args.model, batch_size=args.batch_size, poll_seconds=args.poll_seconds,
            save_per_sample=args.save_per_sample,
        )
        if score is not None:
            scores[tag] = score

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
