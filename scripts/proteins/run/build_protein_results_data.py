#!/usr/bin/env python
"""Assemble the protein evaluation report bundle from an evaluation output directory.

This command-line tool mirrors ``scripts/proteins/run/build_results_data.py`` but targets the
protein generation benchmark. It scans a single evaluation directory that holds per-task
metric JSON files and per-candidate score files, loads them tolerantly, and delegates the
actual composition and serialisation to ``evaluation.proteins.report.build_report`` and
``evaluation.proteins.report.write_report``.

The tool is idempotent: it only reads the evaluation directory and then rewrites the two
report artefacts, so repeated invocations over the same inputs yield identical outputs.

Expected evaluation directory layout, any subset of which is accepted::

    <eval-dir>/
        <task-name>/
            metrics.json                    per-task aggregate metrics
            candidates/<candidate-id>.json  per-candidate score files
            candidates.jsonl                or one candidate record per line
            scores/<candidate-id>.json      alternative per-candidate score files
        metrics/<task-name>.json            alternative flat task metrics
        <task-name>.metrics.json            alternative flat task metrics
        <task-name>.scores.jsonl            alternative flat candidate scores

Two artefacts are produced, both with configurable destinations:

* the JSON bundle selected by ``--out-json`` -- the canonical machine-readable payload;
* the JavaScript bundle selected by ``--out-js`` -- ``window.RESULTS_DATA = {...};`` so a
  static HTML report can render over ``file://`` without tripping CORS on script tags.

Only the Python standard library is imported at module load time, so importing this module
and printing its ``--help`` text both work without any heavy protein dependency installed.
The report module is imported lazily inside :func:`main`; if it is unavailable a clear,
actionable error explains how to restore it.

Example::

    .venv/bin/python scripts/proteins/run/build_protein_results_data.py \
        --eval-dir runs/proteins/eval/latest \
        --out-json results_data.json --out-js results_data.js
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def load_json(path: Path):
    """Load a JSON document, tolerating bare ``NaN`` and ``Infinity`` constants.

    Evaluation metric files legitimately contain non-finite floats. They are read as
    Python floats here and normalised to null only when the bundle is written.
    """
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=lambda _constant: float("nan"))


def load_jsonl(path: Path) -> list:
    """Load a JSON Lines file into a list of records, skipping blank lines."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(
            json.loads(stripped, parse_constant=lambda _constant: float("nan"))
        )
    return records


def _annotate_candidate(record, task, candidate):
    """Return a candidate record as a dict tagged with its task and candidate identity.

    Records that are not already mappings are wrapped under a ``scores`` key so that the
    downstream report always receives uniform candidate dictionaries.
    """
    if isinstance(record, dict):
        tagged = dict(record)
        tagged.setdefault("task", task)
        tagged.setdefault("candidate", candidate)
        return tagged
    return {"task": task, "candidate": candidate, "scores": record}


def discover_task_metrics(eval_dir: Path) -> dict:
    """Collect per-task aggregate metric payloads keyed by task name.

    Several common layouts are accepted, as described in the module docstring. The first
    payload found for a given task name wins, so specific per-task directories take
    precedence over the flatter fallbacks.
    """
    tasks: dict = {}
    for path in sorted(eval_dir.glob("*/metrics.json")):
        tasks.setdefault(path.parent.name, load_json(path))
    metrics_dir = eval_dir / "metrics"
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob("*.json")):
            tasks.setdefault(path.stem, load_json(path))
    suffix = ".metrics.json"
    for path in sorted(eval_dir.glob("*" + suffix)):
        tasks.setdefault(path.name[: -len(suffix)], load_json(path))
    return tasks


def discover_candidate_scores(eval_dir: Path) -> list:
    """Collect per-candidate score records across the accepted layouts.

    Every record is tagged with its originating task and a candidate identifier and is
    returned as a flat, deterministically ordered list.
    """
    records: list = []
    for path in sorted(eval_dir.glob("*/candidates/*.json")):
        records.append(
            _annotate_candidate(
                load_json(path), path.parent.parent.name, path.stem
            )
        )
    for path in sorted(eval_dir.glob("*/scores/*.json")):
        records.append(
            _annotate_candidate(
                load_json(path), path.parent.parent.name, path.stem
            )
        )
    for path in sorted(eval_dir.glob("*/candidates.jsonl")):
        for index, record in enumerate(load_jsonl(path)):
            candidate = (
                record.get("candidate", index)
                if isinstance(record, dict)
                else index
            )
            records.append(
                _annotate_candidate(record, path.parent.name, candidate)
            )
    jsonl_suffix = ".scores.jsonl"
    for path in sorted(eval_dir.glob("*" + jsonl_suffix)):
        task = path.name[: -len(jsonl_suffix)]
        for index, record in enumerate(load_jsonl(path)):
            candidate = (
                record.get("candidate", index)
                if isinstance(record, dict)
                else index
            )
            records.append(_annotate_candidate(record, task, candidate))
    return records


def _clean(value):
    """Recursively replace non-finite floats with ``None`` so output is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _call_with_pool(func, pool):
    """Call ``func`` supplying its arguments by name from ``pool``.

    This adapts to the exact parameter names chosen by the report module: parameters are
    matched by name against the pool, positional-only parameters are passed positionally,
    and a function that accepts ``**kwargs`` receives every remaining pool entry. A clear
    ``TypeError`` is raised when a required parameter cannot be supplied from the pool.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**pool)
    args: list = []
    kwargs: dict = {}
    used: set = set()
    accepts_var_keyword = False
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        if name in pool:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(pool[name])
            else:
                kwargs[name] = pool[name]
            used.add(name)
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"{getattr(func, '__name__', 'callable')} requires parameter {name!r} "
                "which this assembler does not know how to provide"
            )
    if accepts_var_keyword:
        for name, item in pool.items():
            if name not in used:
                kwargs.setdefault(name, item)
    return func(*args, **kwargs)


def _write_bundle(report, out_json: Path, out_js: Path) -> None:
    """Write the report bundle as JSON and as a ``window.RESULTS_DATA`` script.

    This local fallback mirrors the sibling assembler's output format and is used only
    when the report module does not expose its own ``write_report``.
    """
    cleaned = _clean(report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(cleaned, indent=1), encoding="utf-8")
    out_js.parent.mkdir(parents=True, exist_ok=True)
    out_js.write_text(
        "window.PROTEIN_RESULTS = "
        + json.dumps(cleaned, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


def _load_report_module():
    """Import and return ``evaluation.proteins.report``, or raise an actionable error.

    The repository root is placed on ``sys.path`` first so the local evaluation package
    resolves even when this file is executed directly as a script.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from evaluation.proteins import report
    except ImportError as error:
        raise RuntimeError(
            "Could not import evaluation.proteins.report, which composes the protein "
            "report bundle. Ensure the repository root is importable and the protein "
            "evaluation package is present; see the protein-eval extra (uv sync --extra protein-eval) and "
            "scripts/proteins/setup/setup_evaluation.sh for setup."
        ) from error
    return report


def parse_args(argv=None) -> argparse.Namespace:
    """Build the argument parser and parse ``argv`` (defaults to ``sys.argv``)."""
    parser = argparse.ArgumentParser(
        prog="build_protein_results_data.py",
        description=(
            "Assemble the protein evaluation report bundle from an evaluation output "
            "directory of per-task metric JSONs and per-candidate score files."
        ),
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation output directory to scan.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Destination for the JSON bundle "
        "(default: <eval-dir>/protein_results_data.json).",
    )
    parser.add_argument(
        "--out-js",
        type=Path,
        default=None,
        help="Destination for the window.RESULTS_DATA JavaScript bundle "
        "(default: <eval-dir>/protein_results_data.js).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Scan the evaluation directory and write the composed report artefacts.

    Returns a process exit code: ``0`` on success and ``2`` when the evaluation directory
    is missing.
    """
    args = parse_args(argv)
    eval_dir = args.eval_dir
    if not eval_dir.is_dir():
        print(
            f"error: evaluation directory not found: {eval_dir}",
            file=sys.stderr,
        )
        return 2

    out_json = args.out_json or eval_dir / "protein_results_data.json"
    out_js = args.out_js or eval_dir / "protein_results_data.js"

    tasks = discover_task_metrics(eval_dir)
    candidates = discover_candidate_scores(eval_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    eval_data = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "eval_dir": str(eval_dir),
        "tasks": tasks,
        "candidates": candidates,
    }

    report_module = _load_report_module()
    build_report = getattr(report_module, "build_report", None)
    if not callable(build_report):
        raise RuntimeError(
            "evaluation.proteins.report.build_report is missing or not callable; "
            "cannot compose the protein report bundle."
        )
    report = _call_with_pool(
        build_report,
        {
            "task_results": tasks,
            "candidates": candidates,
            "eval_dir": eval_dir,
            "eval_data": eval_data,
            "tasks": tasks,
            "generated_at": generated_at,
        },
    )

    write_report = getattr(report_module, "write_report", None)
    if callable(write_report):
        _call_with_pool(
            write_report,
            {
                "data": report,
                "report": report,
                "out_json": out_json,
                "out_js": out_js,
                "out_dir": out_json.parent,
                "eval_dir": eval_dir,
            },
        )
    else:
        _write_bundle(report, out_json, out_js)

    print(f"tasks: {len(tasks)}  candidates: {len(candidates)}")
    for name in sorted(tasks):
        print(f"  - {name}")
    print(f"wrote {out_json}")
    print(f"wrote {out_js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
