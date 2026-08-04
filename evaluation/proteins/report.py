"""Protein results-report data schema and assembly.

This module defines the JSON data schema for the CoBit protein evaluation
report (plan section 9.3). It extends the top-level results_data.json format
with protein-specific per-task tables, a sortable per-candidate table, a set of
published baseline reference rows, and a single headline composite metric. The
module is pure Python and imports with only the standard library present so that
report assembly never pulls in heavy evaluation tooling. Numpy is imported
lazily and only when a caller passes numpy scalars that need JSON coercion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROTEIN_REPORT_SCHEMA_VERSION = "1.0"

# Timestamp string stamped on every baseline row. All published numbers below
# are currently placeholders and must be filled in and re-verified before use.
BASELINE_LAST_VERIFIED = "2026-07-18, unverified placeholder"

# Canonical task ordering for the per-task tables. These names mirror the task
# families exercised by the protein multimodal benchmark.
TASK_ORDER = [
    "folding",
    "inverse_folding",
    "cogeneration",
    "scaffolding",
    "sequence_only",
    "tokenizer_ceiling",
]

TASK_DISPLAY_NAMES = {
    "folding": "Forward folding (sequence to structure)",
    "inverse_folding": "Inverse folding (structure to sequence)",
    "cogeneration": "Joint co-generation (sequence and structure)",
    "scaffolding": "Motif scaffolding",
    "sequence_only": "Sequence-only generation",
    "tokenizer_ceiling": "Structure tokenizer reconstruction ceiling",
}

TASK_DESCRIPTIONS = {
    "folding": (
        "Predict backbone structure from a clamped sequence and measure "
        "agreement with the reference fold."
    ),
    "inverse_folding": (
        "Recover an amino acid sequence from a clamped backbone and measure "
        "native recovery and self-consistency."
    ),
    "cogeneration": (
        "Generate sequence and structure jointly with no conditioning and "
        "measure designability, diversity, and novelty."
    ),
    "scaffolding": (
        "Complete a protein around a fixed functional motif and measure motif "
        "fidelity and scaffold success."
    ),
    "sequence_only": (
        "Generate sequences alone and measure foldability, diversity, and "
        "distributional match to natural proteins."
    ),
    "tokenizer_ceiling": (
        "Encode and decode reference backbones through the structure tokenizer "
        "to establish the reconstruction ceiling any model can reach."
    ),
}

# Ordered metric columns reported for each task. Every column key is also used
# by the candidate table and the baseline rows so that the three surfaces stay
# aligned.
TASK_METRIC_COLUMNS = {
    "folding": ["tm_score", "rmsd", "lddt", "gdt_ts"],
    "inverse_folding": ["aar", "sc_tm", "sc_rmsd", "perplexity"],
    "cogeneration": [
        "designable_fraction",
        "sc_tm",
        "sc_rmsd",
        "plddt",
        "diversity",
        "novelty",
    ],
    "scaffolding": [
        "success_rate",
        "motif_rmsd",
        "designable_fraction",
        "diversity",
    ],
    "sequence_only": [
        "plddt",
        "designable_fraction",
        "diversity",
        "novelty",
        "esm_pppl",
        "amino_acid_composition_jsd",
    ],
    "tokenizer_ceiling": [
        "tm_score",
        "rmsd",
        "lddt",
        "reconstruction_designable_fraction",
    ],
}

TASK_PRIMARY_METRIC = {
    "folding": "tm_score",
    "inverse_folding": "aar",
    "cogeneration": "designable_fraction",
    "scaffolding": "success_rate",
    "sequence_only": "designable_fraction",
    "tokenizer_ceiling": "tm_score",
}

# Metric keys where a smaller value is better. Every other metric key is assumed
# to be larger-is-better when building the sort direction hints.
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "rmsd",
        "sc_rmsd",
        "motif_rmsd",
        "perplexity",
        "esm_pppl",
        "amino_acid_composition_jsd",
        "fid",
        "mmd",
        "max_train_tm",
    }
)

# Row-identity keys that never count as metric columns when inferring columns
# from free-form task result rows.
_IDENTITY_KEYS = frozenset(
    {
        "method",
        "display_name",
        "name",
        "id",
        "family",
        "source",
        "notes",
        "last_verified",
    }
)

# Default thresholds used to decide whether a generated sample counts as
# designable, novel, and diverse for the headline composite.
HEADLINE_THRESHOLDS = {
    "designable_sc_tm": 0.5,
    "designable_sc_rmsd": 2.0,
    "novel_max_train_tm": 0.5,
    "diverse_cluster_tm": 0.5,
}


def _empty_metrics_for_tasks(tasks):
    """Return a nested metrics dict with a None entry for every task column."""

    return {
        task: {column: None for column in TASK_METRIC_COLUMNS[task]}
        for task in tasks
    }


def _baseline_row(method, display_name, family, source, tasks, notes=""):
    """Build one baseline reference row with empty (None) metric placeholders."""

    return {
        "method": method,
        "display_name": display_name,
        "family": family,
        "source": source,
        "last_verified": BASELINE_LAST_VERIFIED,
        "notes": notes,
        "metrics": _empty_metrics_for_tasks(tasks),
    }


# Published reference numbers, one row per competing method. Every numeric value
# is left as None until a real published figure is pinned and verified. The row
# structure, the source citation, and the last_verified marker are populated so
# downstream tooling can render the table before the numbers land.
BASELINE_ROWS = {
    "dplm2": _baseline_row(
        method="DPLM-2",
        display_name="DPLM-2 (650M)",
        family="Categorical multimodal diffusion language model",
        source=(
            "Wang et al., DPLM-2: A Multimodal Diffusion Protein Language Model, "
            "2024 (arXiv:2410.13782)."
        ),
        tasks=[
            "cogeneration",
            "inverse_folding",
            "folding",
            "sequence_only",
            "tokenizer_ceiling",
        ],
        notes="Reference categorical multimodal diffusion baseline.",
    ),
    "dplm2_bit": _baseline_row(
        method="DPLM-2 Bit",
        display_name="DPLM-2 Bit",
        family="Bitstream reproduction of DPLM-2",
        source="Bitstream reproduction of DPLM-2 (this work); numbers pending.",
        tasks=[
            "cogeneration",
            "inverse_folding",
            "folding",
            "sequence_only",
            "tokenizer_ceiling",
        ],
        notes="Bit-native ablation of the DPLM-2 protocol used as an internal control.",
    ),
    "multiflow": _baseline_row(
        method="MultiFlow",
        display_name="MultiFlow",
        family="Multimodal discrete and continuous flow matching",
        source=(
            "Campbell, Yim et al., Generative Flows on Discrete State-Spaces: "
            "Enabling Multimodal Flows with Applications to Protein Co-Design "
            "(MultiFlow), ICML 2024."
        ),
        tasks=["cogeneration", "inverse_folding", "folding", "scaffolding"],
        notes="Joint sequence and structure flow-matching co-design baseline.",
    ),
    "hd_prot": _baseline_row(
        method="HD-Prot",
        display_name="HD-Prot",
        family="Hierarchical protein diffusion",
        source="HD-Prot published reference; citation pending.",
        tasks=["cogeneration", "sequence_only"],
        notes="Placeholder row; citation and numbers to be confirmed.",
    ),
    "protein_ae": _baseline_row(
        method="ProteinAE",
        display_name="ProteinAE",
        family="Protein structure autoencoder tokenizer",
        source="ProteinAE published reference; citation pending.",
        tasks=["tokenizer_ceiling", "cogeneration"],
        notes="Structure-tokenizer reconstruction baseline; numbers to be confirmed.",
    ),
}


def _coerce_rows(payload):
    """Normalize a task result payload into a list of JSON-serializable rows."""

    if payload is None:
        return []
    if isinstance(payload, list):
        return [
            dict(row) if isinstance(row, dict) else {"value": row}
            for row in payload
        ]
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [
                dict(row) if isinstance(row, dict) else {"value": row}
                for row in payload["rows"]
            ]
        rows = []
        for name, metrics in payload.items():
            if isinstance(metrics, dict):
                row = {"method": name}
                row.update(metrics)
                rows.append(row)
            else:
                rows.append({"method": name, "value": metrics})
        return rows
    return [{"value": payload}]


def _infer_columns(rows):
    """Infer metric column keys from row contents, dropping identity keys."""

    columns = []
    for row in rows:
        for key in row:
            if key not in _IDENTITY_KEYS and key not in columns:
                columns.append(key)
    return columns


def _build_task_table(task, payload):
    """Assemble a single per-task table with columns, rows, and sort hints."""

    rows = _coerce_rows(payload)
    columns = list(TASK_METRIC_COLUMNS.get(task, ()))
    if not columns:
        columns = _infer_columns(rows)
    return {
        "task": task,
        "display_name": TASK_DISPLAY_NAMES.get(
            task, task.replace("_", " ").title()
        ),
        "description": TASK_DESCRIPTIONS.get(task, ""),
        "metric_columns": columns,
        "primary_metric": TASK_PRIMARY_METRIC.get(task),
        "higher_is_better": {
            column: column not in LOWER_IS_BETTER_METRICS for column in columns
        },
        "rows": rows,
        "num_rows": len(rows),
    }


def _as_float(value):
    """Coerce a value to float for sorting, mapping None or bad values to -inf."""

    try:
        if value is None:
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _rank_candidates(candidates):
    """Return candidates ordered best first.

    When every candidate already carries a score the ranking is a simple
    descending sort. Otherwise the sibling ranking module is used if it can be
    imported; if it is unavailable the candidates are sorted by whatever score
    is present so the report still assembles.
    """

    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        return rows
    if not all("score" in row for row in rows):
        try:
            from evaluation.proteins.rank import (
                rank_candidates as external_rank,
            )
        except Exception:
            pass
        else:
            return [dict(row) for row in external_rank(rows)]
    return sorted(
        rows, key=lambda row: _as_float(row.get("score")), reverse=True
    )


def _candidate_name(candidate):
    """Pick a human-readable name for a candidate from common key aliases."""

    return (
        candidate.get("name")
        or candidate.get("display_name")
        or candidate.get("id")
    )


def _build_candidate_table(candidates):
    """Assemble the sortable per-candidate table with metrics and score parts.

    Each candidate contributes one flat row combining identity fields, the union
    of reported metric keys, and the union of score-component keys (prefixed with
    ``component__`` to avoid collisions with metric columns).
    """

    ranked = _rank_candidates(candidates)
    metric_keys = []
    component_keys = []
    for row in ranked:
        for key in row.get("metrics") or {}:
            if key not in metric_keys:
                metric_keys.append(key)
        for key in row.get("score_components") or {}:
            if key not in component_keys:
                component_keys.append(key)

    identity_columns = ["rank", "name", "family", "task", "params", "score"]
    component_columns = ["component__" + key for key in component_keys]
    columns = identity_columns + list(metric_keys) + component_columns
    non_sortable = {"name", "family", "task"}

    table_rows = []
    for position, row in enumerate(ranked, start=1):
        metrics = row.get("metrics") or {}
        components = row.get("score_components") or {}
        flat = {
            "rank": position,
            "name": _candidate_name(row),
            "family": row.get("family"),
            "task": row.get("task"),
            "params": row.get("params"),
            "score": row.get("score"),
        }
        for key in metric_keys:
            flat[key] = metrics.get(key)
        for key in component_keys:
            flat["component__" + key] = components.get(key)
        table_rows.append(flat)

    return {
        "columns": columns,
        "metric_columns": list(metric_keys),
        "score_component_columns": component_columns,
        "sortable_columns": [
            column for column in columns if column not in non_sortable
        ],
        "sort_default": {"column": "score", "descending": True},
        "rows": table_rows,
        "num_rows": len(table_rows),
    }


def _is_designable(sample, thresholds):
    """Decide whether one generated sample counts as designable."""

    if "designable" in sample:
        return bool(sample["designable"])
    sc_tm = sample.get("sc_tm")
    if sc_tm is not None:
        return _as_float(sc_tm) >= thresholds["designable_sc_tm"]
    sc_rmsd = sample.get("sc_rmsd")
    if sc_rmsd is not None:
        return _as_float(sc_rmsd) <= thresholds["designable_sc_rmsd"]
    return False


def _is_novel(sample, thresholds):
    """Decide whether one generated sample counts as novel against training."""

    if "novel" in sample:
        return bool(sample["novel"])
    max_train_tm = sample.get("max_train_tm", sample.get("novelty_tm"))
    if max_train_tm is not None:
        return _as_float(max_train_tm) <= thresholds["novel_max_train_tm"]
    return False


def _is_diverse(sample, thresholds):
    """Decide whether one generated sample counts as diverse within the set."""

    if "diverse" in sample:
        return bool(sample["diverse"])
    cluster_tm = sample.get("cluster_tm", sample.get("nearest_neighbor_tm"))
    if cluster_tm is not None:
        return _as_float(cluster_tm) <= thresholds["diverse_cluster_tm"]
    return False


_HEADLINE_COUNT_KEYS = (
    "num_samples",
    "num_designable",
    "num_novel",
    "num_diverse",
    "num_designable_novel_diverse",
)


def _headline_counts_for_candidate(candidate, thresholds):
    """Return designable, novel, and diverse counts for one candidate or None.

    Precomputed count fields are trusted when present. Otherwise a per-sample
    ``samples`` list is thresholded. Candidates that carry neither contribute
    nothing to the headline and yield None.
    """

    if any(key in candidate for key in _HEADLINE_COUNT_KEYS):
        return {
            key: int(candidate.get(key) or 0) for key in _HEADLINE_COUNT_KEYS
        }
    samples = candidate.get("samples")
    if not samples:
        return None
    counts = {key: 0 for key in _HEADLINE_COUNT_KEYS}
    for sample in samples:
        designable = _is_designable(sample, thresholds)
        novel = _is_novel(sample, thresholds)
        diverse = _is_diverse(sample, thresholds)
        counts["num_samples"] += 1
        counts["num_designable"] += int(designable)
        counts["num_novel"] += int(novel)
        counts["num_diverse"] += int(diverse)
        counts["num_designable_novel_diverse"] += int(
            designable and novel and diverse
        )
    return counts


def _build_headline_composite(candidates, thresholds=None):
    """Compute the headline fraction of designable, novel, and diverse samples.

    The value is the fraction of generated samples that pass all three criteria,
    aggregated across every candidate that reports the needed counts. When no
    candidate supplies usable data the value is None but the definition and
    thresholds are still returned.
    """

    thresholds = dict(
        HEADLINE_THRESHOLDS if thresholds is None else thresholds
    )
    totals = {key: 0 for key in _HEADLINE_COUNT_KEYS}
    per_candidate = []
    for candidate in candidates:
        counts = _headline_counts_for_candidate(candidate, thresholds)
        if counts is None:
            continue
        per_candidate.append({"name": _candidate_name(candidate), **counts})
        for key in _HEADLINE_COUNT_KEYS:
            totals[key] += counts[key]

    denominator = totals["num_samples"]

    def _fraction(numerator):
        return (numerator / denominator) if denominator else None

    return {
        "metric": "fraction_designable_novel_diverse",
        "definition": (
            "Fraction of generated samples that are simultaneously designable, "
            "novel, and diverse."
        ),
        "thresholds": thresholds,
        "value": _fraction(totals["num_designable_novel_diverse"]),
        "counts": totals,
        "fractions": {
            "designable": _fraction(totals["num_designable"]),
            "novel": _fraction(totals["num_novel"]),
            "diverse": _fraction(totals["num_diverse"]),
            "designable_novel_diverse": _fraction(
                totals["num_designable_novel_diverse"]
            ),
        },
        "per_candidate": per_candidate,
    }


def build_report(task_results, candidates, baselines=None):
    """Assemble the full protein report as a JSON-serializable dictionary.

    The returned structure has the sections meta, per_task_tables (folding,
    inverse_folding, cogeneration, scaffolding, sequence_only, and
    tokenizer_ceiling), candidate_table, baseline_rows, and headline_composite.
    ``task_results`` maps a task name to its rows (a list of row dicts, a mapping
    of method name to metrics, or a dict with a ``rows`` key). ``candidates`` is
    a list of candidate dicts as produced by the ranking module. ``baselines``
    defaults to BASELINE_ROWS. The meta ``generated_at`` field is left as None so
    that write_report can stamp the true timestamp at write time.
    """

    task_results = dict(task_results or {})
    candidates = list(candidates or [])
    baselines = BASELINE_ROWS if baselines is None else baselines

    per_task_tables = {
        task: _build_task_table(task, task_results.get(task))
        for task in TASK_ORDER
    }
    for task, payload in task_results.items():
        if task not in per_task_tables:
            per_task_tables[task] = _build_task_table(task, payload)

    return {
        "meta": {
            "schema_version": PROTEIN_REPORT_SCHEMA_VERSION,
            "generated_at": None,
            "report_kind": "protein_multimodal_benchmark",
            "extends": "results_data.json",
            "tasks": list(TASK_ORDER),
        },
        "per_task_tables": per_task_tables,
        "candidate_table": _build_candidate_table(candidates),
        "baseline_rows": baselines,
        "headline_composite": _build_headline_composite(candidates),
    }


def _now_iso():
    """Return the current time as an ISO 8601 string in UTC."""

    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    """Coerce numpy scalars, numpy arrays, and paths for JSON serialization."""

    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(
        "Object of type {} is not JSON serializable".format(type(obj).__name__)
    )


def _with_generated_at(data):
    """Return a copy of the report with meta.generated_at stamped to now."""

    updated = dict(data)
    meta = dict(updated.get("meta") or {})
    meta["generated_at"] = _now_iso()
    updated["meta"] = meta
    return updated


def _atomic_write_text(path, text):
    """Write text to a path atomically by writing a temp file then replacing."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_report(data, out_json, out_js=None):
    """Write the report to JSON and optionally to a results_data.js style file.

    The JSON file is written pretty-printed. When ``out_js`` is provided a
    compact JavaScript file assigning ``window.PROTEIN_RESULTS`` is written
    alongside it. If the report's meta.generated_at is still None it is stamped
    with the current UTC time before writing. Returns a dict mapping ``json`` and
    ``js`` to the written paths (the ``js`` entry is None when not requested).
    """

    if (data.get("meta") or {}).get("generated_at") is None:
        data = _with_generated_at(data)

    out_json = Path(out_json)
    pretty = json.dumps(
        data, indent=2, ensure_ascii=False, default=_json_default
    )
    _atomic_write_text(out_json, pretty + "\n")

    written = {"json": str(out_json), "js": None}
    if out_js is not None:
        out_js = Path(out_js)
        compact = json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        _atomic_write_text(
            out_js, "window.PROTEIN_RESULTS = " + compact + ";\n"
        )
        written["js"] = str(out_js)
    return written


if __name__ == "__main__":
    import tempfile

    report = build_report(task_results={}, candidates=[])
    out_dir = Path(tempfile.mkdtemp(prefix="protein_report_"))
    paths = write_report(
        report,
        out_dir / "protein_results.json",
        out_dir / "protein_results.js",
    )
    print("wrote empty protein report:")
    print("  json:", paths["json"])
    print("  js:  ", paths["js"])
