"""Composite scoring and ranking of generated protein candidates.

This module turns the per-candidate metrics produced elsewhere in the
evaluation pipeline (designability, pLDDT confidence, novelty, backbone
geometry validity, and sequence-structure co-design agreement) into a single
scalar score and a reproducible ranking. Every metric is first mapped onto a
common [0, 1] scale where larger always means better, then combined as a
weighted average. Components that are absent for a given candidate are skipped
and the remaining weights are renormalised, so partial metric coverage still
yields a usable score.

The scoring is deliberately pure Python: it depends only on the standard
library and is fully testable without any external tool, model, or GPU. It can
be imported in an environment that only has numpy and torch installed, since it
imports neither.

Methodological note: never tune the ranking weights on the final test split.
Choose weights on a held-out development split (or from prior knowledge) and
freeze them before scoring the test candidates, otherwise the reported ranking
quality is optimistically biased and no longer reflects generalisation.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

# Default weights for the composite score. The keys are the canonical component
# names understood by normalize_component. The values are relative importances;
# they need not sum to one because composite_score renormalises over whichever
# components are actually present on a candidate.
DEFAULT_RANK_WEIGHTS: dict[str, float] = {
    "designability": 0.35,
    "plddt": 0.20,
    "novelty": 0.20,
    "geometry_validity": 0.15,
    "codesign_agreement": 0.10,
}


def normalize_component(name: str, value: float) -> float:
    """Map a raw component metric onto [0, 1] where larger means better.

    The direction of each canonical component is fixed and documented here:

    - designability: fraction of self-consistency designs that succeed, or a
      binary designable flag. Already in [0, 1] and higher is better; passed
      through and clamped.
    - plddt: predicted local distance difference test confidence. Accepted on
      the 0-100 scale (divided by 100) or, if the value is already at most 1.0,
      treated as pre-normalised. Higher is better.
    - novelty: the raw input is the maximum TM-score to the training set. A
      lower TM-score means the candidate is more novel, so novelty is scored as
      1 - TM. Higher is better after inversion.
    - geometry_validity: fraction of residues whose backbone geometry is within
      valid bond-length and bond-angle tolerances. Already in [0, 1] and higher
      is better.
    - codesign_agreement: TM-score between the structure predicted from the
      co-designed sequence and the generated structure. Already in [0, 1] and
      higher is better.

    Unknown component names raise ValueError so that a mistyped or unsupported
    metric never silently receives an undocumented direction.
    """
    numeric = float(value)
    if math.isnan(numeric):
        raise ValueError(f"component '{name}' received NaN")
    if name == "designability":
        normalized = numeric
    elif name == "plddt":
        normalized = numeric / 100.0 if numeric > 1.0 else numeric
    elif name == "novelty":
        normalized = 1.0 - numeric
    elif name == "geometry_validity":
        normalized = numeric
    elif name == "codesign_agreement":
        normalized = numeric
    else:
        raise ValueError(
            f"unknown ranking component '{name}'; expected one of "
            f"{sorted(DEFAULT_RANK_WEIGHTS)}"
        )
    return min(1.0, max(0.0, normalized))


def _score_and_components(
    candidate: dict, weights: dict[str, float] | None
) -> tuple[float, dict[str, float]]:
    """Return the composite score and the per-component contributions.

    A component participates only when it is present on the candidate with a
    finite numeric value and carries a non-zero weight. The returned
    contributions are weight-renormalised so that they sum exactly to the score,
    which makes the ranking auditable and reproducible.
    """
    active = DEFAULT_RANK_WEIGHTS if weights is None else weights
    normalized: dict[str, float] = {}
    total_weight = 0.0
    for name, weight in active.items():
        if weight == 0.0 or name not in candidate:
            continue
        raw = candidate[name]
        if raw is None:
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        normalized[name] = weight * normalize_component(name, numeric)
        total_weight += weight
    if total_weight == 0.0:
        return 0.0, {}
    contributions = {
        name: part / total_weight for name, part in normalized.items()
    }
    score = math.fsum(contributions.values())
    return score, contributions


def composite_score(
    candidate: dict, weights: dict[str, float] | None = None
) -> float:
    """Return the weighted, renormalised composite score for one candidate.

    Each present component is normalised to [0, 1] via normalize_component and
    combined as a weighted average. Missing components are skipped and the
    weights of the remaining components are renormalised so the score stays in
    [0, 1]. A candidate with no scorable component receives 0.0.
    """
    return _score_and_components(candidate, weights)[0]


def rank_candidates(
    candidates: Iterable[dict], weights: dict[str, float] | None = None
) -> list[dict]:
    """Return a new list of candidates sorted by composite score, best first.

    The input candidates are not mutated. Each returned candidate is a shallow
    copy augmented with two keys: 'score', the composite score, and
    'score_components', a dict of the per-component weight-renormalised
    contributions that sum to 'score'. Ties are broken by the original input
    order, which keeps the ranking deterministic and reproducible.
    """
    scored: list[dict] = []
    for position, candidate in enumerate(candidates):
        score, contributions = _score_and_components(candidate, weights)
        augmented = dict(candidate)
        augmented["score"] = score
        augmented["score_components"] = contributions
        scored.append((position, augmented))
    scored.sort(key=lambda item: (-item[1]["score"], item[0]))
    return [augmented for _, augmented in scored]


def _demo() -> None:
    """Rank three synthetic candidates and print the result as JSON."""
    candidates = [
        {
            "name": "candidate_a",
            "designability": 0.9,
            "plddt": 82.0,
            "novelty": 0.35,
            "geometry_validity": 0.98,
            "codesign_agreement": 0.88,
        },
        {
            "name": "candidate_b",
            "designability": 0.6,
            "plddt": 74.0,
            "novelty": 0.70,
            "geometry_validity": 0.90,
            # codesign_agreement missing on purpose to show renormalisation.
        },
        {
            "name": "candidate_c",
            "designability": 0.95,
            "plddt": 0.65,
            "novelty": 0.55,
            "geometry_validity": 0.80,
            "codesign_agreement": 0.60,
        },
    ]
    ranked = rank_candidates(candidates)
    print(json.dumps(ranked, indent=2, sort_keys=True))


if __name__ == "__main__":
    _demo()
