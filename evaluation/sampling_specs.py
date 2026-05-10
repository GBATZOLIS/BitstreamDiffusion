# evaluation/sampling_specs.py
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List

from evaluation.nfe import steps_for_target_nfe


# ---------------------------------------------------------------------
# Stochastic sampler fields that may be copied from a base sweep spec.
# ---------------------------------------------------------------------

_STOCH_SPEC_KEYS = (
    "stochastic_enabled",
    "s_churn",
    "s_noise",
    "window_mode",
    "entropy_quantile_lo",
    "entropy_quantile_hi",
    "s_tmin",
    "s_tmax",
    "entropy_fallback",
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _default_ati_etas(cfg: Any, sweep_cfg: Any) -> List[float]:
    vals = getattr(sweep_cfg, "ati_etas", None)
    if vals is not None:
        return [float(v) for v in list(vals)]

    ev = getattr(cfg, "evaluation", None)
    if ev is not None:
        ati_cfg = getattr(ev, "ati", None)
        if ati_cfg is not None and bool(getattr(ati_cfg, "enabled", False)):
            return [float(getattr(ati_cfg, "eta", 0.0))]

        if hasattr(ev, "ati_eta"):
            return [float(getattr(ev, "ati_eta", 0.0))]

    return [0.0]


def _fmt_opt_float(x: Any) -> str:
    if x is None:
        return "none"
    return f"{float(x):g}"


def _copy_stochastic_fields(dst: Dict[str, Any], base: Any) -> None:
    for k in _STOCH_SPEC_KEYS:
        if hasattr(base, k):
            dst[k] = getattr(base, k)


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _sweep_values(
    base: Any,
    singular: str,
    plural: str,
    default: Any = None,
) -> list:
    """
    Resolve a scalar field or a plural sweep field.

    Examples:
      s_churn=8.0          -> [8.0]
      s_churns=[8.0,12.0] -> [8.0, 12.0]
      missing             -> [default]

    This intentionally fails if both singular and plural forms are present.
    """
    has_singular = hasattr(base, singular)
    has_plural = hasattr(base, plural)

    if has_singular and has_plural:
        raise ValueError(
            f"Specify either '{singular}' or '{plural}', not both, "
            f"in sampling_sweep spec: {base}"
        )

    if has_plural:
        vals = _as_list(getattr(base, plural))
        if len(vals) == 0:
            raise ValueError(f"'{plural}' must not be empty in sampling_sweep spec: {base}")
        return vals

    if has_singular:
        return [getattr(base, singular)]

    return [default]


def _maybe_float(x: Any) -> Any:
    if x is None:
        return None
    return float(x)


def _maybe_str(x: Any) -> Any:
    if x is None:
        return None
    return str(x)


# ---------------------------------------------------------------------
# Tag construction
# ---------------------------------------------------------------------

def _stochastic_tag_suffix(spec: Dict[str, Any]) -> str | None:
    if not bool(spec.get("stochastic_enabled", False)):
        return None

    s_churn = float(spec.get("s_churn", 0.0))
    s_noise = float(spec.get("s_noise", 1.0))
    window_mode = str(spec.get("window_mode", "entropy_cdf")).lower()

    if window_mode == "entropy_cdf":
        q_lo = float(spec.get("entropy_quantile_lo", 0.10))
        q_hi = float(spec.get("entropy_quantile_hi", 0.90))
        return f"stoch-ch{s_churn:g}-n{s_noise:g}-q{q_lo:g}_{q_hi:g}"

    if window_mode == "fixed":
        return (
            f"stoch-ch{s_churn:g}-n{s_noise:g}-"
            f"t{_fmt_opt_float(spec.get('s_tmin'))}_{_fmt_opt_float(spec.get('s_tmax'))}"
        )

    return f"stoch-ch{s_churn:g}-n{s_noise:g}-wm-{window_mode}"


def _tag_for_spec(spec: Dict[str, Any]) -> str:
    sampler = str(spec["sampler_name"])
    target = int(spec["target_nfe"])
    actual = int(spec["actual_nfe"])
    steps = int(spec["num_steps"])
    gs = float(spec["guidance_scale"])

    sigma = spec.get("terminal_sigma", None)
    sc_mode = spec.get("sc_refresh_mode", None)
    ati_eta = float(spec.get("ati_eta", 0.0))

    parts = [
        sampler,
        f"tnfe-{target}",
        f"anfe-{actual}",
        f"steps-{steps}",
        f"gs{gs:.1f}",
    ]

    if sc_mode is not None:
        parts.insert(1, f"scr-{sc_mode}")

    if ati_eta > 0.0:
        parts.append(f"ati{ati_eta:.2f}")

    if sigma is not None and float(sigma) > 0.0:
        parts.append(f"term{math.log10(float(sigma)):.2f}")

    stoch_suffix = _stochastic_tag_suffix(spec)
    if stoch_suffix is not None:
        parts.append(stoch_suffix)

    return "_".join(parts)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_resolved_spec(spec: Dict[str, Any]) -> None:
    if "sampler_name" not in spec:
        raise ValueError(f"Resolved sampling spec missing sampler_name: {spec}")

    if "num_steps" not in spec:
        raise ValueError(f"Resolved sampling spec missing num_steps: {spec}")

    if "actual_nfe" not in spec:
        raise ValueError(f"Resolved sampling spec missing actual_nfe: {spec}")

    if "target_nfe" not in spec:
        raise ValueError(f"Resolved sampling spec missing target_nfe: {spec}")

    if bool(spec.get("stochastic_enabled", False)):
        if "s_churn" not in spec:
            raise ValueError(
                "stochastic_enabled=True but no s_churn was resolved. "
                "Use s_churn=... or s_churns=[...] in the sampling sweep config. "
                f"Spec: {spec}"
            )

        if float(spec.get("s_churn", 0.0)) <= 0.0:
            raise ValueError(
                "stochastic_enabled=True but resolved s_churn <= 0. "
                "This would silently produce a deterministic run. "
                f"Spec: {spec}"
            )

        # Fill safe defaults explicitly, so downstream metadata/cache keys are explicit.
        spec.setdefault("s_noise", 1.0)
        spec.setdefault("window_mode", "entropy_cdf")
        spec.setdefault("entropy_quantile_lo", 0.10)
        spec.setdefault("entropy_quantile_hi", 0.90)
        spec.setdefault("s_tmin", None)
        spec.setdefault("s_tmax", None)
        spec.setdefault("entropy_fallback", "deterministic")


def _validate_all_specs(specs: List[Dict[str, Any]]) -> None:
    seen_tags = set()

    for spec in specs:
        _validate_resolved_spec(spec)

        tag = spec.get("tag", None)
        if tag is None:
            raise ValueError(f"Resolved sampling spec missing tag: {spec}")

        if tag in seen_tags:
            raise ValueError(
                f"Duplicate sampling spec tag resolved: {tag}\n"
                f"Spec: {spec}\n"
                "Tags must be unique because they key result rows and cache files."
            )

        seen_tags.add(tag)


# ---------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------

def _expand_stochastic_sweep_values(base: Any) -> list[dict]:
    """
    Expand stochastic sweep-valued fields into explicit per-spec override dicts.

    Supported scalar/plural pairs:
      s_churn / s_churns
      s_noise / s_noises
      window_mode / window_modes
      entropy_quantile_lo / entropy_quantile_los
      entropy_quantile_hi / entropy_quantile_his
      s_tmin / s_tmins
      s_tmax / s_tmaxs
      entropy_fallback / entropy_fallbacks

    Missing fields return [None] and therefore do not override base/defaults.
    """
    stochastic_enabled = bool(getattr(base, "stochastic_enabled", False))

    if not stochastic_enabled:
        return [dict(stochastic_enabled=False)]

    s_churn_values = _sweep_values(base, "s_churn", "s_churns", None)
    s_noise_values = _sweep_values(base, "s_noise", "s_noises", None)
    window_mode_values = _sweep_values(base, "window_mode", "window_modes", None)
    qlo_values = _sweep_values(base, "entropy_quantile_lo", "entropy_quantile_los", None)
    qhi_values = _sweep_values(base, "entropy_quantile_hi", "entropy_quantile_his", None)
    s_tmin_values = _sweep_values(base, "s_tmin", "s_tmins", None)
    s_tmax_values = _sweep_values(base, "s_tmax", "s_tmaxs", None)
    fallback_values = _sweep_values(base, "entropy_fallback", "entropy_fallbacks", None)

    expanded: list[dict] = []

    for (
        s_churn,
        s_noise,
        window_mode,
        qlo,
        qhi,
        s_tmin,
        s_tmax,
        fallback,
    ) in itertools.product(
        s_churn_values,
        s_noise_values,
        window_mode_values,
        qlo_values,
        qhi_values,
        s_tmin_values,
        s_tmax_values,
        fallback_values,
    ):
        item: dict = {"stochastic_enabled": True}

        if s_churn is not None:
            item["s_churn"] = float(s_churn)
        if s_noise is not None:
            item["s_noise"] = float(s_noise)
        if window_mode is not None:
            item["window_mode"] = str(window_mode)
        if qlo is not None:
            item["entropy_quantile_lo"] = float(qlo)
        if qhi is not None:
            item["entropy_quantile_hi"] = float(qhi)
        if s_tmin is not None:
            item["s_tmin"] = float(s_tmin)
        if s_tmax is not None:
            item["s_tmax"] = float(s_tmax)
        if fallback is not None:
            item["entropy_fallback"] = str(fallback)

        expanded.append(item)

    return expanded


def build_sampling_specs(
    *,
    cfg: Any,
    metric_cfg: Any,
) -> List[Dict[str, Any]] | None:
    sweep_cfg = getattr(getattr(cfg, "evaluation", object()), "sampling_sweep", None)
    if sweep_cfg is None or not bool(getattr(sweep_cfg, "enabled", False)):
        return None

    framework = str(cfg.framework).lower()
    global_target_nfes = list(getattr(sweep_cfg, "target_nfes", []))
    base_specs = list(getattr(sweep_cfg, "specs", []))
    terminal_sigmas = list(getattr(metric_cfg, "terminal_sigmas", [0.08]))
    guidance_scales = list(getattr(metric_cfg, "guidance_scales", [0.0]))

    out: List[Dict[str, Any]] = []

    for base in base_specs:
        sampler_name = str(getattr(base, "sampler_name"))

        target_nfes = list(getattr(base, "target_nfes", global_target_nfes))
        if len(target_nfes) == 0:
            continue

        # --------------------------------------------------------------
        # Discrete branch
        # --------------------------------------------------------------
        if framework.startswith("discrete"):
            for target_nfe in target_nfes:
                num_steps, actual_nfe = steps_for_target_nfe(
                    framework=framework,
                    sampler_name=sampler_name,
                    target_nfe=int(target_nfe),
                    self_condition=False,
                    sc_refresh_mode="refined",
                    return_probs=False,
                )

                for gs in guidance_scales:
                    spec = dict(
                        sampler_name=sampler_name,
                        target_nfe=int(target_nfe),
                        actual_nfe=int(actual_nfe),
                        num_steps=int(num_steps),
                        terminal_sigma=0.0,
                        guidance_scale=float(gs),
                        ati_eta=0.0,
                    )

                    # Discrete branch currently does not use continuous stochastic churn.
                    spec["tag"] = _tag_for_spec(spec)
                    out.append(spec)

            continue

        # --------------------------------------------------------------
        # Continuous branch
        # --------------------------------------------------------------
        sc_refresh_modes = list(getattr(base, "sc_refresh_modes", ["refined"]))
        ati_etas = list(getattr(base, "ati_etas", _default_ati_etas(cfg, sweep_cfg)))
        stochastic_overrides = _expand_stochastic_sweep_values(base)

        for sc_refresh_mode in sc_refresh_modes:
            for target_nfe in target_nfes:
                num_steps, actual_nfe = steps_for_target_nfe(
                    framework=framework,
                    sampler_name=sampler_name,
                    target_nfe=int(target_nfe),
                    self_condition=bool(getattr(cfg.model, "self_condition", False)),
                    sc_refresh_mode=str(sc_refresh_mode),
                    return_probs=True,
                )

                for sigma in terminal_sigmas:
                    for gs in guidance_scales:
                        for ati_eta in ati_etas:
                            for stoch_override in stochastic_overrides:
                                spec = dict(
                                    sampler_name=sampler_name,
                                    sc_refresh_mode=str(sc_refresh_mode),
                                    ati_eta=float(ati_eta),
                                    target_nfe=int(target_nfe),
                                    actual_nfe=int(actual_nfe),
                                    num_steps=int(num_steps),
                                    terminal_sigma=float(sigma),
                                    guidance_scale=float(gs),
                                )

                                # Copy scalar stochastic fields from base, then overwrite
                                # with explicit expanded values. This keeps compatibility
                                # with existing singular configs.
                                _copy_stochastic_fields(spec, base)
                                spec.update(stoch_override)

                                # If stochastic is enabled, ensure defaults are explicit
                                # before the tag/cache metadata are constructed.
                                _validate_resolved_spec(spec)

                                spec["tag"] = _tag_for_spec(spec)
                                out.append(spec)

    _validate_all_specs(out)
    return out