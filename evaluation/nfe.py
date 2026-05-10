# evaluation/nfe.py
from __future__ import annotations


def _normalize_sc_refresh_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"refined", "refresh", "full"}:
        return "refined"
    if mode in {"carry", "unrefined", "no_refresh", "no-refine"}:
        return "carry"
    raise ValueError(f"Unknown sc_refresh_mode='{mode}'")


def compute_nfe(
    framework: str,
    sampler_name: str,
    num_steps: int,
    *,
    self_condition: bool = False,
    sc_refresh_mode: str = "refined",
    return_probs: bool = False,
    discrete_denoise: bool = True,
) -> int:
    fw = str(framework).lower()
    s = str(sampler_name).lower()
    steps = int(num_steps)
    sc_refresh_mode = _normalize_sc_refresh_mode(sc_refresh_mode)

    if fw.startswith("continuous"):
        if s in {"heun", "heun_karras", "karras"}:
            nfe = (2 * steps) if not self_condition else ((3 if sc_refresh_mode == "refined" else 2) * steps)
            if return_probs:
                nfe += 1
            return nfe

        if s in {"ddim", "ddim_karras", "ddim_entropic", "entropic", "euler"}:
            nfe = steps if not self_condition else ((2 if sc_refresh_mode == "refined" else 1) * steps)
            if return_probs:
                nfe += 1
            return nfe

        return steps + (1 if return_probs else 0)

    if fw.startswith("discrete"):
        if s in {"tweedie", "tweedie_tau", "tweedie_tau_leaping"}:
            return steps + (1 if discrete_denoise else 0)

        if s in {"euler", "euler_rate"}:
            return steps + (1 if discrete_denoise else 0)

    return steps


def steps_for_target_nfe(
    *,
    framework: str,
    sampler_name: str,
    target_nfe: int,
    self_condition: bool,
    sc_refresh_mode: str = "refined",
    return_probs: bool = True,
) -> tuple[int, int]:
    target_nfe = int(target_nfe)
    fw = str(framework).lower()
    s = str(sampler_name).lower()

    if fw.startswith("discrete"):
        final_eval = 1 if s in {"tweedie", "tweedie_tau", "tweedie_tau_leaping", "euler", "euler_rate"} else 0
        num_steps = max(1, target_nfe - final_eval)
        actual_nfe = compute_nfe(
            framework=framework,
            sampler_name=sampler_name,
            num_steps=num_steps,
            self_condition=False,
            sc_refresh_mode="refined",
            return_probs=False,
            discrete_denoise=True,
        )
        return num_steps, actual_nfe

    final_eval = 1 if return_probs else 0
    one_step_nfe = compute_nfe(
        framework=framework,
        sampler_name=sampler_name,
        num_steps=1,
        self_condition=self_condition,
        sc_refresh_mode=sc_refresh_mode,
        return_probs=False,
    )
    usable = max(target_nfe - final_eval, 1)
    num_steps = max(1, usable // one_step_nfe)

    actual_nfe = compute_nfe(
        framework=framework,
        sampler_name=sampler_name,
        num_steps=num_steps,
        self_condition=self_condition,
        sc_refresh_mode=sc_refresh_mode,
        return_probs=return_probs,
    )
    return num_steps, actual_nfe