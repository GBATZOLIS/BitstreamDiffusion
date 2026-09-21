# models/__init__.py
import os

from ml_collections import config_dict

from .sedd_wrapper import OfficialSEDDWrapper
from .sdt import SequenceVDTContinuousModel

# ---------------------------------------------------------------------------
# cfg.model.* validation
#
# A config that asks for a component this codebase does not implement used to be
# silently ignored. That is how CoBit-MLS came to be trained with
# `cfg.model.use_segment_embed = True` set in its config and no segment
# embedding in the model: the flag read as a documented architectural choice for
# months while having no effect whatsoever. Unrecognised keys now raise.
#
# RECOGNISED_MODEL_KEYS is the set of cfg.model keys the code actually reads.
# Adding a flag to a config without implementing it is now an error, and
# implementing one without listing it here is caught by the smoke test.
# ---------------------------------------------------------------------------
RECOGNISED_MODEL_KEYS = frozenset({
    "abs_pos_mode", "center_inputs", "content_dim_continuous",
    "content_dim_discrete", "continuous_logit_scaling", "dim_ff", "dropout",
    "embed_dim", "head_embed_dim", "head_hidden", "head_kernel", "head_type",
    "head_use_cross_attn", "head_use_local_mixer", "head_use_self_attn",
    "matched_filter_center", "matched_filter_clip", "matched_filter_scale",
    "max_seq_len", "name", "n_blocks", "n_fourier_global", "n_fourier_local",
    "n_heads", "out_dim", "patch_size", "rope_base", "rpb_max_distance",
    "scale_by_sigma", "self_condition", "use_adaln", "use_flash_attn",
    "use_rope_trunk", "use_swiglu",
})

# Vestigial keys: carried by the published OWT / LM1B configs, read by nothing.
# Each is pinned to the one value that was in force for every published run, so
# a config cannot use them to request behaviour that does not exist. They are
# tolerated rather than removed so the archived configs keep loading verbatim;
# a NEW config should not set them.
INERT_MODEL_KEYS = {
    "head_dilation": 1,
    "head_variant": "single",
    "n_pos_features": 1,
}


def validate_model_cfg(cfg: config_dict.ConfigDict) -> None:
    """Raise if cfg.model asks for something this codebase does not implement."""
    if os.environ.get("ALLOW_UNKNOWN_MODEL_CFG", "").strip() not in ("", "0", "false"):
        return

    unknown, wrong_inert = [], []
    for key in cfg.model.keys():
        if key in RECOGNISED_MODEL_KEYS:
            continue
        if key in INERT_MODEL_KEYS:
            expected = INERT_MODEL_KEYS[key]
            actual = cfg.model[key]
            if actual != expected:
                wrong_inert.append((key, actual, expected))
            continue
        unknown.append(key)

    if not unknown and not wrong_inert:
        return

    lines = ["cfg.model asks for components this codebase does not implement:"]
    for key in sorted(unknown):
        lines.append(f"  - cfg.model.{key} = {cfg.model[key]!r}: nothing reads this key.")
    for key, actual, expected in sorted(wrong_inert):
        lines.append(
            f"  - cfg.model.{key} = {actual!r}: this key is inert in this codebase and "
            f"only {expected!r} (its value for every published run) is accepted."
        )
    lines += [
        "",
        "Either implement the component and add its key to "
        "models.RECOGNISED_MODEL_KEYS, or remove the line from the config.",
        "A silently-ignored architecture flag is what made the published "
        "MNIST-Sum and CoBit-MLS configs disagree with the models they trained.",
        "Set ALLOW_UNKNOWN_MODEL_CFG=1 to bypass this check for local debugging.",
    ]
    raise ValueError("\n".join(lines))


def create_model(cfg: config_dict.ConfigDict):
    """
    Factory function that instantiates the model requested in the config.
    """
    validate_model_cfg(cfg)

    model_name = str(cfg.model.name).lower()

    if model_name == "official_sedd":
        print(f"✅ Instantiating Official SEDD backbone for framework: '{cfg.framework}'")
        return OfficialSEDDWrapper(cfg)

    elif model_name == "sdt":
        print(f"✅ Instantiating SequenceVDTContinuousModel for framework: '{cfg.framework}'")
        return SequenceVDTContinuousModel(cfg)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
