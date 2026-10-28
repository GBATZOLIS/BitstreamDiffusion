"""
MNIST-Sum, patch_size=28, 12x512 (~63M).

Derived from cc3m_lfq19_medium_24x1024_joint.py; see _mnist_sum_base.py for the
exhaustive list of what is overridden (data geometry, trunk scale, intra-patch
positional encoding, and the exact-accuracy callback). Everything else -- the
optimizer, EMA, loss, sigma block and the entire entropy schedule -- is the CC3M
recipe unchanged.
"""
import importlib.util as _ilu
from pathlib import Path as _P

# Loaded by file path: configs are exec'd standalone via spec_from_file_location,
# not imported as a package, so a relative import would fail.
_spec = _ilu.spec_from_file_location(
    "_mnist_sum_base", _P(__file__).with_name("_mnist_sum_base.py"))
_base = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_base)
build = _base.build


def get_config():
    return build(patch_size=28, caption_len_tokens=12,
                 embed_dim=512, n_blocks=12, dim_ff=2048, n_heads=8)
