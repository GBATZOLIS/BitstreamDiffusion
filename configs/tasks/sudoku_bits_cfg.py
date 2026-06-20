# configs/tasks/sudoku_bits_cfg.py
#
# CFG-capable variant of configs/tasks/sudoku_bits.py: identical in every way
# EXCEPT it trains with classifier-free-guidance conditioning dropout
# (cfg.cond.p_uncond > 0), so the model learns a null-conditioned (unconditional)
# path consistent with the sampler's _make_null_full("half"). Writes to a SEPARATE
# run dir (..._cfg) so the original p_uncond=0 checkpoints are never clobbered.
#
#   SUDOKU_DIFFICULTY=easy|medium|hard   (same as base)
#   SUDOKU_P_UNCOND=0.1                  (CFG dropout prob; default 0.1)
import os
import importlib.util

_BASE = os.path.join(os.path.dirname(__file__), "sudoku_bits.py")


def _base_get_config():
    spec = importlib.util.spec_from_file_location("_sudoku_bits_base", _BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config


def get_config():
    cfg = _base_get_config()()

    # Enable classifier-free guidance training: drop the puzzle-clue prefix to the
    # "half" null with this probability so an unconditional path is learned.
    cfg.cond.p_uncond = float(os.environ.get("SUDOKU_P_UNCOND", "0.1"))

    # Fresh, non-clobbering run dir.
    difficulty = cfg.data.difficulty
    cfg.experiment = f"tasks/sudoku/{difficulty}/cobit_raw_binary_bits_cfg"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000020000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/sudoku_eval"

    return cfg
