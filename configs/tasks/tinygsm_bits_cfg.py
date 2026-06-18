# configs/tasks/tinygsm_bits_cfg.py
#
# Classifier-free-guidance variant of the TinyGSM CoBit experiment.
# Identical to tinygsm_bits.py EXCEPT:
#   - cfg.cond.p_uncond = 0.1   -> conditioning dropout during training, so the
#     model learns an unconditional (null-prefix) mode. This is REQUIRED for CFG;
#     the base run uses p_uncond=0.0 and therefore cannot be guided.
#   - distinct experiment dir so it does not clobber the no-guidance run.
# The tokenized cache is shared (tag depends only on tokenizer/len/cap/val split),
# so no re-prebuild is needed.
import importlib.util
import os


def get_config():
    base_path = os.path.join(os.path.dirname(__file__), "tinygsm_bits.py")
    spec = importlib.util.spec_from_file_location("tinygsm_bits_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    # --- the only substantive change: enable classifier-free guidance training ---
    cfg.cond.p_uncond = 0.1          # 10% of examples trained with the null prefix
    # null_strategy stays "half" (uninformative 0.5 bits) as in the base config.

    # --- separate run dir so the two experiments coexist ---
    cfg.experiment = "tasks/tinygsm/cobit_raw_binary_bits_cfg"
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000250000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/gsm8k_eval"
    return cfg
