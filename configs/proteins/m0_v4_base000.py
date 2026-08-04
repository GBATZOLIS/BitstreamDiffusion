"""M0 v4 control: fully learned entropic schedule after transition."""
from configs.proteins.m0_v4 import build_config


def get_config():
    return build_config(0.00, "000")
