"""M0 v4 control: pure base log-normal sigma sampling."""
from configs.proteins.m0_v4 import build_config


def get_config():
    return build_config(1.00, "100")
