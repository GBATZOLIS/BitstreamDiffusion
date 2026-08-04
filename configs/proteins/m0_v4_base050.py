"""M0 v4: balanced 50% base log-normal / 50% learned entropy sampling."""
from configs.proteins.m0_v4 import build_config


def get_config():
    return build_config(0.50, "050")
