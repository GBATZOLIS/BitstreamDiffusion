"""M0 v4: 25% base log-normal / 75% learned entropy sampling."""
from configs.proteins.m0_v4 import build_config


def get_config():
    return build_config(0.25, "025")
