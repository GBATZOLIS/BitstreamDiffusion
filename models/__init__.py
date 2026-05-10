# models/__init__.py
from ml_collections import config_dict

from .sedd_wrapper import OfficialSEDDWrapper
from .sdt import SequenceVDTContinuousModel


def create_model(cfg: config_dict.ConfigDict):
    """
    Factory function that instantiates the model requested in the config.
    """
    model_name = str(cfg.model.name).lower()

    if model_name == "official_sedd":
        print(f"✅ Instantiating Official SEDD backbone for framework: '{cfg.framework}'")
        return OfficialSEDDWrapper(cfg)

    elif model_name == "sdt":
        print(f"✅ Instantiating SequenceVDTContinuousModel for framework: '{cfg.framework}'")
        return SequenceVDTContinuousModel(cfg)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
