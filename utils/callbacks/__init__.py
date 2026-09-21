from __future__ import annotations

# Base
from .base import Callback

# Individual callbacks
from .generation import GenerationCallback
from .sigma_data import SigmaDataEstimator
from .sigma_grad_norm import SigmaGradNormCallback
from .entropy_schedule_plot import EntropySchedulePlotCallback
from .offline_entropy_profile import OfflineEntropyProfileCallback
from .vlb_bound import VLBBoundCallback
from .external_ppl import ExternalPPLCallback
from .mauve import MauveCallback
from .visualization import VisualizationCallback
# TextAudioCallback's module level reaches the full speech stack (jiwer /
# fairseq / s3prl / NeMo -- see textaudio_install.sh). Guard it so that
# importing this package does not require the audio dependencies; an audio
# config that actually asks for the callback still fails, loudly, on
# construction rather than being silently skipped.
try:
    from .textaudio_generation import TextAudioCallback
except ImportError as _e:  # pragma: no cover - depends on optional extras
    _TEXTAUDIO_IMPORT_ERROR = _e

    class TextAudioCallback:  # type: ignore[no-redef]
        """Placeholder raised into existence only if the audio extras are absent."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TextAudioCallback requires the text+audio extras, which are not "
                f"installed in this environment (original error: {_TEXTAUDIO_IMPORT_ERROR}). "
                "Install them with textaudio_install.sh."
            ) from _TEXTAUDIO_IMPORT_ERROR
from .mnist_sum_eval import MNISTSumEvalCallback

# Optional extras (export only if you want them available)
# from .entropy_schedule import EntropyScheduleCallback
# from .denoise import denoise_grid

__all__ = [
    "Callback",
    "GenerationCallback",
    "SigmaDataEstimator",
    "SigmaGradNormCallback",
    "EntropySchedulePlotCallback",
    "OfflineEntropyProfileCallback",
    "VLBBoundCallback",
    # "EntropyScheduleCallback",
    # "denoise_grid",
    "TextAudioCallback",
    "MNISTSumEvalCallback",
]
