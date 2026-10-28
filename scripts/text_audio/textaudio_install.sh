#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the BitstreamDiffusion Conda environment before running this script." >&2
    exit 1
fi

# Keep the language-modeling guide's invariant: every package is installed by
# the Python interpreter belonging to the active Conda environment.
python -m pip --version

python -m pip install stable-audio-tools==0.0.17 --no-deps
python -m pip install stable-codec==0.1.2 --no-deps

# These imports are irrelevant to codec encode/decode but eagerly pull in
# optional training/sampling stacks. Patch the installed package by resolving
# it through the active interpreter rather than assuming ~/.conda, Python 3.10,
# or a particular Conda distribution.
python - <<'PY'
from pathlib import Path

import stable_audio_tools

root = Path(stable_audio_tools.__file__).resolve().parent
replacements = {
    root / "inference" / "sampling.py": (
        "import k_diffusion as K",
        "K = None  # lazy: not needed for autoencoder encode/decode",
    ),
    root / "training" / "utils.py": (
        "from pytorch_lightning.loggers import WandbLogger, CometLogger",
        "pass  # lazy: not needed for autoencoder encode/decode",
    ),
}
for path, (old, new) in replacements.items():
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY

python -m pip install alias-free-torch==0.0.6 --no-deps
python -m pip install einops einops-exts --no-deps
python -m pip install packaging scipy soundfile
python -m pip install torchsde torchdiffeq trampoline --no-deps
python -m pip install omegaconf einx

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "SparkAudio/Spark-TTS-0.5B",
    allow_patterns="BiCodec/*",
    local_dir="Spark-TTS/pretrained_models/SparkTTS-0.5B",
)
print("done")
PY

python -m pip install jiwer
python -m pip install git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git --no-deps
python -m pip install pysptk pyworld fastdtw

# NeMo/TITANET and the explicitly selected dependency surface used by the
# evaluation code. --no-deps retains the versions chosen by the project
# requirements instead of letting these optional packages replace them.
python -m pip install "nemo_toolkit[asr]" --no-deps
python -m pip install hydra-core==1.3.2 --no-deps
python -m pip install lightning==2.4.0 --no-deps
python -m pip install \
    wget onnx ml_dtypes fiddle libcst cloudpickle nv-one-logger-core \
    StrEnum nv-one-logger-training-telemetry overrides toml \
    nv-one-logger-pytorch-lightning-integration lhotse intervaltree \
    sortedcontainers cytoolz toolz sentencepiece kaldialign pyannote.core \
    braceexpand text-unidecode webdataset editdistance pyannote.metrics \
    --no-deps
