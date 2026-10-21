#!/bin/bash

pip install stable-audio-tools==0.0.17 --no-deps
pip install stable-codec==0.1.2 --no-deps

# If using conda
sed -i 's/^import k_diffusion as K$/K = None  # lazy: not needed for autoencoder encode\/decode/' \
  ~/.conda/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/inference/sampling.py

sed -i '1s/^from pytorch_lightning.loggers import WandbLogger, CometLogger$/pass  # lazy: not needed for encode\/decode/' \
  ~/.conda/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/training/utils.py
# If using miniforge
# sed -i 's/^import k_diffusion as K$/K = None  # lazy: not needed for autoencoder encode\/decode/' \
#   ~/miniforge3/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/inference/sampling.py

# sed -i '1s/^from pytorch_lightning.loggers import WandbLogger, CometLogger$/pass  # lazy: not needed for encode\/decode/' \
#   ~/miniforge3/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/training/utils.py

pip install alias-free-torch==0.0.6 --no-deps
pip install einops einops-exts --no-deps
pip install packaging scipy soundfile
pip install torchsde torchdiffeq trampoline --no-deps

pip install omegaconf einx

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'SparkAudio/Spark-TTS-0.5B',
    allow_patterns='BiCodec/*',
    local_dir='Spark-TTS/pretrained_models/SparkTTS-0.5B',
)
print('done')
"

pip install jiwer
pip install git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git --no-deps
pip install pysptk pyworld fastdtw


# nemo install (TITANET) and dependencies
pip install nemo_toolkit[asr] --no-deps
pip install hydra-core==1.3.2 --no-deps
pip install lightning==2.4.0 --no-deps
pip install wget --no-deps
pip install onnx --no-deps
pip install ml_dtypes --no-deps
pip install fiddle --no-deps
pip install libcst --no-deps
pip install cloudpickle --no-deps
pip install nv-one-logger-core --no-deps
pip install StrEnum --no-deps
pip install nv-one-logger-training-telemetry --no-deps
pip install overrides --no-deps
pip install toml --no-deps
pip install nv-one-logger-pytorch-lightning-integration --no-deps
pip install lhotse --no-deps
pip install intervaltree --no-deps
pip install sortedcontainers --no-deps
pip install cytoolz --no-deps
pip install toolz --no-deps
pip install sentencepiece --no-deps
pip install kaldialign --no-deps
pip install pyannote.core --no-deps
pip install braceexpand --no-deps
pip install text-unidecode --no-deps
pip install webdataset --no-deps
pip install editdistance --no-deps
pip install pyannote.metrics --no-deps
