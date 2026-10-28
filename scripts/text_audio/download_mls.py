from huggingface_hub import snapshot_download
import os

# Make sure to export HF_TOKEN and HF_HOME before running
snapshot_download(
    repo_id="parler-tts/mls_eng",
    repo_type="dataset",
    allow_patterns=["data/*.parquet"],
    local_dir=f"{os.environ['HF_HOME']}/mls_eng",
)