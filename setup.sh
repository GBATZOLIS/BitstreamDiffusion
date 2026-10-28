#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PROFILE="${1:-train}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    cat >&2 <<'EOF'
No active Conda environment was detected.

Create and activate one first, following the language-modeling workflow:
  conda create -n bitstream python=3.11 -y
  conda activate bitstream

Then rerun this script, for example:
  bash setup.sh train
EOF
    exit 1
fi

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"BitstreamDiffusion requires Python 3.11; active interpreter is "
        f"{sys.version.split()[0]} at {sys.executable}"
    )
print(f"Using Conda Python: {sys.executable}")
PY

# Always address pip through the active interpreter. On clusters, a bare
# `pip` command frequently resolves to a system installation instead of the
# activated Conda environment.
python -m pip --version
python -m pip install --upgrade pip

install_pytorch() {
    local include_audio="${1:-0}"
    local packages=("torch==2.7.1" "torchvision==0.22.1")
    if [[ "$include_audio" == "1" ]]; then
        packages+=("torchaudio==2.7.1")
    fi
    python -m pip install "${packages[@]}" --index-url "$PYTORCH_INDEX_URL"
}

case "$PROFILE" in
    train|protein)
        install_pytorch 0
        python -m pip install -r requirements.txt
        ;;
    eval)
        install_pytorch 0
        python -m pip install -r requirements-eval.txt
        ;;
    mnist-sum)
        install_pytorch 0
        python -m pip install -r requirements-mnist-sum.txt
        ;;
    text)
        install_pytorch 0
        python -m pip install -r requirements-text.txt
        ;;
    text-audio)
        install_pytorch 1
        python -m pip install -r requirements-text-audio.txt
        ;;
    protein-eval)
        install_pytorch 0
        python -m pip install -r requirements-proteins.txt
        python scripts/proteins/download_models.py
        ;;
    dev)
        install_pytorch 1
        python -m pip install -r requirements-dev.txt
        ;;
    *)
        echo "Usage: ./setup.sh {train|eval|mnist-sum|text|text-audio|protein|protein-eval|dev}" >&2
        exit 2
        ;;
esac

cat <<EOF

Installed the '$PROFILE' profile into the active Conda environment:
  $CONDA_PREFIX

Optional FlashAttention build (Linux/CUDA toolchain required):
  python -m pip install flash-attn --no-build-isolation
EOF
