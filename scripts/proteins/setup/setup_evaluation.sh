#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PROFILE="${1:-core}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
EVODIFF_COMMIT="33206e99446f799ec11cf6e57d66ffaec837be91"
DPLM_COMMIT="8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

setup_core() {
  if [ ! -x .venv/bin/python ]; then
    echo "Run ./setup.sh first to create the main training environment."
    exit 1
  fi
  uv pip install --python .venv/bin/python -r "$ROOT/pyproject.toml" --extra protein-eval
  .venv/bin/python -c "import Bio, sentencepiece, sklearn; print('protein evaluator dependencies: OK')"
}

setup_evodiff() {
  mkdir -p external .venvs
  if [ ! -d external/evodiff/.git ]; then
    git clone https://github.com/microsoft/evodiff.git external/evodiff
  fi
  git -C external/evodiff fetch origin "$EVODIFF_COMMIT"
  git -C external/evodiff checkout --detach "$EVODIFF_COMMIT"
  uv venv --python 3.11 --allow-existing .venvs/evodiff
  uv pip install --python .venvs/evodiff/bin/python --index-url "$TORCH_INDEX" torch
  uv pip install --python .venvs/evodiff/bin/python -e external/evodiff
  PYTHONPATH="$ROOT" .venvs/evodiff/bin/python -m evaluation.proteins.generate_evodiff --help >/dev/null
  echo "EvoDiff environment: .venvs/evodiff"
}

setup_dplm() {
  mkdir -p external .venvs
  if [ ! -d external/dplm/.git ]; then
    git clone https://github.com/bytedance/dplm.git external/dplm
  fi
  git -C external/dplm fetch origin "$DPLM_COMMIT"
  git -C external/dplm checkout --detach "$DPLM_COMMIT"
  uv venv --python 3.11 --allow-existing .venvs/dplm
  uv pip install --python .venvs/dplm/bin/python --index-url "$TORCH_INDEX" torch torchvision
  uv pip install --python .venvs/dplm/bin/python -r "$ROOT/pyproject.toml" --extra dplm-inference
  uv pip install --python .venvs/dplm/bin/python --no-deps -e external/dplm
  PYTHONPATH="$ROOT" .venvs/dplm/bin/python -m evaluation.proteins.generate_dplm --help >/dev/null
  echo "DPLM environment: .venvs/dplm"
}

case "$PROFILE" in
  core) setup_core ;;
  evodiff) setup_evodiff ;;
  dplm) setup_dplm ;;
  all) setup_core; setup_evodiff; setup_dplm ;;
  *) echo "Usage: $0 {core|evodiff|dplm|all}"; exit 2 ;;
esac

