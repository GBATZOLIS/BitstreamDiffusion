#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible environment for BitstreamDiffusion (text + protein runs) using
# the uv package manager.
#
#   - installs uv if missing
#   - creates a local venv at ".venv", displayed as "bitstream"
#   - installs a Blackwell-capable (CUDA 12.8 / sm_120) PyTorch
#   - installs every dependency from pyproject.toml (the 'train' extra + dev)
#   - verifies torch sees the GPUs and the repo imports for text + protein
#
# After the environment it prepares data idempotently by composing the
# per-dataset downloader scripts: the DPLM structure tokenizer and the M0 paired
# dataset plus the sequence corpora by default. Each downloader checks the target
# path and a hash/size/row-count before doing work and skips anything already
# complete, so re-running setup only does the remaining work. Large structure
# corpora (AFDB representatives, ESMAtlas) are gated behind explicit flags so the
# default run does not pull terabytes.
#
# Usage:
#   ./setup.sh                 # env + tokenizer + M0 + sequence corpora
#   SKIP_DATA=1 ./setup.sh     # env only
#   WITH_AFDB=1 ./setup.sh     # also prepare AFDB representatives (large)
#   WITH_ESMATLAS=1 ./setup.sh # also prepare ESMAtlas representatives (very large)
#   source .venv/bin/activate
#
# Overridable via env:
#   PYTHON_VERSION (default 3.11), TORCH_CUDA (default cu128), ENV_DIR (.venv),
#   ENV_PROMPT (bitstream), SKIP_DATA, WITH_AFDB, WITH_ESMATLAS
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ENV_DIR="${ENV_DIR:-.venv}"
ENV_PROMPT="${ENV_PROMPT:-bitstream}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"                         # cu128 = Blackwell (sm_120)
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

# 1. uv --------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# 2. local venv ------------------------------------------------------------
if [ ! -d "$ENV_DIR" ]; then
  echo "[setup] creating uv venv '$ENV_DIR' (python $PYTHON_VERSION) ..."
  uv venv --python "$PYTHON_VERSION" --prompt "$ENV_PROMPT" "$ENV_DIR"
else
  echo "[setup] reusing existing venv '$ENV_DIR'"
  # Refresh the activation scripts and prompt without removing installed packages.
  uv venv --python "$PYTHON_VERSION" --allow-existing --prompt "$ENV_PROMPT" "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
echo "[setup] python: $(python --version) at $(command -v python)"

# 3. PyTorch (+ torchvision) from the CUDA 12.8 index (Blackwell support) ---
# torchvision is imported unconditionally by the trainer's viz/generation
# callbacks (make_grid), so it is required even for text/protein runs. Installing
# it from the same index keeps the CUDA build in lockstep with torch.
echo "[setup] installing torch + torchvision from $TORCH_INDEX ..."
uv pip install --index-url "$TORCH_INDEX" torch torchvision

# 4. All repo dependencies from pyproject.toml (the lock does not pin torch) --
# `uv sync` installs the resolved lock into .venv; --inexact keeps the
# separately-installed CUDA torch/torchvision that are intentionally not in the
# lock. The DPLM-inference and protein-eval stacks are separate environments
# (uv sync --extra dplm-inference / --extra protein-eval); see pyproject.toml.
echo "[setup] installing repo dependencies (train extra + dev) ..."
uv sync --extra train --group dev --inexact

# 5. Sanity: env correctness (imports + torch build) -----------------------
echo "[setup] verifying install ..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| built cuda:", torch.version.cuda)
archs = torch.cuda.get_arch_list()          # compiled arch list; no device init
assert "sm_120" in archs, f"torch build lacks Blackwell sm_120 support: {archs}"
print("arch list:", archs, "| Blackwell sm_120: OK")

# text-run stack
import transformers, datasets, tokenizers, mauve, gensim  # noqa: F401
print("transformers:", transformers.__version__, "| datasets:", datasets.__version__)

# repo import + protein path
import data  # noqa: F401
from data.proteins import make_char_tokenizer
print("protein char tokenizer vocab:", make_char_tokenizer().vocab_size)
print("repo imports: OK")
PY

# 6. GPU kernel check (isolated, non-fatal) --------------------------------
# Run on a single visible GPU so a wedged sibling GPU can't break the check via
# torch's NVML-vs-runtime device enumeration. This is a health probe, not a gate.
echo "[setup] GPU kernel probe (CUDA_VISIBLE_DEVICES=0) ..."
if CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
x = torch.randn(512, 512, device="cuda")
_ = (x @ x).sum().item()
print("cuda kernel test OK on:", torch.cuda.get_device_name(0))
PY
then :; else
  echo "[setup] WARN: single-GPU kernel probe failed. Check 'nvidia-smi' below;"
  echo "       a GPU marked '[GPU requires reset]' must be reset (sudo nvidia-smi -r -i <idx>) or rebooted."
fi
echo "[setup] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# 7. Data preparation (idempotent; composes per-dataset downloaders) --------
# Every downloader is smart: it checks for the artifact by path plus a hash or
# size or row-count and skips completed work, so this section is safe to re-run.
# Failures here are non-fatal (for example, no network) so the environment stays
# usable; a clear message is printed and setup continues.
run_prep() {
  local desc="$1"; shift
  echo "[setup] prepare: ${desc}"
  if "$@"; then
    echo "[setup] prepare OK: ${desc}"
  else
    echo "[setup] WARN: preparation step failed or was skipped: ${desc}"
    echo "        re-run './setup.sh' once the source is reachable; it resumes only the remaining work."
  fi
}

if [ "${SKIP_DATA:-0}" != "1" ]; then
  echo "[setup] preparing datasets (set SKIP_DATA=1 to skip) ..."

  # DPLM structure tokenizer (frozen codec; small).
  if [ -f scripts/proteins/setup/download_dplm_tokenizer.py ]; then
    run_prep "DPLM structure tokenizer" \
      python scripts/proteins/setup/download_dplm_tokenizer.py --out-dir datasets/dplm_struct_tokenizer
  fi

  # Sequence corpora: frozen EvoDiff UniRef50 and DiMA Swiss-Prot protocols.
  if [ -f scripts/proteins/setup/prepare_evodiff_uniref50.py ]; then
    run_prep "EvoDiff UniRef50 sequence corpus" \
      python scripts/proteins/setup/prepare_evodiff_uniref50.py --download
  fi
  if [ -f scripts/proteins/setup/prepare_dima_swissprot.py ]; then
    run_prep "DiMA Swiss-Prot sequence corpus" \
      python scripts/proteins/setup/prepare_dima_swissprot.py
  fi

  # M0 paired dataset (mandatory first result; ships with structure tokens).
  if [ -f scripts/proteins/setup/prepare_dplm_paired.py ]; then
    run_prep "M0 DPLM paired dataset" \
      python scripts/proteins/setup/prepare_dplm_paired.py --out-dir datasets/dplm_paired_m0
  fi

  # Gated large structure corpora (raw coordinates; must be tokenized offline).
  if [ "${WITH_AFDB:-0}" = "1" ] && [ -f scripts/proteins/setup/prepare_afdb_representatives.py ]; then
    run_prep "AFDB representatives (M1, large)" \
      python scripts/proteins/setup/prepare_afdb_representatives.py --enable --out-dir datasets/afdb_representatives
  fi
  if [ "${WITH_ESMATLAS:-0}" = "1" ] && [ -f scripts/proteins/setup/prepare_esmatlas.py ]; then
    run_prep "ESMAtlas representatives (M2, very large)" \
      python scripts/proteins/setup/prepare_esmatlas.py --enable --out-dir datasets/esmatlas_representatives
  fi
else
  echo "[setup] SKIP_DATA=1 set; skipping dataset preparation."
fi

echo ""
echo "[setup] done. Activate the environment with:"
echo "    source ${ENV_DIR}/bin/activate"
