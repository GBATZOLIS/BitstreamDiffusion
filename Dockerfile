# ---------------------------------------------------------------------------
# BitstreamDiffusion container image.
#
# Base is the NGC PyTorch image, which already ships a CUDA-matched torch +
# torchvision build. That is the one thing setup.sh installs by hand (from the
# cu128 wheel index, for Blackwell sm_120); inside the container we inherit it
# from NGC instead and never reinstall torch, so the CUDA build always stays
# matched to the driver the image was built against.
#
# The venv at /opt/venv is created with --system-site-packages precisely so the
# NGC torch remains visible while uv manages everything else from uv.lock.
# `uv sync --inexact` then leaves those inherited packages alone.
#
# Build:
#   docker build -t bitstream:latest .
#   docker build --build-arg NGC_TAG=25.03-py3 -t bitstream:latest .
#
# NOTE: verify NGC_TAG against https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# and against the CUDA driver on the target cluster before a real build.
# ---------------------------------------------------------------------------
ARG NGC_TAG=25.03-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

# System tools. git/rsync/openssh-client are needed for the clone + transfer
# workflow; the rest are what the protein evaluation stack shells out to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget rsync openssh-client ca-certificates \
        build-essential tmux less vim jq unzip \
    && rm -rf /var/lib/apt/lists/*

# uv, pinned by the installer to /root/.local/bin.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# Dependency layer: copy only the resolver inputs so edits to source code do
# not invalidate the (slow) dependency install.
COPY pyproject.toml uv.lock ./

# --system-site-packages keeps NGC's torch/torchvision importable; --inexact
# stops uv from removing them as "not in the lock". This installs the default
# training/sampling stack (the `train` extra) plus the dev lint tooling.
RUN uv venv --system-site-packages --python "$(command -v python3)" /opt/venv \
    && uv sync --extra train --group dev --inexact --no-install-project

# The dplm-inference stack pins transformers==4.39.2 and numpy<2.0, which
# conflicts with `train` and cannot share an interpreter (see pyproject.toml
# [tool.uv] conflicts). It gets its own venv, also inheriting NGC torch.
RUN uv venv --system-site-packages --python "$(command -v python3)" /opt/venv-dplm \
    && UV_PROJECT_ENVIRONMENT=/opt/venv-dplm VIRTUAL_ENV=/opt/venv-dplm \
       uv sync --extra dplm-inference --inexact --no-install-project

# Source is bind-mounted in the devcontainer and baked in for batch/enroot use.
COPY . .

# Flat-layout research repo: imports resolve from the repo root, not a package.
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
