#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs once after the devcontainer is created.
#
# Installs the agent CLIs and writes the Codex config that requests full auto
# run. The container is the isolation boundary that makes that mode appropriate:
# no host filesystem outside /workspace and the bound data dirs, and the repo is
# under version control so every change is reviewable.
#
# IMPORTANT: the sandbox_mode below is a *request*. NVIDIA enforces the policy
# server-side, so it only takes effect once your account is provisioned via the
# AI Agent Autorun Request. Verify with `/debug-config` inside Codex, and check
# ~/.codex/cloud-requirements-cache.json for allowed_sandbox_modes.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "[post-create] verifying environment ..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| built cuda:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
PY

echo "[post-create] installing agent CLIs ..."
if command -v npm >/dev/null 2>&1; then
  npm install -g @openai/codex @anthropic-ai/claude-code || \
    echo "[post-create] WARN: agent CLI install failed; install manually."
else
  echo "[post-create] WARN: npm not present in this base image; skipping."
fi

echo "[post-create] writing Codex full-auto config ..."
mkdir -p /root/.codex
cat > /root/.codex/config.toml <<'TOML'
# Full auto run. Valid here because the devcontainer is the approved isolated
# environment: no approvals, no Codex sandbox, scoped to /workspace.
sandbox_mode = "danger-full-access"
approval_policy = "never"

[history]
persistence = "save-all"
TOML

echo "[post-create] done."
echo "  python : $(command -v python)"
echo "  dplm   : /opt/venv-dplm/bin/python  (structure tokenizer / folding)"
echo "  codex  : codex   (verify mode with /status and /debug-config)"
