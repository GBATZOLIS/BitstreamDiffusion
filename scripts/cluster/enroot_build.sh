#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Turn the Docker image into an enroot squashfs (.sqsh) for Slurm/pyxis on EOS.
#
# The squash must be produced per cluster -- a .sqsh is tied to the enroot
# version and filesystem it will run against, so do not copy one between
# clusters. Run this script ON EOS (registry path) or locally (dockerd path).
#
# Two paths, pick by whether you have a registry you can push to:
#
#   REGISTRY path (preferred, run on EOS):
#     local $ docker build -t bitstream:latest .
#     local $ docker tag bitstream:latest <registry>/<you>/bitstream:latest
#     local $ docker push <registry>/<you>/bitstream:latest
#     eos   $ IMAGE=<registry>/<you>/bitstream:latest scripts/cluster/enroot_build.sh
#
#   DOCKERD path (no registry; run locally, then ship the .sqsh):
#     local $ MODE=dockerd IMAGE=bitstream:latest scripts/cluster/enroot_build.sh
#     local $ rsync -av --progress bitstream.sqsh EOS-2:$DEST/
#
# Squashing is slow (tens of minutes for a multi-GB NGC image) and needs disk
# roughly equal to the image size -- do it in scratch, not $HOME.
# ---------------------------------------------------------------------------
set -euo pipefail

MODE="${MODE:-registry}"
IMAGE="${IMAGE:?set IMAGE, e.g. IMAGE=<registry>/<you>/bitstream:latest}"
SQSH="${SQSH:-bitstream.sqsh}"

if ! command -v enroot >/dev/null 2>&1; then
  echo "[enroot] enroot not found on this host." >&2
  echo "         On EOS it is usually behind a module: 'module load enroot'." >&2
  exit 1
fi

echo "[enroot] mode  : $MODE"
echo "[enroot] image : $IMAGE"
echo "[enroot] out   : $SQSH"

case "$MODE" in
  registry)
    # Credentials come from ~/.config/enroot/.credentials on the cluster.
    # For nvcr.io that file holds:  machine nvcr.io login $oauthtoken password <API key>
    enroot import -o "$SQSH" "docker://${IMAGE}"
    ;;
  dockerd)
    # Pulls straight from the local Docker daemon; no registry round-trip.
    enroot import -o "$SQSH" "dockerd://${IMAGE}"
    ;;
  *)
    echo "[enroot] unknown MODE '$MODE' (expected 'registry' or 'dockerd')" >&2
    exit 1
    ;;
esac

echo ""
echo "[enroot] built: $(du -h "$SQSH" | cut -f1)  $SQSH"
echo ""
echo "[enroot] smoke test on a compute node:"
cat <<'EOF'
    srun --account=<acct> --partition=<part> --gpus-per-node=1 --time=00:10:00 \
         --container-image=/path/to/bitstream.sqsh \
         --container-mounts=/lustre/.../bitstream_data:/workspace/data,$PWD:/workspace \
         --container-workdir=/workspace \
         python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
EOF
