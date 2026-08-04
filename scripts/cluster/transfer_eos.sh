#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push the packed tarballs to EOS over rsync.
#
# rsync (not scp) because these transfers run for hours and get interrupted:
# --partial --append-verify resumes a half-sent tarball instead of restarting
# it, and the ControlMaster in ~/.ssh/config means every invocation reuses one
# authenticated connection.
#
# This is the reliable, always-available path. EOS also offers a dedicated
# datamover for bulk staging, which is faster for 100G+ but whose exact
# interface you should confirm on the cluster (see scripts/cluster/datamover.sbatch).
# For a 150G payload over a login node, expect many hours -- run it under tmux.
#
# Usage:
#   scripts/cluster/transfer_eos.sh                 # all tarballs
#   scripts/cluster/transfer_eos.sh datasets__swissprot.tar
#   REMOTE=EOS-2 DEST=/lustre/fsw/portfolios/.../brunod/bitstream_data \
#     scripts/cluster/transfer_eos.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE="${REMOTE:-EOS-2}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/_transfer}"
# Set DEST to your EOS scratch/lustre path -- $HOME on EOS is small and is the
# wrong place for 150G. Find yours with: `sacctmgr -nP show assoc where user=$(whoami) format=account`
# and the matching /lustre/fsw/portfolios/<account>/users/$USER path.
DEST="${DEST:?set DEST to the EOS destination path, e.g. /lustre/fsw/portfolios/<acct>/users/brunod/bitstream_data}"

if [ ! -d "$OUT_DIR" ]; then
  echo "[xfer] no tarball dir at $OUT_DIR; run scripts/cluster/tar_data.sh first." >&2
  exit 1
fi

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
  mapfile -t FILES < <(cd "$OUT_DIR" && ls -- *.tar *.tar.gz 2>/dev/null || true)
fi
if [ ${#FILES[@]} -eq 0 ]; then
  echo "[xfer] nothing to transfer in $OUT_DIR" >&2
  exit 1
fi

echo "[xfer] remote : $REMOTE"
echo "[xfer] dest   : $DEST"
echo "[xfer] files  : ${#FILES[@]}"

ssh "$REMOTE" "mkdir -p '$DEST'"

for f in "${FILES[@]}"; do
  echo ""
  echo "[xfer] ==> $f"
  # --append-verify resumes an interrupted file by checksumming the common
  # prefix; safe here because tarballs are immutable once written.
  rsync -av --progress --partial --append-verify \
    "$OUT_DIR/$f" "$OUT_DIR/$f.sha256" \
    "$REMOTE:$DEST/"
done

echo ""
echo "[xfer] verifying checksums on $REMOTE ..."
ssh "$REMOTE" "cd '$DEST' && sha256sum -c ./*.sha256"

echo ""
echo "[xfer] transfer verified. Unpack on EOS with:"
echo "    ssh $REMOTE"
echo "    cd $DEST"
echo "    for t in datasets__*.tar; do tar -xf \"\$t\" -C <repo>/datasets; done"
echo "    for t in runs__*.tar;     do tar -xf \"\$t\" -C <repo>/runs;     done"
