#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pack datasets/ and runs/ into per-directory tarballs for bulk transfer.
#
# One tarball per top-level subdirectory rather than a single 150G blob: a
# failed or interrupted transfer then costs one directory, not everything, and
# the pieces can move in parallel. Each tarball gets a sha256 written alongside
# it so the far side can verify before unpacking.
#
# Compression is off by default. datasets/ and runs/ are dominated by already
# compact binary formats (checkpoints, tokenized arrays), where gzip costs hours
# of CPU for a few percent. Set COMPRESS=1 if your corpus is text-heavy.
#
# Usage:
#   scripts/cluster/tar_data.sh                    # both trees
#   scripts/cluster/tar_data.sh datasets           # one tree
#   OUT_DIR=/mnt/scratch/xfer scripts/cluster/tar_data.sh
#   COMPRESS=1 scripts/cluster/tar_data.sh datasets
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

OUT_DIR="${OUT_DIR:-$REPO_DIR/_transfer}"
COMPRESS="${COMPRESS:-0}"
TREES=("$@")
if [ ${#TREES[@]} -eq 0 ]; then
  TREES=(datasets runs)
fi

if [ "$COMPRESS" = "1" ]; then
  TAR_FLAGS="-czf"; EXT="tar.gz"
else
  TAR_FLAGS="-cf";  EXT="tar"
fi

mkdir -p "$OUT_DIR"
echo "[tar] output dir: $OUT_DIR (compress=$COMPRESS)"

for tree in "${TREES[@]}"; do
  if [ ! -d "$tree" ]; then
    echo "[tar] WARN: no such directory '$tree'; skipping."
    continue
  fi
  # Resolve symlinks one level: datasets/ entries are often symlinks into a
  # shared cache, and we want the contents, not a dangling link on the far side.
  for sub in "$tree"/*; do
    [ -e "$sub" ] || continue
    name="$(basename "$sub")"
    out="$OUT_DIR/${tree}__${name}.${EXT}"
    if [ -f "$out" ] && [ -f "$out.sha256" ]; then
      echo "[tar] skip (already packed): $out"
      continue
    fi
    echo "[tar] packing $sub -> $out"
    # -h dereferences symlinks; --sparse keeps sparse checkpoint files compact.
    tar -h --sparse $TAR_FLAGS "$out" -C "$tree" "$name"
    ( cd "$OUT_DIR" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
    echo "[tar] done: $(du -h "$out" | cut -f1)  $out"
  done
done

echo ""
echo "[tar] manifest:"
du -sh "$OUT_DIR"/*.${EXT} 2>/dev/null | sort -rh
echo ""
echo "[tar] total: $(du -sh "$OUT_DIR" | cut -f1)"
echo "[tar] next: scripts/cluster/transfer_eos.sh"
