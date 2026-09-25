#!/usr/bin/env bash
# Fetch the released MNIST-Sum artifacts and put them where the configs expect.
#
#   bash scripts/multimodal/fetch_mnist_sum_release.sh            # eval bundle (~1 GB)
#   bash scripts/multimodal/fetch_mnist_sum_release.sh --full     # + the 1M-example training corpus
#
# Everything lands under the repository root:
#   runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt
#   runs/multimodal/mnist_sum_p28_12x512/entropy_{pdf,cdf,sigmas,edges}.pt
#   runs/mnist_sum_quadrant_cnn.pt
#   datasets/mnist_sum_p28/{meta.json,val,val_holdout}
set -euo pipefail

FOLDER_ID="${MNIST_SUM_FOLDER_ID:-1SH-2RQ6gzcTZOSe3wh2l-SXA-qi-J-CY}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$REPO_ROOT/runs/multimodal/mnist_sum_p28_12x512"
STAGE="${MNIST_SUM_STAGE:-$REPO_ROOT/.mnist_sum_release}"
WANT_FULL=0
[[ "${1:-}" == "--full" ]] && WANT_FULL=1

command -v gdown >/dev/null 2>&1 || { echo "gdown not found: python -m pip install gdown"; exit 1; }

echo "==> downloading release bundle (Google Drive folder $FOLDER_ID)"
mkdir -p "$STAGE"
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$STAGE"

BUNDLE="$STAGE/mnist_sum_p28_12x512"
[[ -d "$BUNDLE" ]] || BUNDLE="$STAGE"

echo "==> verifying checksums"
if [[ -f "$BUNDLE/SHA256SUMS" ]]; then
  ( cd "$BUNDLE" && sha256sum -c --ignore-missing SHA256SUMS ) \
    || { echo "CHECKSUM MISMATCH -- do not use these files; re-download."; exit 1; }
else
  echo "   ! SHA256SUMS not present in the download; skipping verification."
fi

echo "==> placing artifacts"
mkdir -p "$RUN_DIR/checkpoints" "$REPO_ROOT/runs" "$REPO_ROOT/datasets"
cp "$BUNDLE/checkpoints/step=000500000.pt" "$RUN_DIR/checkpoints/"
# The entropy tables MUST sit in the run directory: the sampler resolves them
# from the checkpoint's parent.parent and hard-errors if they are absent. See
# the note in the README -- a missing table is a silent 30-40 point regression
# in other CoBit experiments and is refused outright here.
cp "$BUNDLE"/entropy_{pdf,cdf,sigmas,edges}.pt "$RUN_DIR/"
cp "$BUNDLE/config.json" "$BUNDLE/original_config.py" "$RUN_DIR/" 2>/dev/null || true
# One classifier across all arms: text->image and the image half of joint are
# read by THIS file. Retraining it makes your numbers incomparable to the paper.
cp "$BUNDLE/mnist_sum_quadrant_cnn.pt" "$REPO_ROOT/runs/"

echo "==> unpacking the evaluation corpus (val + val_holdout, ~11 MB on disk)"
tar xzf "$BUNDLE/mnist_sum_p28_eval.tar.gz" -C "$REPO_ROOT/datasets"

if [[ "$WANT_FULL" == "1" ]]; then
  if [[ -f "$BUNDLE/mnist_sum_p28_full.tar.gz" ]]; then
    echo "==> unpacking the full 1M-example corpus (~510 MB on disk)"
    tar xzf "$BUNDLE/mnist_sum_p28_full.tar.gz" -C "$REPO_ROOT/datasets"
  else
    echo "   ! mnist_sum_p28_full.tar.gz not in the bundle; rebuild it instead with"
    echo "     python scripts/multimodal/phase_50_mnist_sum_dataset.py --patch-size 28"
  fi
fi

echo
echo "done. Placed:"
echo "  $RUN_DIR/checkpoints/step=000500000.pt"
echo "  $RUN_DIR/entropy_{pdf,cdf,sigmas,edges}.pt"
echo "  $REPO_ROOT/runs/mnist_sum_quadrant_cnn.pt"
echo "  $REPO_ROOT/datasets/mnist_sum_p28/"
echo
echo "next:"
echo "  python scripts/multimodal/eval_mnist_sum_offline.py \\"
echo "      --config configs/multimodal/mnist_sum_p28_small.py \\"
echo "      --ckpt   $RUN_DIR/checkpoints/step=000500000.pt \\"
echo "      --out    results/mnist_sum_p28_500k --n 4096 --n_joint 4096"
echo "  python scripts/multimodal/check_mnist_sum_results.py results/mnist_sum_p28_500k/results.json"
