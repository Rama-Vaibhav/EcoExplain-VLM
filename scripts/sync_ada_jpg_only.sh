#!/bin/bash
# Push code + JPG chips to Ada WITHOUT .npy tensors (saves disk quota).
# VLM / explain only needs dataset.jsonl paths to *.jpg under data/processed_images/.
#
# Usage (from repo root on your Mac):
#   export ADA_HOST=cprv67ab97@ada-gw1
#   export ADA_REPO=~/EcoExplain-VLM
#   ./scripts/sync_ada_jpg_only.sh
#
# Optional: delete stray .npy on Ada first (SSH):
#   find ~/EcoExplain-VLM/data/processed_images -name '*.npy' -delete

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADA_HOST="${ADA_HOST:-cprv67ab97@ada-gw1}"
ADA_REPO="${ADA_REPO:-EcoExplain-VLM}"

echo "Sync repo (no processed_images bulk) -> ${ADA_HOST}:${ADA_REPO}/"
rsync -avz --progress \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'data/raw_tiffs' \
  --exclude 'data/processed_tensors' \
  --exclude 'data/processed_images' \
  --exclude 'results/checkpoints' \
  --exclude '__pycache__' \
  --exclude '*.npy' \
  "${ROOT}/" "${ADA_HOST}:${ADA_REPO}/"

echo "Sync JPGs only (no .npy) -> data/processed_images/"
rsync -avz --progress \
  --include '*/' \
  --include '*.jpg' \
  --include '*.jpeg' \
  --include '*.JPG' \
  --exclude '*' \
  "${ROOT}/data/processed_images/" "${ADA_HOST}:${ADA_REPO}/data/processed_images/"

echo "Done. On Ada: cd ${ADA_REPO} && find data/processed_images -name '*.jpg' | wc -l"
