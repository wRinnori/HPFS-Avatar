#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "[1/5] Checking forbidden large artifact extensions..."
artifact_hit=$(find "$ROOT" -type f \( \
  -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.ckpt' -o -name '*.safetensors' -o \
  -name '*.h5' -o -name '*.hdf5' -o -name '*.npy' -o -name '*.npz' -o -name '*.pkl' -o -name '*.sav' -o \
  -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' -o \
  -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.gif' \
\) -print -quit)
if [ -n "$artifact_hit" ]; then
  echo "[FAIL] Found files that should not be uploaded:" >&2
  find "$ROOT" -type f \( \
    -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.ckpt' -o -name '*.safetensors' -o \
    -name '*.h5' -o -name '*.hdf5' -o -name '*.npy' -o -name '*.npz' -o -name '*.pkl' -o -name '*.sav' -o \
    -name '*.mp4' -o -name '*.avi' -o -name '*.mov' -o -name '*.mkv' -o \
    -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.gif' \
  \) >&2
  exit 1
fi

echo "[2/5] Checking generated local asset/output directories..."
local_dirs=(
  "$ROOT/avatar_rendering/common/utils/human_model_files"
  "$ROOT/avatar_rendering/output"
  "$ROOT/avatar_rendering/motion"
  "$ROOT/avatar_rendering/data/XHumans/data"
  "$ROOT/motion_generation/checkpoints"
  "$ROOT/motion_generation/data"
  "$ROOT/motion_generation/outputs"
  "$ROOT/motion_generation/mydiffusion/outputs"
  "$ROOT/third_party/pytorch3d"
  "$ROOT/third_party/diff-gaussian-rasterization-depth"
)
local_file_hits=""
for dir in "${local_dirs[@]}"; do
  if [ -d "$dir" ]; then
    hit=$(find "$dir" \( -type f -o -type l \) ! -name '.gitkeep' ! -name 'README.md' -print -quit)
    if [ -n "$hit" ]; then
      local_file_hits="${local_file_hits}${hit}"$'\n'
    fi
  fi
done
if [ -n "$local_file_hits" ]; then
  echo "[FAIL] Found local data, checkpoints, generated outputs, or extension source files that should not be uploaded:" >&2
  for dir in "${local_dirs[@]}"; do
    if [ -d "$dir" ]; then
      find "$dir" \( -type f -o -type l \) ! -name '.gitkeep' ! -name 'README.md' >&2
    fi
  done
  exit 1
fi

echo "[3/5] Checking Python cache files..."
cache_hit=$(find "$ROOT" \( -type d -name '__pycache__' -o -type f -name '*.pyc' \) -print -quit)
if [ -n "$cache_hit" ]; then
  echo "[FAIL] Found Python cache files that should not be uploaded:" >&2
  find "$ROOT" \( -type d -name '__pycache__' -o -type f -name '*.pyc' \) >&2
  exit 1
fi

echo "[4/5] Checking hard-coded local machine paths..."
local_path_hits=$(rg -n '/home/oem/Desktop|/home/oem/Downloads|Desktop/HPFS|Desktop/ExAvatar|Desktop/PantoMatrix|Desktop/TalkSHOW' "$ROOT" --glob '!UPLOAD_CONTENTS.md' --glob '!README.md' --glob '!docs/**' --glob '!scripts/check_upload_clean.sh' || true)
if [ -n "$local_path_hits" ]; then
  echo "[FAIL] Found hard-coded local paths." >&2
  echo "$local_path_hits" >&2
  exit 1
fi

echo "[5/5] Size summary:"
du -sh "$ROOT"
echo "[OK] Upload directory looks clean."
