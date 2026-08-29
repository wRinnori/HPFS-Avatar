#!/usr/bin/env bash
set -euo pipefail

# Copy this file to scripts/link_local_assets.sh, fill in the source paths below,
# and run it from the repository root. The created links are ignored by Git.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Motion-generation assets.
BEAT_H5_SRC="/path/to/BEAT_v2e_train.h5"
BEAT_SPLIT_SRC="/path/to/BEATv2_split"
BEAT_SMPL_SRC="/path/to/BEAT_SMPL/beat_english_v2.0.0"
BEAT_MEAN_SRC="/path/to/gesture_BEAT_mean_v2e.npy"
BEAT_STD_SRC="/path/to/gesture_BEAT_std_v2e.npy"
HPFS_FINAL_SRC="/path/to/hpfs_final.pt"
AUTOENCODER_SRC="/path/to/autoencoder_smplx.bin"

# Optional motion ablation/baseline checkpoints.
PFS_BASELINE_SRC=""
HPFS_NO_ROUTER_REG_SRC=""
HPFS_NO_PART_LOSS_SRC=""

# Avatar-rendering assets.
HUMAN_SMPLX_SRC="/path/to/human_model_files/smplx"
HUMAN_FLAME_SRC="/path/to/human_model_files/flame"
SUBJECT_ID="00058"
XHUMANS_SUBJECT_SRC="/path/to/XHumans/data/00058"
AVATAR_ID="00058"
AVATAR_BEST_SRC="/path/to/avatar/best.pth"

link_force() {
  local src="$1"
  local dst="$2"
  if [ -z "${src}" ]; then
    return 0
  fi
  if [ ! -e "${src}" ]; then
    echo "[WARN] source does not exist, skip: ${src}" >&2
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  ln -sfnT "${src}" "${dst}"
  echo "[OK] ${dst} -> ${src}"
}

link_force "${BEAT_H5_SRC}" "${ROOT}/motion_generation/data/BEAT_v2e_train.h5"
link_force "${BEAT_SPLIT_SRC}/train" "${ROOT}/motion_generation/data/BEATv2_split/train"
link_force "${BEAT_SPLIT_SRC}/val" "${ROOT}/motion_generation/data/BEATv2_split/val"
link_force "${BEAT_SPLIT_SRC}/test" "${ROOT}/motion_generation/data/BEATv2_split/test"
link_force "${BEAT_SMPL_SRC}" "${ROOT}/motion_generation/data/BEAT_SMPL/beat_english_v2.0.0"
link_force "${BEAT_MEAN_SRC}" "${ROOT}/motion_generation/data/stats/gesture_BEAT_mean_v2e.npy"
link_force "${BEAT_STD_SRC}" "${ROOT}/motion_generation/data/stats/gesture_BEAT_std_v2e.npy"
link_force "${HPFS_FINAL_SRC}" "${ROOT}/motion_generation/checkpoints/hpfs_final.pt"
link_force "${AUTOENCODER_SRC}" "${ROOT}/motion_generation/checkpoints/autoencoder_smplx.bin"
link_force "${PFS_BASELINE_SRC}" "${ROOT}/motion_generation/checkpoints/pfs_smplx_baseline.pt"
link_force "${HPFS_NO_ROUTER_REG_SRC}" "${ROOT}/motion_generation/checkpoints/hpfs_no_router_reg.pt"
link_force "${HPFS_NO_PART_LOSS_SRC}" "${ROOT}/motion_generation/checkpoints/hpfs_no_part_loss.pt"

mkdir -p "${ROOT}/motion_generation/data/BEAT_SMPL"
link_force "${HUMAN_SMPLX_SRC}" "${ROOT}/avatar_rendering/common/utils/human_model_files/smplx"
link_force "${HUMAN_FLAME_SRC}" "${ROOT}/avatar_rendering/common/utils/human_model_files/flame"
link_force "${XHUMANS_SUBJECT_SRC}" "${ROOT}/avatar_rendering/data/XHumans/data/${SUBJECT_ID}"
link_force "${AVATAR_BEST_SRC}" "${ROOT}/avatar_rendering/output/model_dump/${AVATAR_ID}/best.pth"

echo
echo "Run these checks next:"
echo "  python scripts/check_repro_assets.py --stage motion"
echo "  python scripts/check_repro_assets.py --stage avatar --subject-id ${SUBJECT_ID} --avatar-id ${AVATAR_ID}"
