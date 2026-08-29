#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTORCH3D_ROOT="${PYTORCH3D_ROOT:-${REPO_ROOT}/third_party/pytorch3d}"
DGR_DEPTH_ROOT="${DGR_DEPTH_ROOT:-${REPO_ROOT}/third_party/diff-gaussian-rasterization-depth}"
TORCH_CUDA_VERSION=$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)
MATCHED_CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VERSION}"
USER_CUDA_HOME="${CUDA_HOME:-}"
if [ -n "${TORCH_CUDA_VERSION}" ] && [ -d "${MATCHED_CUDA_HOME}" ] && [ -z "${HPFS_RESPECT_CUDA_HOME:-}" ]; then
  CUDA_HOME="${MATCHED_CUDA_HOME}"
else
  CUDA_HOME="${CUDA_HOME:-${MATCHED_CUDA_HOME}}"
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
MAX_JOBS="${MAX_JOBS:-8}"

export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS FORCE_CUDA=1
if [ -z "${CUDA_HOME}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "[ERROR] nvcc was not found under CUDA_HOME=${CUDA_HOME:-<empty>}." >&2
  echo "Install the CUDA toolkit that matches torch.version.cuda=${TORCH_CUDA_VERSION}, or set CUDA_HOME explicitly." >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [ -n "${USER_CUDA_HOME}" ] && [ "${USER_CUDA_HOME}" != "${CUDA_HOME}" ]; then
  echo "[WARN] Override CUDA_HOME=${USER_CUDA_HOME} with ${CUDA_HOME} to match PyTorch CUDA ${TORCH_CUDA_VERSION}."
  echo "       Set HPFS_RESPECT_CUDA_HOME=1 if you intentionally want to keep your own CUDA_HOME."
fi
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] CUDA_HOME: ${CUDA_HOME}"
echo "[INFO] TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

if [ ! -f "${PYTORCH3D_ROOT}/setup.py" ]; then
  echo "[ERROR] PyTorch3D source not found at ${PYTORCH3D_ROOT}" >&2
  echo "Clone or copy PyTorch3D into third_party/pytorch3d, or set PYTORCH3D_ROOT." >&2
  exit 1
fi

if [ ! -f "${DGR_DEPTH_ROOT}/setup.py" ]; then
  echo "[ERROR] depth Gaussian rasterizer source not found at ${DGR_DEPTH_ROOT}" >&2
  echo "Clone or copy it into third_party/diff-gaussian-rasterization-depth, or set DGR_DEPTH_ROOT." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -v --no-build-isolation "${PYTORCH3D_ROOT}"
"${PYTHON_BIN}" -m pip install -v --no-build-isolation "${DGR_DEPTH_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import importlib
for name in ["pytorch3d", "diff_gaussian_rasterization_depth"]:
    mod = importlib.import_module(name)
    print(f"[OK] {name}: {getattr(mod, '__version__', 'imported')}")
PY
