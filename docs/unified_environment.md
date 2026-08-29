# Unified Environment

The upload-ready repository uses one Conda environment for both motion generation and avatar rendering: Python 3.10 with PyTorch 2.6.0 CUDA 12.4.

Use `environment.yml` on a clean machine, then install the avatar-rendering CUDA extensions with `scripts/install_avatar_cuda_extensions.sh` as described in the root `README.md`.

The runtime targets NVIDIA A40 GPUs. Building PyTorch3D and the depth-aware Gaussian rasterizer requires an `nvcc` toolkit matching the PyTorch CUDA build; with the provided requirements this means CUDA 12.4. Motion-generation imports work without local datasets. Avatar-rendering imports additionally require local SMPL-X/FLAME assets under `avatar_rendering/common/utils/human_model_files/`.
