# HPFS Avatar

Official PyTorch implementation of the HPFS Avatar pipeline for emotional SMPL-X motion generation and avatar rendering.

Please prepare the required datasets, body models, and checkpoints following the links and paths below.

## News

- First release: HPFS motion generation code, avatar rendering code, evaluation scripts, and reproduction helpers.

## 1. Setup Environment

The tested environment uses Python 3.10, PyTorch 2.6.0 + CUDA 12.4, and NVIDIA A40 GPUs.

```bash
cd /path/to/HPFS_Avatar
conda env create -f environment.yml
conda activate hpfs-avatar
```

Check the basic environment:

```bash
python scripts/validate_hpfs_avatar_env.py --skip-avatar-probe --skip-cuda-extensions
```

For avatar rendering, install PyTorch3D and the depth-aware Gaussian rasterizer after placing their source trees under `third_party/`:

```bash
cd /path/to/HPFS_Avatar
rm -rf third_party/pytorch3d
git clone --depth 1 --branch v0.7.8 https://github.com/facebookresearch/pytorch3d.git third_party/pytorch3d

# Download or copy the depth-aware rasterizer source to:
# third_party/diff-gaussian-rasterization-depth/setup.py

bash scripts/install_avatar_cuda_extensions.sh
python scripts/validate_hpfs_avatar_env.py --skip-avatar-probe
```

With the provided PyTorch CUDA 12.4 environment, the extension build requires a matching CUDA 12.4 `nvcc`. The installer defaults to `TORCH_CUDA_ARCH_LIST=8.6` for NVIDIA A40 GPUs.

## 2. Data Preparation

Download the required datasets and assets from the following sources:

- BEAT2 / BEATv2 English data: [H-Liu1997/BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2) and the [BEAT project](https://github.com/srymurphy/BEAT).
- X-Humans dataset: [official dataset page](https://xhumans.ait.ethz.ch/) and [X-Avatar repository](https://github.com/Skype-line/X-Avatar).
- SMPL-X model files: [SMPL-X official website](https://smpl-x.is.tuebingen.mpg.de/).
- FLAME model files: [FLAME official website](https://flame.is.tue.mpg.de/).
- WavLM-Large: [Microsoft UniLM WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) or [Hugging Face](https://huggingface.co/microsoft/wavlm-large).
- FastText English vectors: [fastText English word vectors](https://fasttext.cc/docs/en/english-vectors.html).

After preprocessing, place or symlink the files to the expected project layout:

```text
motion_generation/data/BEAT_v2e_train.h5
motion_generation/data/BEATv2_split/{train,val,test}/
motion_generation/data/BEAT_SMPL/beat_english_v2.0.0/
motion_generation/data/stats/gesture_BEAT_mean_v2e.npy
motion_generation/data/stats/gesture_BEAT_std_v2e.npy
avatar_rendering/common/utils/human_model_files/
avatar_rendering/data/XHumans/data/<subject_id>/
```

The avatar renderer expects the following licensed SMPL-X/FLAME files under `avatar_rendering/common/utils/human_model_files/`:

```text
smplx/SMPLX_NEUTRAL.npz
smplx/SMPLX_MALE.npz
smplx/SMPL-X__FLAME_vertex_ids.npy
smplx/MANO_SMPLX_vertex_ids.pkl
flame/FLAME_NEUTRAL.pkl
flame/FLAME_texture.npz
flame/flame_static_embedding.pkl
flame/flame_dynamic_embedding.npy
flame/2019/generic_model.pkl
```

For local use, edit the template below and create symbolic links instead of copying large files:

```bash
cp scripts/link_local_assets.template.sh scripts/link_local_assets.sh
# Fill in your local source paths.
bash scripts/link_local_assets.sh
```

Check whether the local files are ready:

```bash
python scripts/check_repro_assets.py --stage motion
python scripts/check_repro_assets.py --stage avatar --subject-id 00058 --avatar-id 00058
```

## 3. Pretrained Checkpoints

For direct testing, download the released checkpoints from [Google Drive](https://drive.google.com/drive/folders/1ElYtuSKnU7DtaxoVx32Ir9XWixmENBKx?usp=drive_link).

Place the downloaded files as follows:

```text
motion_generation/checkpoints/hpfs_final.pt
motion_generation/checkpoints/autoencoder_smplx.bin
avatar_rendering/output/model_dump/<avatar_id>/best.pth
```

If the avatar checkpoint is downloaded with a subject-specific name, copy or rename it to `best.pth` under the corresponding `<avatar_id>` folder before testing.

The released avatar-rendering demo checkpoint uses X-Humans subject `00058`; place it at `avatar_rendering/output/model_dump/00058/best.pth` and place the matching subject data at `avatar_rendering/data/XHumans/data/00058/`.

The SMPL-X autoencoder checkpoint is trained following the autoencoder training code and FGD-style evaluation protocol of MMoFusion, using the 265-D SMPL-X pose-and-expression representation in `motion_generation/mydiffusion/configs/autoencoder_smplx.yml`. It is used only for feature-space evaluation metrics such as FGD, Diversity, and L1div.

## 4. Test

### 4.1 Motion Generation

Edit the paths in `motion_generation/mydiffusion/configs/hpfs_final.yml` if needed, then run:

```bash
cd /path/to/HPFS_Avatar/motion_generation/mydiffusion
python sample_linear.py --config ./configs/hpfs_final.yml --gpu 0 --dataset BEAT
```

Metrics are saved to:

```text
motion_generation/mydiffusion/outputs/sample_dir_hpfs_final_whole/metrics_latest.json
```

### 4.2 Avatar Rendering

Place the released avatar checkpoint at `avatar_rendering/output/model_dump/00058/best.pth` and the matching X-Humans subject data at `avatar_rendering/data/XHumans/data/00058/`, then run:

```bash
cd /path/to/HPFS_Avatar/avatar_rendering/main
CUDA_VISIBLE_DEVICES=0 python test.py --subject_id 00058 --test_epoch best
```

Rendered test images and meshes are saved under `avatar_rendering/output/result/00058/test/`.

## 5. Train Motion Generation

Train HPFS from scratch:

```bash
cd /path/to/HPFS_Avatar/motion_generation/mydiffusion
python train.py --config ./configs/hpfs_final.yml --gpu 0 --dataset BEAT
```

Train with validation-based early stopping:

```bash
python early_stop_hpfs.py \
  --config ./configs/hpfs_final.yml \
  --gpu 0 --dataset BEAT \
  --val-path ../data/BEATv2_split/val \
  --chunk-steps 5000 --max-steps 300000 --save-iters 5000 \
  --patience 4 --min-delta 0.2 \
  --metric fgd --mode min \
  --sample-max-len 0 --randm-num 120 \
  --work-dir outputs/early_stop_work \
  --results-csv outputs/early_stop_hpfs.history.csv \
  --summary-json outputs/early_stop_hpfs.summary.json
```

Train the autoencoder used for FGD/Diversity/L1div evaluation:

```bash
python train_ae.py --config ./configs/autoencoder_smplx.yml --gpu 0 --dataset BEAT
```

## 6. Custom Motion Generation

Prepare custom audio/text features using the same folder convention as the processed BEAT data:

```text
motion_generation/data/custom_demo/audio_BEAT/
motion_generation/data/custom_demo/text_BEAT/
```

Generate SMPL-X motion:

```bash
cd /path/to/HPFS_Avatar/motion_generation/mydiffusion
python custom2motion.py --config ./configs/custom_smplx.yml --gpu 0 --dataset BEAT
```

## 7. Avatar Rendering

Train an avatar on X-Humans:

```bash
cd /path/to/HPFS_Avatar/avatar_rendering/main
CUDA_VISIBLE_DEVICES=0 python train.py --subject_id 00058
```

Convert generated HPFS SMPL-X `.npz` motion to the renderer motion format:

```bash
cd /path/to/HPFS_Avatar
python avatar_rendering/tools/convert_hpfs_npz_to_exavatar_motion.py \
  --npz motion_generation/mydiffusion/outputs/sample_dir_hpfs_final_whole/10_kieks_0_1_1_generated.npz \
  --out-dir avatar_rendering/motion/hpfs_demo \
  --max-frames 300
```

Render the avatar:

```bash
cd /path/to/HPFS_Avatar/avatar_rendering/main
CUDA_VISIBLE_DEVICES=0 python animate_hpfs.py \
  --avatar_id 00058 \
  --identity_id 00058 \
  --test_epoch best \
  --motion_path ../motion/hpfs_demo \
  --output_video ../output/result/hpfs_demo_00058.mp4 \
  --render_mode avatar \
  --save_frames_dir ../output/result/hpfs_demo_00058_frames
```

For a quick smoke test, convert only 10 frames and render at low resolution:

```bash
# Add to the conversion command: --max-frames 10 --frame-step 30
# Add to the rendering command: --render_height 256 --render_width 144
```

## 8. Release Check

Before uploading to GitHub, run:

```bash
bash scripts/check_upload_clean.sh
```
