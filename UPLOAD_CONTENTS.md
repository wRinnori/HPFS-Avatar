# Upload Contents

Upload the whole `HPFS_Avatar` directory as the GitHub repository.

## Included Code

```text
HPFS_Avatar/
├── README.md
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── check_repro_assets.py
│   ├── install_avatar_cuda_extensions.sh
│   ├── link_local_assets.template.sh
│   ├── check_upload_clean.sh
│   └── validate_hpfs_avatar_env.py
├── docs/
│   └── unified_environment.md
├── motion_generation/
│   ├── diffusion/
│   ├── model/
│   ├── mydiffusion/
│   │   ├── configs/
│   │   │   ├── hpfs_final.yml
│   │   │   ├── pfs_smplx_baseline.yml
│   │   │   ├── hpfs_no_router_reg.yml
│   │   │   ├── hpfs_no_part_loss.yml
│   │   │   ├── autoencoder_smplx.yml
│   │   │   └── custom_smplx.yml
│   │   ├── data_loader/
│   │   ├── optimizers/
│   │   ├── train.py
│   │   ├── train_ae.py
│   │   ├── sample_linear.py
│   │   ├── custom2motion.py
│   │   ├── early_stop_hpfs.py
│   │   └── smplx_motion_io.py
│   ├── train/
│   └── utils/
├── avatar_rendering/
│   ├── main/
│   │   └── animate_hpfs.py
│   ├── tools/
│   │   └── convert_hpfs_npz_to_exavatar_motion.py
│   ├── common/
│   ├── data/
│   │   └── XHumans/
│   │       └── XHumans.py
│   ├── motion/
│   └── output/
└── third_party/
```

## Excluded Local Artifacts

The repository intentionally excludes:

- trained checkpoints: `*.pt`, `*.pth`, `*.bin`, `*.ckpt`, `*.safetensors`
- processed datasets and arrays: `*.h5`, `*.npy`, `*.npz`, `*.pkl`, `*.sav`
- generated media/results: `*.mp4`, `*.png`, `*.jpg`, etc.
- local output directories: `motion_generation/outputs/`, `avatar_rendering/output/`
- licensed SMPL-X/FLAME assets: `avatar_rendering/common/utils/human_model_files/`
- optional third-party CUDA source trees under `third_party/`

## Local Placement Before Running

Put local-only files into the placeholder directories when running experiments:

```text
motion_generation/data/BEAT_v2e_train.h5
motion_generation/data/BEATv2_split/{train,val,test}/
motion_generation/data/BEAT_SMPL/beat_english_v2.0.0/
motion_generation/data/stats/gesture_BEAT_mean_v2e.npy
motion_generation/data/stats/gesture_BEAT_std_v2e.npy
motion_generation/checkpoints/hpfs_final.pt
motion_generation/checkpoints/autoencoder_smplx.bin
avatar_rendering/common/utils/human_model_files/
avatar_rendering/data/XHumans/data/<subject_id>/
avatar_rendering/output/model_dump/<avatar_id>/best.pth
avatar_rendering/motion/<motion_name>/
```
