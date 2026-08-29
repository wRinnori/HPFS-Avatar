#!/usr/bin/env python3
"""Validate the upload-ready HPFS_Avatar runtime without starting training."""
import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_import(name, package=None, required=True):
    package = package or name
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "imported")
        print(f"[OK] {name}: {version}")
        return True
    except Exception as exc:
        status = "MISSING" if required else "OPTIONAL-MISSING"
        print(f"[{status}] {name}: {type(exc).__name__}: {exc}")
        return not required


def run_probe(title, cwd, code):
    print(f"\n== {title} ==")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/hpfs_avatar_matplotlib")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {title} probe failed with exit code {proc.returncode}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--motion-root", default=None)
    parser.add_argument("--avatar-root", default=None)
    parser.add_argument("--skip-avatar-probe", action="store_true")
    parser.add_argument("--skip-cuda-extensions", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    motion_root = Path(args.motion_root).resolve() if args.motion_root else repo_root / "motion_generation"
    avatar_root = Path(args.avatar_root).resolve() if args.avatar_root else repo_root / "avatar_rendering"

    print("== Runtime ==")
    print(f"repo root: {repo_root}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")

    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"gpu[{idx}]: {torch.cuda.get_device_name(idx)}")

    print("\n== Core Imports ==")
    required = [
        ("torchvision", "torchvision"), ("numpy", "numpy"), ("scipy", "scipy"),
        ("sklearn", "sklearn"), ("pandas", "pandas"), ("yaml", "yaml"),
        ("easydict", "easydict"), ("configargparse", "configargparse"),
        ("omegaconf", "omegaconf"), ("einops", "einops"),
        ("h5py", "h5py"), ("librosa", "librosa"), ("soundfile", "soundfile"), ("cv2", "cv2"),
        ("torchmetrics", "torchmetrics"),
        ("smplx", "smplx"), ("pyrender", "pyrender"), ("trimesh", "trimesh"),
        ("lpips", "lpips"), ("timm", "timm"),
    ]
    ok = True
    for display, package in required:
        ok = check_import(display, package, required=True) and ok
    if not args.skip_cuda_extensions:
        ok = check_import("pytorch3d", "pytorch3d", required=True) and ok
        ok = check_import("diff_gaussian_rasterization_depth", "diff_gaussian_rasterization_depth", required=True) and ok
    check_import("open3d", "open3d", required=False)
    if not ok:
        raise SystemExit("[FAIL] Missing required imports above.")

    mm_code = rf'''
import os, sys
root = r"{motion_root}"
sys.path.insert(0, os.path.join(root, "mydiffusion"))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "model"))
from utils.model_util import create_gaussian_diffusion
from model.hpfs import HPFS
import train
import train_ae
import sample_linear
import custom2motion
import early_stop_hpfs
print("[OK] Motion-generation imports: model, diffusion util, train, train_ae, sample_linear, custom2motion, early_stop_hpfs")
'''
    run_probe("HPFS Motion-Generation Probe", motion_root / "mydiffusion", mm_code)

    converter_path = avatar_root / "tools" / "convert_hpfs_npz_to_exavatar_motion.py"
    if converter_path.exists():
        compile(converter_path.read_text(encoding="utf-8"), str(converter_path), "exec")
        print(f"[OK] Motion-to-avatar converter syntax: {converter_path.relative_to(repo_root)}")
    else:
        raise SystemExit(f"[FAIL] Missing motion-to-avatar converter: {converter_path}")

    avatar_entry_paths = [
        avatar_root / "main" / "train.py",
        avatar_root / "main" / "test.py",
        avatar_root / "main" / "animate_hpfs.py",
    ]
    for entry_path in avatar_entry_paths:
        if not entry_path.exists():
            raise SystemExit(f"[FAIL] Missing avatar entry script: {entry_path}")
        compile(entry_path.read_text(encoding="utf-8"), str(entry_path), "exec")
        print(f"[OK] Avatar entry syntax: {entry_path.relative_to(repo_root)}")

    if not args.skip_avatar_probe:
        asset_root = avatar_root / "common" / "utils" / "human_model_files"
        required_assets = [
            asset_root / "smplx" / "SMPLX_NEUTRAL.npz",
            asset_root / "smplx" / "SMPLX_MALE.npz",
            asset_root / "smplx" / "SMPL-X__FLAME_vertex_ids.npy",
            asset_root / "smplx" / "MANO_SMPLX_vertex_ids.pkl",
            asset_root / "flame" / "FLAME_NEUTRAL.pkl",
            asset_root / "flame" / "FLAME_texture.npz",
            asset_root / "flame" / "flame_static_embedding.pkl",
            asset_root / "flame" / "flame_dynamic_embedding.npy",
            asset_root / "flame" / "2019" / "generic_model.pkl",
        ]
        missing_assets = [str(x) for x in required_assets if not x.exists()]
        if missing_assets:
            print("\n== Avatar Rendering Probe ==")
            print("[SKIP] SMPL-X/FLAME assets are not present in the repository-local asset directory.")
            print("       Put licensed assets under avatar_rendering/common/utils/human_model_files/ to enable this probe.")
            for item in missing_assets:
                print(f"       missing: {item}")
        else:
            ex_code = rf'''
import os, sys
root = r"{avatar_root}"
sys.path.insert(0, os.path.join(root, "main"))
sys.path.insert(0, os.path.join(root, "common"))
import config
from config import cfg
from model import get_model
from nets.module import HumanGaussian, GaussianRenderer
import train
import test
import animate_hpfs
print("[OK] Avatar-rendering imports: config, model, renderer modules, train, test, animate_hpfs")
'''
            run_probe("Avatar Rendering Probe", avatar_root / "main", ex_code)

    print("\n[OK] HPFS_Avatar repository validation finished.")


if __name__ == "__main__":
    main()
