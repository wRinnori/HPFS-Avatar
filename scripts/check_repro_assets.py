#!/usr/bin/env python3
"""Check local datasets, checkpoints, and licensed assets for HPFS_Avatar reproduction."""
import argparse
from pathlib import Path


def mark(path: Path, is_dir: bool = False) -> bool:
    ok = path.is_dir() if is_dir else path.is_file()
    tag = "OK" if ok else "MISSING"
    kind = "dir" if is_dir else "file"
    print(f"[{tag}] {kind}: {path}")
    return ok


def check_motion(root: Path) -> bool:
    print("\n== Motion Generation Assets ==")
    ok = True
    ok &= mark(root / "motion_generation" / "data" / "BEAT_v2e_train.h5")
    ok &= mark(root / "motion_generation" / "data" / "BEATv2_split" / "train", True)
    ok &= mark(root / "motion_generation" / "data" / "BEATv2_split" / "val", True)
    ok &= mark(root / "motion_generation" / "data" / "BEATv2_split" / "test", True)
    ok &= mark(root / "motion_generation" / "data" / "BEAT_SMPL" / "beat_english_v2.0.0", True)
    ok &= mark(root / "motion_generation" / "data" / "stats" / "gesture_BEAT_mean_v2e.npy")
    ok &= mark(root / "motion_generation" / "data" / "stats" / "gesture_BEAT_std_v2e.npy")
    ok &= mark(root / "motion_generation" / "checkpoints" / "hpfs_final.pt")
    ok &= mark(root / "motion_generation" / "checkpoints" / "autoencoder_smplx.bin")
    print("\nOptional ablation/baseline checkpoints:")
    mark(root / "motion_generation" / "checkpoints" / "pfs_smplx_baseline.pt")
    mark(root / "motion_generation" / "checkpoints" / "hpfs_no_router_reg.pt")
    mark(root / "motion_generation" / "checkpoints" / "hpfs_no_part_loss.pt")
    return ok


def check_avatar(root: Path, subject_id: str, avatar_id: str) -> bool:
    print("\n== Avatar Rendering Assets ==")
    asset_root = root / "avatar_rendering" / "common" / "utils" / "human_model_files"
    ok = True
    ok &= mark(asset_root / "smplx" / "SMPLX_NEUTRAL.npz")
    ok &= mark(asset_root / "smplx" / "SMPLX_MALE.npz")
    ok &= mark(asset_root / "smplx" / "SMPL-X__FLAME_vertex_ids.npy")
    ok &= mark(asset_root / "smplx" / "MANO_SMPLX_vertex_ids.pkl")
    ok &= mark(asset_root / "flame" / "FLAME_NEUTRAL.pkl")
    ok &= mark(asset_root / "flame" / "FLAME_texture.npz")
    ok &= mark(asset_root / "flame" / "flame_static_embedding.pkl")
    ok &= mark(asset_root / "flame" / "flame_dynamic_embedding.npy")
    ok &= mark(asset_root / "flame" / "2019" / "generic_model.pkl")
    ok &= mark(root / "avatar_rendering" / "data" / "XHumans" / "data" / subject_id, True)
    ok &= mark(root / "avatar_rendering" / "output" / "model_dump" / avatar_id / "best.pth")
    print("\nOptional custom motion directory for animation:")
    mark(root / "avatar_rendering" / "motion" / "example_motion", True)
    return ok


def check_third_party(root: Path) -> bool:
    print("\n== Third-Party CUDA Extension Sources ==")
    ok = True
    ok &= mark(root / "third_party" / "pytorch3d" / "setup.py")
    ok &= mark(root / "third_party" / "diff-gaussian-rasterization-depth" / "setup.py")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--stage", choices=["all", "motion", "avatar", "third_party"], default="all")
    parser.add_argument("--subject-id", default="00058")
    parser.add_argument("--avatar-id", default="00058")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    print(f"repo root: {root}")
    ok = True
    if args.stage in {"all", "motion"}:
        ok &= check_motion(root)
    if args.stage in {"all", "avatar"}:
        ok &= check_avatar(root, args.subject_id, args.avatar_id)
    if args.stage in {"all", "third_party"}:
        ok &= check_third_party(root)

    if ok:
        print("\n[OK] Required assets for the selected stage are present.")
    else:
        print("\n[MISSING] Some required files/directories are absent. Place/download them as documented in README.md.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
