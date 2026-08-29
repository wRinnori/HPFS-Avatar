import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def parse_vec3(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated values.")
    return np.array(vals, dtype=np.float32)


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32)
    expr = np.asarray(data["expressions"], dtype=np.float32) if "expressions" in data else None
    trans = np.asarray(data["trans"], dtype=np.float32) if "trans" in data else None

    if poses.ndim != 2 or poses.shape[1] < 165:
        raise ValueError(f"Expected poses shape [T, >=165], got {poses.shape}")
    if expr is None:
        expr = np.zeros((poses.shape[0], 100), dtype=np.float32)
    if trans is None:
        trans = np.zeros((poses.shape[0], 3), dtype=np.float32)

    return poses, expr, trans


def load_template_motion(template_motion_dir):
    template_motion_dir = Path(template_motion_dir)
    json_dir = template_motion_dir / "smplx_optimized" / "smplx_params_smoothed"
    json_files = sorted(json_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not json_files:
        raise RuntimeError(f"No template smplx json found under: {json_dir}")
    motions = []
    for p in json_files:
        with open(p) as f:
            motions.append(json.load(f))
    return motions


def template_index(out_i, out_len, template_len):
    if template_len <= 1:
        return 0
    if out_len <= 1:
        return 0
    ratio = out_i / float(out_len - 1)
    idx = int(round(ratio * (template_len - 1)))
    return max(0, min(idx, template_len - 1))


def make_cam_param(focal_x, focal_y, princpt_x, princpt_y):
    return {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": [0.0, 0.0, 0.0],
        "focal": [float(focal_x), float(focal_y)],
        "princpt": [float(princpt_x), float(princpt_y)],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert HPFS SMPL-X npz to ExAvatar motion_path format.")
    parser.add_argument("--npz", type=str, required=True, help="Input SMPL-X npz file.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output ExAvatar motion directory.")
    parser.add_argument("--camera-template-dir", type=str, default="", help="Optional directory containing ExAvatar cam_params/*.json to copy.")
    parser.add_argument("--template-motion-dir", type=str, default="", help="Optional ExAvatar template motion directory containing smplx_optimized/smplx_params_smoothed and cam_params.")
    parser.add_argument("--template-root-mode", type=str, default="none", choices=["none", "sequence", "first"], help="How to use template root pose: none=generated root pose, sequence=per-frame template root pose, first=use the first template frame root pose for all output frames.")
    parser.add_argument("--template-trans-mode", type=str, default="none", choices=["none", "sequence", "first"], help="How to use template translation: none=generated translation, sequence=per-frame template translation, first=use the first template frame translation for all output frames.")
    parser.add_argument("--expr-dim", type=int, default=50, help="Expression dimension expected by ExAvatar.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick tests.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-id-offset", type=int, default=0, help="Frame numbering offset in output json names.")
    parser.add_argument("--trans-scale", type=float, default=1.0, help="Scale applied to translation vectors.")
    parser.add_argument("--trans-offset", type=str, default="0,0,0", help="Comma-separated xyz offset added to translations.")
    parser.add_argument("--root-pose-offset", type=str, default="0,0,0", help="Comma-separated xyz offset added to root pose axis-angle.")
    parser.add_argument("--focal-x", type=float, default=2000.0)
    parser.add_argument("--focal-y", type=float, default=2000.0)
    parser.add_argument("--princpt-x", type=float, default=540.0)
    parser.add_argument("--princpt-y", type=float, default=960.0)
    args = parser.parse_args()

    poses, expr_all, trans_all = load_npz(args.npz)
    trans_offset = parse_vec3(args.trans_offset)
    root_pose_offset = parse_vec3(args.root_pose_offset)

    out_dir = Path(args.out_dir)
    smplx_dir = out_dir / "smplx_optimized" / "smplx_params_smoothed"
    smplx_raw_dir = out_dir / "smplx_optimized" / "smplx_params"
    cam_dir = out_dir / "cam_params"
    smplx_dir.mkdir(parents=True, exist_ok=True)
    smplx_raw_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = list(range(args.start_frame, poses.shape[0], max(args.frame_step, 1)))
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]

    if not frame_indices:
        raise RuntimeError("No frames selected.")

    template_motion = None
    if args.template_motion_dir:
        template_motion = load_template_motion(args.template_motion_dir)
        if not args.camera_template_dir:
            args.camera_template_dir = str(Path(args.template_motion_dir) / "cam_params")

    cam_template_files = []
    if args.camera_template_dir:
        cam_template_files = sorted(Path(args.camera_template_dir).glob("*.json"), key=lambda p: int(p.stem))

    default_cam = make_cam_param(args.focal_x, args.focal_y, args.princpt_x, args.princpt_y)

    for out_i, src_i in enumerate(frame_indices):
        pose = poses[src_i]
        expr = expr_all[src_i]
        trans = trans_all[src_i] * float(args.trans_scale) + trans_offset
        root_pose = pose[0:3] + root_pose_offset

        tmpl = None
        if template_motion is not None:
            if args.template_root_mode == "first" or args.template_trans_mode == "first":
                tmpl_first = template_motion[0]
            else:
                tmpl_first = None

            if args.template_root_mode == "sequence" or args.template_trans_mode == "sequence":
                tmpl = template_motion[template_index(out_i, len(frame_indices), len(template_motion))]

            if args.template_trans_mode == "sequence":
                trans = np.asarray(tmpl["trans"], dtype=np.float32)
            elif args.template_trans_mode == "first":
                trans = np.asarray(tmpl_first["trans"], dtype=np.float32)

            if args.template_root_mode == "sequence":
                root_pose = np.asarray(tmpl["root_pose"], dtype=np.float32)
            elif args.template_root_mode == "first":
                root_pose = np.asarray(tmpl_first["root_pose"], dtype=np.float32)

        frame_id = out_i + args.frame_id_offset
        payload = {
            "root_pose": np.asarray(root_pose, dtype=np.float32).tolist(),
            "body_pose": pose[3:66].reshape(21, 3).astype(np.float32).tolist(),
            "jaw_pose": pose[66:69].astype(np.float32).tolist(),
            "leye_pose": pose[69:72].astype(np.float32).tolist(),
            "reye_pose": pose[72:75].astype(np.float32).tolist(),
            "lhand_pose": pose[75:120].reshape(15, 3).astype(np.float32).tolist(),
            "rhand_pose": pose[120:165].reshape(15, 3).astype(np.float32).tolist(),
            "expr": expr[: args.expr_dim].astype(np.float32).tolist(),
            "trans": trans.astype(np.float32).tolist(),
        }

        with open(smplx_dir / f"{frame_id}.json", "w") as f:
            json.dump(payload, f)
        with open(smplx_raw_dir / f"{frame_id}.json", "w") as f:
            json.dump(payload, f)

        if cam_template_files:
            cam_src = cam_template_files[min(out_i, len(cam_template_files) - 1)]
            shutil.copy2(cam_src, cam_dir / f"{frame_id}.json")
        else:
            with open(cam_dir / f"{frame_id}.json", "w") as f:
                json.dump(default_cam, f)

    meta = {
        "source_npz": str(Path(args.npz).resolve()),
        "selected_src_frames": frame_indices,
        "output_frame_count": len(frame_indices),
        "template_motion_dir": args.template_motion_dir,
        "template_root_mode": args.template_root_mode,
        "template_trans_mode": args.template_trans_mode,
        "expr_dim": args.expr_dim,
        "trans_scale": args.trans_scale,
        "trans_offset": trans_offset.tolist(),
        "root_pose_offset": root_pose_offset.tolist(),
    }
    with open(out_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved ExAvatar motion directory to: {out_dir}")
    print(f"Frames written: {len(frame_indices)}")


if __name__ == "__main__":
    main()
