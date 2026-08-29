"""Lightweight SMPL-X motion conversion utilities used by evaluation/inference.

This module keeps the upload package independent from the legacy BVH preprocessing
stack. It contains only the functions needed by SMPL-X testing, metric computation,
and optional npz export.
"""
import os

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

ORDER = "XYZ"


def pose2euler(poses, gt=False, fixroot=False, normalize_root=False, upper_body=False):
    """Convert legacy rotation-matrix motion [T, J*9] to Euler degrees [T, J*3]."""
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]), dtype=np.float32)

    if not gt:
        for i in range(poses.shape[1]):
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)
    else:
        out_poses = poses

    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3), dtype=np.float32)
    for i in range(out_poses.shape[0]):
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler(ORDER, degrees=True).flatten()

    if not gt and fixroot:
        out_euler[:, :3] = 0.0

    if not gt and normalize_root:
        pos_mean = np.mean(out_euler[:, :3], axis=0)
        pos_std = np.std(out_euler[:, :3], axis=0)
        pos_std = np.where(pos_std < 1e-8, 1.0, pos_std)
        out_euler[:, :3] = (out_euler[:, :3] - pos_mean) / pos_std

    if not gt and upper_body:
        root_euler = np.expand_dims(out_euler[0, 0:6], 0).repeat(out_euler.shape[0], axis=0)
        first_euler = np.expand_dims(out_euler[0, 192:], 0).repeat(out_euler.shape[0], axis=0)
        out_euler[:, 0:6] = root_euler
        out_euler[:, 192:] = first_euler

    return out_euler


def pose2bvh(*_args, **_kwargs):
    raise NotImplementedError("BVH export is not included in the compact SMPL-X release package.")


def pose2bvh_style(*_args, **_kwargs):
    raise NotImplementedError("BVH export is not included in the compact SMPL-X release package.")


def smplx_pose_to_legacy_matrix(poses, target_dim=684):
    """Convert SMPL-X axis-angle pose [T, J*3] to legacy rotmat layout [T, target_dim]."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] % 3 != 0:
        raise ValueError(f"Unexpected SMPL-X pose shape: {poses.shape}")

    n_frames = poses.shape[0]
    rot_mats = R.from_rotvec(poses.reshape(-1, 3)).as_matrix().astype(np.float32)
    rot_mats = rot_mats.reshape(n_frames, -1, 9).reshape(n_frames, -1)

    root_slot = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (n_frames, 1))
    legacy = np.concatenate([root_slot, rot_mats], axis=1)

    if legacy.shape[1] < target_dim:
        pad_dim = target_dim - legacy.shape[1]
        if pad_dim % 9 != 0:
            raise ValueError(f"target_dim({target_dim}) must be a multiple of 9.")
        pad_joints = pad_dim // 9
        pad = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (n_frames, pad_joints))
        legacy = np.concatenate([legacy, pad], axis=1)
    elif legacy.shape[1] > target_dim:
        legacy = legacy[:, :target_dim]

    return legacy.astype(np.float32)


def smplx_pose_to_legacy_euler(poses, gt=False, fixroot=False, normalize_root=False, upper_body=False, target_dim=684):
    legacy_motion = smplx_pose_to_legacy_matrix(poses, target_dim=target_dim)
    return pose2euler(
        legacy_motion,
        gt=gt,
        fixroot=fixroot,
        normalize_root=normalize_root,
        upper_body=upper_body,
    )


def _orthonormalize_rotmats(rotmats):
    u, _, vt = np.linalg.svd(rotmats)
    r = u @ vt
    det = np.linalg.det(r)
    neg = det < 0
    if np.any(neg):
        u[neg, :, -1] *= -1.0
        r = u @ vt
    return r


def legacy_matrix_to_smplx_pose(legacy_motion, smplx_joint_num=55):
    legacy_motion = np.asarray(legacy_motion)
    if legacy_motion.ndim != 2:
        raise ValueError(f"legacy_motion should be 2D, got shape {legacy_motion.shape}")
    if legacy_motion.shape[1] % 9 != 0:
        raise ValueError(f"legacy_motion dim should be multiple of 9, got {legacy_motion.shape[1]}")

    t = legacy_motion.shape[0]
    mats = legacy_motion.reshape(t, -1, 9).reshape(t, -1, 3, 3)
    need = 1 + smplx_joint_num
    if mats.shape[1] < need:
        raise ValueError(f"legacy joint count {mats.shape[1]} < required {need}")
    smplx_mats = mats[:, 1:need].reshape(-1, 3, 3)
    smplx_mats = _orthonormalize_rotmats(smplx_mats)
    smplx_pose = R.from_matrix(smplx_mats).as_rotvec().reshape(t, smplx_joint_num * 3)
    return smplx_pose.astype(np.float32)


def _resize_time(arr, target_len):
    arr = np.asarray(arr)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)
    x_old = np.linspace(0, 1, arr.shape[0], dtype=np.float64)
    x_new = np.linspace(0, 1, target_len, dtype=np.float64)
    arr_2d = arr.reshape(arr.shape[0], -1)
    out = np.zeros((target_len, arr_2d.shape[1]), dtype=np.float64)
    for i in range(arr_2d.shape[1]):
        out[:, i] = np.interp(x_new, x_old, arr_2d[:, i])
    return out.reshape((target_len,) + arr.shape[1:]).astype(arr.dtype)


def _load_ref_meta(ref_npz_path, target_len, exp_dim=100):
    betas = np.zeros((300,), dtype=np.float32)
    expressions = np.zeros((target_len, exp_dim), dtype=np.float32)
    trans = np.zeros((target_len, 3), dtype=np.float32)
    model_name = "smplx2020"
    gender = "neutral"
    mocap_frame_rate = 30

    if ref_npz_path is not None and os.path.exists(ref_npz_path):
        ref = np.load(ref_npz_path, allow_pickle=True)
        if "betas" in ref:
            ref_betas = ref["betas"]
            betas = ref_betas.astype(np.float32) if ref_betas.ndim == 1 else ref_betas[0].astype(np.float32)
        if "expressions" in ref:
            expressions = _resize_time(ref["expressions"].astype(np.float32), target_len)
            if expressions.shape[1] < exp_dim:
                pad = np.zeros((target_len, exp_dim - expressions.shape[1]), dtype=np.float32)
                expressions = np.concatenate([expressions, pad], axis=1)
            expressions = expressions[:, :exp_dim]
        if "trans" in ref:
            trans = _resize_time(ref["trans"].astype(np.float32), target_len)
        if "model" in ref:
            model_name = str(ref["model"])
        if "gender" in ref:
            gender = str(ref["gender"])
        if "mocap_frame_rate" in ref:
            try:
                mocap_frame_rate = int(ref["mocap_frame_rate"])
            except Exception:
                mocap_frame_rate = 30

    return betas, expressions, trans, model_name, gender, mocap_frame_rate


def save_smplx_npz_from_legacy(out_npz_path, legacy_motion, ref_npz_path=None):
    pose = legacy_matrix_to_smplx_pose(legacy_motion)
    t = pose.shape[0]
    betas, expressions, trans, model_name, gender, mocap_frame_rate = _load_ref_meta(ref_npz_path, t, exp_dim=100)
    np.savez(
        out_npz_path,
        betas=betas,
        poses=pose,
        expressions=expressions,
        trans=trans,
        model=model_name,
        gender=gender,
        mocap_frame_rate=np.int32(mocap_frame_rate),
    )


def save_smplx_npz_from_poseexp(out_npz_path, poseexp_motion, pose_dim=165, exp_dim=100, ref_npz_path=None):
    arr = np.asarray(poseexp_motion, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"poseexp_motion should be 2D, got shape {arr.shape}")
    if arr.shape[1] < pose_dim + exp_dim:
        raise ValueError(
            f"poseexp_motion dim {arr.shape[1]} is smaller than pose_dim+exp_dim ({pose_dim + exp_dim})"
        )

    t = arr.shape[0]
    pose = arr[:, :pose_dim].astype(np.float32)
    expressions = arr[:, pose_dim:pose_dim + exp_dim].astype(np.float32)
    betas, _, trans, model_name, gender, mocap_frame_rate = _load_ref_meta(ref_npz_path, t, exp_dim=exp_dim)
    np.savez(
        out_npz_path,
        betas=betas,
        poses=pose,
        expressions=expressions,
        trans=trans,
        model=model_name,
        gender=gender,
        mocap_frame_rate=np.int32(mocap_frame_rate),
    )
