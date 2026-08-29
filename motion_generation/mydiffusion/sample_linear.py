import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../model']]
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from utils import metric, other_tools, data_tools
# import subprocess
import os
import json
from datetime import datetime
import copy
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from smplx_motion_io import (
    pose2bvh,
    pose2euler,
    smplx_pose_to_legacy_euler,
    save_smplx_npz_from_legacy,
    save_smplx_npz_from_poseexp,
)
import argparse
from model.cfg_sampler import ClassifierFreeSampleModel
from model.motion_autoencoder import HalfEmbeddingNet

def create_model_and_diffusion(args):
    model_name = str(getattr(args, "model_name", "")).lower()
    if model_name.startswith(('hpfs', 'pfs')):
        from model.hpfs import HPFS
        print(f'use HPFS motion model ({args.model_name})')
    else:
        raise ValueError(
            f"Unsupported model_name: {getattr(args, 'model_name', None)}. "
            f"Expected a name prefixed with 'hpfs' or 'pfs'."
        )

    model = HPFS(args, modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=args.device_name,
                style_dim=args.speaker_num, emo_dim=args.emo_dim, num_layers=args.num_layers, num_heads=args.num_heads, source_audio_dim=args.audio_feature_dim, 
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def merge_subdivisions(sample_seq, stride, seed, device):
    """
    Merge overlapped diffusion chunks with linear cross-fade.
    sample_seq: [num_subdivision, seq_len, dim]
    """
    b, s, c = sample_seq.shape
    if b == 1:
        return sample_seq[0]

    fade_f = torch.linspace(1, 0, seed, device=device)[None, :, None]
    fade_b = torch.linspace(0, 1, seed, device=device)[None, :, None]
    left, right = sample_seq[:-1, stride:, :], sample_seq[1:, :seed, :]
    merged = fade_f * left + fade_b * right

    full = torch.zeros((s + stride * (b - 1), c), device=device)
    full[:stride] += sample_seq[0, :stride]
    idx = stride
    for it, q_slice in enumerate(merged, start=1):
        full[idx:idx + seed] += q_slice
        full[idx + seed:idx + stride] += sample_seq[it, seed:stride]
        idx += stride
    full[idx:idx + seed] += sample_seq[-1, :seed]
    return full


def _extract_smplx_arm_triplets_deg(poseexp, pose_dim=165):
    """
    Build 6-joint arm rotations for SRGR/BeatAlign from SMPL-X pose+exp.
    Output shape: [T, 18] in degrees, ordered as:
    [R_shoulder, R_elbow, R_wrist, L_shoulder, L_elbow, L_wrist], each xyz.
    """
    arr = np.asarray(poseexp, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D motion array, got shape {arr.shape}")
    if arr.shape[1] < pose_dim:
        raise ValueError(f"Motion dim {arr.shape[1]} is smaller than pose_dim={pose_dim}")
    if pose_dim % 3 != 0:
        raise ValueError(f"pose_dim must be divisible by 3, got {pose_dim}")

    pose = arr[:, :pose_dim]
    num_joints = pose_dim // 3
    pose_j = pose.reshape(-1, num_joints, 3)

    # SMPL-X joint ids in the 55-joint pose order.
    # 16/17: shoulders, 18/19: elbows, 20/21: wrists.
    joint_ids = [17, 19, 21, 16, 18, 20]
    if max(joint_ids) >= num_joints:
        raise ValueError(
            f"pose joint count {num_joints} is not enough for arm ids {joint_ids}. "
            f"Current pose_dim={pose_dim}"
        )

    arm = pose_j[:, joint_ids, :].reshape(pose_j.shape[0], -1)
    return np.rad2deg(arm).astype(np.float32)


def _extract_smplx_joint_set_deg(poseexp, joint_ids, pose_dim=165):
    """Extract selected SMPL-X joints (xyz) and convert radians to degrees."""
    arr = np.asarray(poseexp, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D motion array, got shape {arr.shape}")
    if arr.shape[1] < pose_dim:
        raise ValueError(f"Motion dim {arr.shape[1]} is smaller than pose_dim={pose_dim}")
    if pose_dim % 3 != 0:
        raise ValueError(f"pose_dim must be divisible by 3, got {pose_dim}")

    pose = arr[:, :pose_dim]
    num_joints = pose_dim // 3
    pose_j = pose.reshape(-1, num_joints, 3)
    if max(joint_ids) >= num_joints:
        raise ValueError(
            f"pose joint count {num_joints} is not enough for joint ids {joint_ids}. "
            f"Current pose_dim={pose_dim}"
        )
    joints = pose_j[:, joint_ids, :].reshape(pose_j.shape[0], -1)
    return np.rad2deg(joints).astype(np.float32)


def _resolve_smplx_srgr_joint_ids(joint_set, pose_dim=165):
    """
    Resolve preset joint sets for SRGR-like metric on SMPL-X.
    - arm6: [R/L shoulder, elbow, wrist]
    - upper14: upper body torso+neck+arms
    - body22: SMPL-X body joints 0..21
    - all55: all pose joints in pose_dim
    """
    name = str(joint_set or "upper14").lower()
    if name == "arm6":
        ids = [17, 19, 21, 16, 18, 20]
    elif name == "upper14":
        ids = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    elif name == "body22":
        ids = list(range(22))
    elif name == "all55":
        ids = list(range(pose_dim // 3))
    else:
        raise ValueError(
            f"Unsupported smplx_srgr_paper_joint_set={joint_set}. "
            f"Use one of: arm6, upper14, body22, all55."
        )
    return ids, name


def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, mydevice=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start
    eval_model = None
    if args.e_name is not None:
        eval_model_module = __import__(f"model.{args.eval_model}", fromlist=["something"])
        eval_model = getattr(eval_model_module, args.e_name)(args)
        other_tools.load_checkpoints(eval_model, args.e_path, args.e_name)  
        eval_model.to(mydevice)
        eval_model.eval()
            
    tst_audio_dir = os.path.join(tst_path, 'audio_' + dataset)
    tst_wav_dir = os.path.join(tst_path, 'wav_' + dataset)
    tst_text_dir = os.path.join(tst_path, 'text_' + dataset)
    tst_gesture_dir = os.path.join(tst_path, 'gesture_' + dataset)
    tst_sem_dir = os.path.join(tst_path, 'sem_' + dataset)
    tst_emo_dir = os.path.join(tst_path, 'emo_' + dataset)
    motion_repr = getattr(args, 'motion_repr', None) or 'legacy'
    use_smplx_repr = (str(motion_repr).lower() == 'smplx')
    save_npz = bool(getattr(args, 'save_npz', False))
    smplx_ref_dir = getattr(args, 'smplx_ref_dir', None)
    smplx_pose_dim = int(getattr(args, 'smplx_pose_dim', None) or 165)
    smplx_exp_dim = int(getattr(args, 'smplx_exp_dim', None) or 100)
    alignmenter = metric.alignment(0.3, 2)
    if use_smplx_repr:
        pose_joints = 6
        srgr_threshold = float(getattr(args, 'smplx_srgr_threshold', 4.0))
        srgr_paper_threshold = float(getattr(args, 'smplx_srgr_paper_threshold', srgr_threshold))
        srgr_paper_joint_ids, srgr_paper_joint_set = _resolve_smplx_srgr_joint_ids(
            getattr(args, 'smplx_srgr_paper_joint_set', 'upper14'),
            pose_dim=smplx_pose_dim,
        )
        srgr_paper_calculator = metric.SRGR(srgr_paper_threshold, len(srgr_paper_joint_ids))
        srgr_legacy_like_calculator = metric.SRGR(4.0, 76)
        srgr_legacy_like_upper_calculator = metric.SRGR(4.0, 62)
    elif args.body == 'upper':
        pose_joints = 62
        srgr_threshold = 4.0
        srgr_paper_threshold = srgr_threshold
        srgr_paper_joint_set = "legacy_same_as_srgr"
        srgr_paper_joint_ids = None
        srgr_paper_calculator = None
        srgr_legacy_like_calculator = None
        srgr_legacy_like_upper_calculator = None
    else:
        pose_joints = 76
        srgr_threshold = 4.0
        srgr_paper_threshold = srgr_threshold
        srgr_paper_joint_set = "legacy_same_as_srgr"
        srgr_paper_joint_ids = None
        srgr_paper_calculator = None
        srgr_legacy_like_calculator = None
        srgr_legacy_like_upper_calculator = None
    srgr_calculator = metric.SRGR(srgr_threshold, pose_joints)
    l1_calculator = metric.L1div()
    t_start = 10
    t_end = 500
    align = 0 
    pose_fps = 30
    randm_num = args.randm_num

    def resolve_ref_npz(_filename):
        if not smplx_ref_dir:
            return None
        cand1 = os.path.join(smplx_ref_dir, "smplxflame_30", _filename + ".npz")
        cand2 = os.path.join(smplx_ref_dir, _filename + ".npz")
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
        return None

    test_files = sorted(os.listdir(tst_gesture_dir))
    if tst_prefix:
        test_files = [f for f in test_files if any(f.startswith(p) for p in tst_prefix)]
    if max_len and max_len > 0:
        test_files = test_files[:max_len]
    if len(test_files) == 0:
        raise ValueError(f"No test files found in {tst_gesture_dir} after applying tst_prefix/max_len filters.")
    print(f"Total test files to process: {len(test_files)}")

    processed_samples = 0

    for its, file in enumerate(test_files):
        filename = file.split(".")[0]
        print(f"Processing: {filename}, i={its}")
        if dataset == 'BEAT':
            speaker_id = int(filename.split('_')[0]) - 1
        else:
            speaker_id = 0
            
        audio_path = os.path.join(tst_audio_dir, filename + '.npy')
        audio = np.load(audio_path)
        wav_path = os.path.join(tst_wav_dir, filename + '.npy')
        wav = np.load(wav_path)
        text_path = os.path.join(tst_text_dir, filename + '.npy')
        text = np.load(text_path)
        sem_path = os.path.join(tst_sem_dir, filename + '.npy')
        sem = np.load(sem_path)
        emo_path = os.path.join(tst_emo_dir, filename + '.npy')
        emo = np.load(emo_path)
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)

        gesture_path = os.path.join(tst_gesture_dir, filename + '.npy')
        gesture = np.load(gesture_path)

        emotion = torch.from_numpy(emo).to(mydevice)
        
        seed=123456
        torch.manual_seed(seed)

        style = int(speaker_id)

        n_frames = textaudio.shape[0]

        if n_frames >=6000:
            n_frames = 6000
            textaudio = textaudio[:n_frames]

        real_n_frames = copy.deepcopy(n_frames)   
        # stride_poses = args.n_poses + args.n_seed
        stride_poses = args.n_poses
        stride = args.stride

        if n_frames < stride_poses:
            num_subdivision = 1
            n_frames = stride_poses
        else:
            num_subdivision = math.ceil(n_frames / stride)
            n_frames = num_subdivision * stride
            print('real_n_frames: {}, num_subdivision: {}, stride_poses: {}, stride: {}, pad_frames: {}, speaker_id: {}'.format(real_n_frames, num_subdivision, stride_poses, stride, n_frames - real_n_frames + args.n_seed, style))
        
        textaudio_pad = torch.zeros([n_frames - real_n_frames + args.n_seed, args.audio_feature_dim]).to(mydevice)
        textaudio = torch.cat((textaudio, textaudio_pad), 0)
        textaudio_shape = textaudio.shape[0]
        if emotion.shape[0] < textaudio_shape:
            emotion_pad = torch.zeros([textaudio_shape - emotion.shape[0]]).to(mydevice)
            emotion = torch.cat((emotion, emotion_pad), 0)

        if dataset == 'BEAT':
            data_mean_ = np.load(os.path.join(getattr(args, 'stats_path', '../data/stats'), 'gesture_BEAT_mean_' + args.version + '.npy'))
            data_std_ = np.load(os.path.join(getattr(args, 'stats_path', '../data/stats'), 'gesture_BEAT_std_' + args.version + '.npy'))
        elif dataset == 'TWH':
            data_mean_ = np.load(os.path.join(getattr(args, 'stats_path', '../data/stats'), 'gesture_TWH_mean_v0.npy'))
            data_std_ = np.load(os.path.join(getattr(args, 'stats_path', '../data/stats'), 'gesture_TWH_std_v0.npy'))

        data_mean = np.array(data_mean_)
        data_std = np.array(data_std_)
        # std = np.clip(data_std, a_min=0.01, a_max=None)
        
        shape_ = (num_subdivision, model.njoints, model.nfeats, args.n_poses)
        for i in range(0, num_subdivision):
            if i == 0:
                audio_sub = textaudio[i*stride:i*stride+stride_poses, :].unsqueeze(0) 
                emo_sub = emotion[i*stride:i*stride+stride_poses].unsqueeze(0) 
            else:

                audio_sub = torch.cat((audio_sub, textaudio[i*stride:i*stride+stride_poses, :].unsqueeze(0)), 0)
                emo_sub = torch.cat((emo_sub, emotion[i*stride:i*stride+stride_poses].unsqueeze(0)), 0)
            # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].unsqueeze(0)     # attention 4


            
        model_kwargs_ = {'y': {}}

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs_['y']['scale'] = torch.ones(num_subdivision, device=mydevice) * args.guidance_param

        model_kwargs_['y']['mask'] = (torch.zeros([num_subdivision, 1, 1, args.n_poses]) < 1).to(mydevice)
        model_kwargs_['y']['style'] = torch.tensor([[style]], dtype=torch.long, device=mydevice).repeat(num_subdivision, 1)

        model_kwargs_['y']['mask_local'] = torch.ones(num_subdivision, args.n_poses).bool().to(mydevice)

        model_kwargs_['y']['audio'] = audio_sub.to(mydevice)
        model_kwargs_['y']['emotion'] = emo_sub.int().to(mydevice)

        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )

        # print(sample.shape) 

        sample_seq = sample[:, :args.njoints] # ([b, 648, 1, 150])

        sample_seq = sample_seq.squeeze(2).permute(0, 2, 1).reshape(num_subdivision, -1, model.njoints)

        seed = args.n_seed
        if use_smplx_repr:
            full_motion = merge_subdivisions(sample_seq, stride=stride, seed=seed, device=mydevice)
            out_matrix = full_motion.detach().cpu().numpy()
        else:
            # legacy BEAT matrix representation: blend root slot and other joints separately
            pos = sample_seq[:, :, :9]
            joints = sample_seq[:, :, 9:]
            b, s, c1 = joints.shape

            fade_forward = torch.ones((1, s, 1)).to(mydevice)
            fade_back = torch.ones((1, s, 1)).to(mydevice)
            fade_forward[:, stride:, :] = torch.linspace(1, 0, seed)[None, :, None].to(mydevice)
            fade_back[:, :seed, :] = torch.linspace(0, 1, seed)[None, :, None].to(mydevice)

            pos[:-1] *= fade_forward
            pos[1:] *= fade_back

            full_pos = torch.zeros((s + stride * (b - 1), 9)).to(mydevice)
            idx = 0
            for pos_slice in pos:
                full_pos[idx:idx + s] += pos_slice
                idx += stride

            fade_f = torch.linspace(1, 0, seed)[None, :, None].to(mydevice)
            fade_b = torch.linspace(0, 1, seed)[None, :, None].to(mydevice)
            left, right = joints[:-1, stride:], joints[1:, :seed]
            merged = fade_f * left + fade_b * right

            full_q = torch.zeros((s + stride * (b - 1), c1)).to(pos.device)
            full_q[:stride] += joints[0, :stride]
            idx = stride
            for it, q_slice in enumerate(merged, start=1):
                full_q[idx:idx + seed] += q_slice
                full_q[idx + seed:idx + stride] += joints[it, seed:stride]
                idx += stride
            full_q[idx:idx + seed] += joints[-1, :seed]
            out_matrix = torch.cat([full_pos, full_q], 1).detach().cpu().numpy()

            if args.body == 'upper':
                zeros_sample = np.zeros((out_matrix.shape[0], 42 * 3))
                root_sample = np.concatenate([zeros_sample[:, :6 * 3], out_matrix], axis=1)
                lower_sample = np.concatenate([root_sample, zeros_sample[:, 6 * 3:]], axis=1)
                out_matrix = lower_sample
        
        out_poses = np.multiply(out_matrix, data_std) + data_mean
        print(out_poses.shape, real_n_frames)

        out = torch.FloatTensor(out_poses).unsqueeze(0).to(mydevice)
        tar = torch.FloatTensor(gesture).unsqueeze(0).to(mydevice)
        
        if gesture.shape[0] < real_n_frames:
            real_n_frames = gesture.shape[0]
        out_poses = out_poses[:real_n_frames]
        tar_poses = gesture[:real_n_frames]
        # print(tar_poses.shape)

        if save_npz:
            ref_npz = resolve_ref_npz(filename)
            npz_path = os.path.join(save_dir, filename + "_generated.npz")
            if use_smplx_repr:
                save_smplx_npz_from_poseexp(
                    npz_path,
                    out_poses,
                    pose_dim=smplx_pose_dim,
                    exp_dim=smplx_exp_dim,
                    ref_npz_path=ref_npz
                )
            else:
                save_smplx_npz_from_legacy(npz_path, out_poses, ref_npz_path=ref_npz)

        if use_smplx_repr:
            test_out_motion = out_poses
            test_tar_motion = tar_poses
        else:
            out_euler = pose2euler(out_poses, gt=False, fixroot=False, normalize_root=False, upper_body=False)
            tar_euler = pose2euler(tar_poses, gt=True, fixroot=False)
            if args.body == 'upper':
                test_out_motion = out_euler[:, 6:192]
                test_tar_motion = tar_euler[:, 6:192]
            else:
                test_out_motion = out_euler
                test_tar_motion = tar_euler

        motion_stride = 120
        motion_stride_poses = 150
        if eval_model is not None:
            for i in range(0, num_subdivision):
                if i == 0:
                    cat_results = out[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]
                    cat_targets = tar[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]
                else:
                    cat_results = torch.cat([cat_results, out[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]], 0)
                    cat_targets = torch.cat([cat_targets, tar[:, i*motion_stride:i*motion_stride+motion_stride_poses, :]], 0)

            latent_out = eval_model(cat_results)
            latent_ori = eval_model(cat_targets)
            if its == 0:
                latent_out_all = latent_out.detach().cpu().numpy()
                latent_ori_all = latent_ori.detach().cpu().numpy()
            else:
                latent_out_all = np.concatenate([latent_out_all, latent_out.detach().cpu().numpy()], axis=0)
                latent_ori_all = np.concatenate([latent_ori_all, latent_ori.detach().cpu().numpy()], axis=0)

        if not use_smplx_repr:
            speaker_name = filename.split("_")[0]
            pose2bvh(
                save_dir,
                filename,
                out_euler,
                pipeline=None,
                gt=False
            )

        _ = l1_calculator.run(test_out_motion.copy())

        if use_smplx_repr:
            arm_out = _extract_smplx_arm_triplets_deg(test_out_motion, pose_dim=smplx_pose_dim)
            arm_tar = _extract_smplx_arm_triplets_deg(test_tar_motion, pose_dim=smplx_pose_dim)
            _ = srgr_calculator.run(arm_out, arm_tar, sem)
            paper_out = _extract_smplx_joint_set_deg(test_out_motion, srgr_paper_joint_ids, pose_dim=smplx_pose_dim)
            paper_tar = _extract_smplx_joint_set_deg(test_tar_motion, srgr_paper_joint_ids, pose_dim=smplx_pose_dim)
            sem_use = sem[:paper_out.shape[0]]
            _ = srgr_paper_calculator.run(paper_out, paper_tar, sem_use)
            legacy_out = smplx_pose_to_legacy_euler(test_out_motion[:, :smplx_pose_dim], gt=False, upper_body=False)
            legacy_tar = smplx_pose_to_legacy_euler(test_tar_motion[:, :smplx_pose_dim], gt=True, upper_body=False)
            legacy_len = min(legacy_out.shape[0], legacy_tar.shape[0], sem.shape[0])
            _ = srgr_legacy_like_calculator.run(legacy_out[:legacy_len], legacy_tar[:legacy_len], sem[:legacy_len])
            _ = srgr_legacy_like_upper_calculator.run(
                legacy_out[:legacy_len, 6:192],
                legacy_tar[:legacy_len, 6:192],
                sem[:legacy_len],
            )
            onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(wav.reshape(-1), t_start, t_end, True)
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = alignmenter.load_pose(
                arm_out, t_start, t_end, pose_fps, True
            )
            align += alignmenter.calculate_align(
                onset_raw, onset_bt, onset_bt_rms,
                beat_right_arm, beat_right_shoulder, beat_right_wrist,
                beat_left_arm, beat_left_shoulder, beat_left_wrist,
                pose_fps
            )
        else:
            _ = srgr_calculator.run(test_out_motion, test_tar_motion, sem)
            onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(wav.reshape(-1), t_start, t_end, True)
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = alignmenter.load_pose(test_out_motion, t_start, t_end, pose_fps, True)
            align += alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, pose_fps)
        processed_samples += 1
    metrics = {
        "dataset": dataset,
        "motion_repr": motion_repr,
        "model_path": model_path,
        "tst_path": tst_path,
        "num_samples_total": len(test_files),
        "num_samples_processed": processed_samples,
        "guidance_param": float(args.guidance_param),
    }

    if eval_model is not None:
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        diversity = data_tools.FIDCalculator.get_diversity_scores(latent_out_all, randm_num)
        metrics["fid"] = float(fid)
        metrics["fgd"] = float(fid)
        metrics["diversity"] = float(diversity)
        print(f"fid score: {fid}")
        print(f"fgd score: {fid}")
        print(f"diversity score: {diversity}")

    l1div = l1_calculator.avg()
    metrics["l1div"] = float(l1div)
    print(f"l1div score: {l1div}")
    srgr = srgr_calculator.avg()
    if use_smplx_repr:
        srgr_paper = srgr_paper_calculator.avg()
        srgr_legacy_like = srgr_legacy_like_calculator.avg()
        srgr_legacy_like_upper = srgr_legacy_like_upper_calculator.avg()
    else:
        srgr_paper = srgr
        srgr_legacy_like = srgr
        srgr_legacy_like_upper = srgr
    align_avg = align / max(len(test_files), 1)
    metrics["srgr"] = float(srgr)
    metrics["srgr_paper"] = float(srgr_paper)
    metrics["srgr_paper_joint_set"] = srgr_paper_joint_set
    metrics["srgr_paper_threshold"] = float(srgr_paper_threshold)
    metrics["srgr_legacy_like"] = float(srgr_legacy_like)
    metrics["srgr_legacy_like_threshold"] = 4.0
    metrics["srgr_legacy_like_joint_set"] = "legacy_whole76_euler"
    metrics["srgr_legacy_like_upper"] = float(srgr_legacy_like_upper)
    metrics["srgr_legacy_like_upper_threshold"] = 4.0
    metrics["srgr_legacy_like_upper_joint_set"] = "legacy_upper62_euler"
    metrics["align_variant"] = "six_stream_onset_bt_rms_mean"
    metrics["align"] = float(align_avg)
    metrics["beat_align"] = float(align_avg)
    metrics["bc"] = float(align_avg)
    print(f"srgr score: {srgr}")
    print(f"srgr_paper score: {srgr_paper} (joint_set={srgr_paper_joint_set}, threshold={srgr_paper_threshold})")
    print(f"srgr_legacy_like score: {srgr_legacy_like} (joint_set=legacy_whole76_euler, threshold=4.0)")
    print(f"srgr_legacy_like_upper score: {srgr_legacy_like_upper} (joint_set=legacy_upper62_euler, threshold=4.0)")
    print("align score:", align_avg)

    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_latest_path = os.path.join(save_dir, "metrics_latest.json")
    metrics_ts_path = os.path.join(save_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_latest_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_ts_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {metrics_latest_path}")
    print(f"Saved metrics to: {metrics_ts_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPFS')
    parser.add_argument('--config', default='./configs/hpfs_final.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', nargs='+')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--tst_path', type=str, default=None)
    parser.add_argument('--wavlm_path', type=str, default=None)
    parser.add_argument('--word2vector_path', type=str, default=None)
    parser.add_argument('--motion_repr', type=str, default=None, choices=['legacy', 'smplx'])
    parser.add_argument('--save_npz', dest='save_npz', action='store_true')
    parser.add_argument('--no_save_npz', dest='save_npz', action='store_false')
    parser.add_argument('--smplx_ref_dir', type=str, default=None)
    parser.add_argument('--smplx_pose_dim', type=int, default=None)
    parser.add_argument('--smplx_exp_dim', type=int, default=None)
    parser.add_argument('--smplx_srgr_threshold', type=float, default=None)
    parser.add_argument('--smplx_srgr_paper_threshold', type=float, default=None)
    parser.add_argument('--smplx_srgr_paper_joint_set', type=str, default=None)
    parser.set_defaults(save_npz=None)
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    optional_cli_keys = {
        'motion_repr', 'save_npz', 'smplx_ref_dir', 'smplx_pose_dim', 'smplx_exp_dim', 'tst_path',
        'smplx_srgr_threshold', 'smplx_srgr_paper_threshold', 'smplx_srgr_paper_joint_set'
    }
    for k, v in vars(args).items():
        if k in optional_cli_keys and v is None:
            continue
        config[k] = v
    # pprint(config)
    config = EasyDict(config)
    config.model_name = config.name
    config.cond_mode = ''

    config.device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    config.save_dir = "./outputs/" + 'sample_dir_' + config.name + '_' + config.body + '/'

    print('tst_path', config.tst_path, 'save_dir', config.save_dir, 'name', config.name)


    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, mydevice=mydevice)
