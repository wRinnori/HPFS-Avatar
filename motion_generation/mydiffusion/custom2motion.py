import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../model']]
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from utils import metric, other_tools, data_tools
import subprocess
import os
from datetime import datetime
import copy
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from smplx_motion_io import (
    pose2euler,
    pose2bvh_style,
    save_smplx_npz_from_legacy,
    save_smplx_npz_from_poseexp,
)
from torch.utils.data import DataLoader
from data_loader.h5_data_loader import AllSpeechGestureDataset, RandomSampler, SequentialSampler
import argparse
from tqdm import tqdm
from model.cfg_sampler import ClassifierFreeSampleModel

emotion_id_dict = {
    'neutral': 0,
    'happiness': 1,
    'anger': 2,
    'sadness': 3,
    'contempt': 4,
    'surprise': 5,
    'fear': 6,
    'disgust': 7
}

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
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
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


def resolve_ckpt_path(path, root_path=""):
    path = str(path or "")
    root_path = str(root_path or "")
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    local_candidate = os.path.normpath(path)
    if os.path.exists(local_candidate):
        return local_candidate
    if root_path:
        return os.path.normpath(os.path.join(root_path, path))
    return os.path.normpath(path)

def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, txt_path=None, wavlm_path=None, word2vector_path=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start
    if args.e_name is not None:
        eval_model_module = __import__(f"model.{args.eval_model}", fromlist=["something"])
        eval_model = getattr(eval_model_module, args.e_name)(args)
        eval_ckpt = resolve_ckpt_path(getattr(args, "e_path", ""), getattr(args, "root_path", ""))
        if eval_ckpt and os.path.exists(eval_ckpt):
            other_tools.load_checkpoints(eval_model, eval_ckpt, args.e_name)
            eval_model.to(mydevice)
            eval_model.eval()
        else:
            print(f"[WARN] Eval model checkpoint not found at: {eval_ckpt}. Skip loading eval model; generation will continue.")
            
    tst_audio_dir = os.path.join(tst_path, 'audio_' + dataset)
    tst_text_dir = os.path.join(tst_path, 'text_' + dataset)

    l1_calculator = metric.L1div()

    motion_repr = getattr(args, 'motion_repr', None) or 'legacy'
    use_smplx_repr = (str(motion_repr).lower() == 'smplx')
    save_npz = bool(getattr(args, 'save_npz', False))
    smplx_ref_dir = getattr(args, 'smplx_ref_dir', None)
    smplx_pose_dim = int(getattr(args, 'smplx_pose_dim', None) or 165)
    smplx_exp_dim = int(getattr(args, 'smplx_exp_dim', None) or 100)

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

    for its, file in enumerate(os.listdir(tst_audio_dir)):
        filename = file.split(".")[0]
        print(f"Processing: {filename}")
        if dataset == 'BEAT':
            if 'our_style' in args.name:
                speaker_id = int(args.id)
                style_idx = speaker_id - 1
            else:
                speaker_id = int(filename.split('_')[0]) - 1
                speaker = np.zeros([args.speaker_num])
                speaker[speaker_id] = 1
                style_idx = int(np.where(speaker == np.max(speaker))[0][0])
            
        audio_path = os.path.join(tst_audio_dir, filename + '.npy')
        audio = np.load(audio_path)
        text_path = os.path.join(tst_text_dir, filename + '.npy')
        text = np.load(text_path)

        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)

        
        emo = np.array(emotion_id_dict[args.emo])

        emotion = torch.from_numpy(emo).to(mydevice)
        emotion = emotion.repeat(2).unsqueeze(0)
        
        seed=123456
        torch.manual_seed(seed)

        style = style_idx if 'our_style' in args.name else speaker

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
            print('real_n_frames: {}, num_subdivision: {}, stride_poses: {}, stride: {}, pad_frames: {}, speaker_id: {}'.format(real_n_frames, num_subdivision, stride_poses, stride, n_frames - real_n_frames + args.n_seed, style_idx))
        
        textaudio_pad = torch.zeros([n_frames - real_n_frames + args.n_seed, args.audio_feature_dim]).to(mydevice)
        textaudio = torch.cat((textaudio, textaudio_pad), 0)
        # textaudio_shape = textaudio.shape[0]

        # if emotion.shape[0] < textaudio_shape:
        #     emotion_pad = torch.zeros([textaudio_shape - emotion.shape[0]]).to(mydevice)
        #     emotion = torch.cat((emotion, emotion_pad), 0)

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
                # emo_sub = emotion[i*stride:i*stride+stride_poses].unsqueeze(0) 
            else:

                audio_sub = torch.cat((audio_sub, textaudio[i*stride:i*stride+stride_poses, :].unsqueeze(0)), 0)
                # emo_sub = torch.cat((emo_sub, emotion[i*stride:i*stride+stride_poses].unsqueeze(0)), 0)
            # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].unsqueeze(0)     # attention 4

        model_kwargs_ = {'y': {}}
        if args.guidance_param != 1:
            model_kwargs_['y']['scale'] = torch.ones(num_subdivision, device=mydevice) * args.guidance_param
        model_kwargs_['y']['mask'] = (torch.zeros([num_subdivision, 1, 1, args.n_poses]) < 1).to(mydevice)
        if 'our_style' in args.name:
            model_kwargs_['y']['style'] = torch.tensor([[style_idx]], dtype=torch.long, device=mydevice).repeat(num_subdivision, 1)
        else:
            model_kwargs_['y']['style'] = torch.tensor([[style_idx]]).long().repeat(num_subdivision, 1).to(mydevice)
        model_kwargs_['y']['mask_local'] = torch.ones(num_subdivision, args.n_poses).bool().to(mydevice)

        model_kwargs_['y']['audio'] = audio_sub.to(mydevice)
        model_kwargs_['y']['emotion'] = emotion.int().to(mydevice)

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
        
        out_poses = out_poses[:real_n_frames]
        if use_smplx_repr:
            test_out_motion = out_poses
        else:
            out_euler = pose2euler(out_poses, gt=False, fixroot=False, normalize_root=False, upper_body=False)
            if args.body == 'upper':
                test_out_motion = out_euler[:, 6:192]
            else:
                test_out_motion = out_euler
        
        # speaker_name = filename.split("_")[0]
        if args.id == 14:
            speaker_name = filename.split("_")[0]
        else:
            speaker_name = args.id
        suffix = str(args.id) + '_' + args.emo
        print('speaker_name', speaker_name)
        if not use_smplx_repr:
            pose2bvh_style(
                save_dir,
                filename,
                out_euler,
                pipeline=None,
                gt=False,
                suffix=suffix
            )
        if save_npz:
            ref_npz = resolve_ref_npz(filename)
            npz_path = os.path.join(save_dir, filename + "_" + suffix + ".npz")
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
        _ = l1_calculator.run(test_out_motion.copy())
        

    l1div = l1_calculator.avg()
    print(f"l1div score: {l1div}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPFS')
    parser.add_argument('--config', default='./configs/custom_smplx.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', nargs='+')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--wavlm_path', type=str, default=None)
    parser.add_argument('--word2vector_path', type=str, default=None)
    parser.add_argument('--motion_repr', type=str, default=None, choices=['legacy', 'smplx'])
    parser.add_argument('--save_npz', dest='save_npz', action='store_true')
    parser.add_argument('--no_save_npz', dest='save_npz', action='store_false')
    parser.add_argument('--smplx_ref_dir', type=str, default=None)
    parser.add_argument('--smplx_pose_dim', type=int, default=None)
    parser.add_argument('--smplx_exp_dim', type=int, default=None)
    parser.set_defaults(save_npz=None)
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    optional_cli_keys = {'motion_repr', 'save_npz', 'smplx_ref_dir', 'smplx_pose_dim', 'smplx_exp_dim'}
    for k, v in vars(args).items():
        if k in optional_cli_keys and v is None:
            continue
        config[k] = v
    # pprint(config)
    config = EasyDict(config)
    config.model_name = config.name
    config.cond_mode = ''

    device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    model_path_str = str(getattr(config, "model_path", "") or "")
    model_parts = [p for p in model_path_str.split('/') if p]
    model_root = model_parts[-5] if len(model_parts) >= 5 else ""
    model_spicific = os.path.splitext(os.path.basename(model_path_str))[0] or "model"
    config.save_dir = os.path.join(
        "..", "outputs", "custom",
        'sample_dir_' + model_spicific + '_' + 'audio2motion_' + config.body,
    ) + '/'

    print('model_root', model_root, 'tst_path', config.tst_path, 'save_dir', config.save_dir, 'name', config.name)

    # Keep YAML-controlled dataset/checkpoint paths for repository-local reproduction.
    if not getattr(config, 'h5file', None):
        config.h5file = '../data/' + config.dataset + '_' + config.version + '_test' + '.h5'

    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, txt_path=config.txt_path, wavlm_path=config.wavlm_path, word2vector_path=config.word2vector_path)

 
