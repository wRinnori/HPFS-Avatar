import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from pytorch3d.io import save_obj

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument(
        '--ablation',
        type=str,
        default=None,
        choices=[
            'full',
            'base_protocol',
            'geom_only',
            'deform_only',
            'silhouette_only',
            'depth_only',
            'deform_no_root_local',
            'deform_no_ramp',
            'deform_no_reg',
        ],
    )
    parser.add_argument('--silhouette_loss_weight', type=float, default=None)
    parser.add_argument('--depth_loss_weight', type=float, default=None)
    parser.add_argument('--deformation_residual_weight', type=float, default=None)
    parser.add_argument('--deformation_lr', type=float, default=None)
    parser.add_argument('--deformation_start_itr', type=int, default=None)
    parser.add_argument('--deformation_ramp_end_itr', type=int, default=None)
    parser.add_argument('--deformation_max_disp_start', type=float, default=None)
    parser.add_argument('--deformation_max_disp_end', type=float, default=None)
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    overrides = {
        'silhouette_loss_weight': args.silhouette_loss_weight,
        'depth_loss_weight': args.depth_loss_weight,
        'deformation_residual_weight': args.deformation_residual_weight,
        'deformation_lr': args.deformation_lr,
        'deformation_start_itr': args.deformation_start_itr,
        'deformation_ramp_end_itr': args.deformation_ramp_end_itr,
        'deformation_max_disp_start': args.deformation_max_disp_start,
        'deformation_max_disp_end': args.deformation_max_disp_end,
    }
    cfg.set_args(
        args.subject_id,
        ablation=args.ablation,
        exp_name=args.exp_name,
        overrides=overrides,
    )

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
 
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, 'test')
        
        # save
        human_img = out['human_img'].cpu().numpy()
        human_img_refined = out['human_img_refined'].cpu().numpy()
        human_face_img = out['human_face_img'].cpu().numpy()
        human_face_img_refined = out['human_face_img_refined'].cpu().numpy()
        smplx_mesh = out['smplx_mesh'].cpu().numpy()
        batch_size = human_img.shape[0]
        for i in range(batch_size):
            capture_id = 'Take' + str(int(data['capture_idx'][i]))
            frame_idx = int(data['frame_idx'][i])
            save_root_path = osp.join(cfg.result_dir, 'test', capture_id)
            os.makedirs(save_root_path, exist_ok=True)

            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_refined.png'), human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face.png'), human_face_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face_refined.png'), human_face_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)

            save_obj(osp.join(save_root_path, str(frame_idx) + '_smplx.obj'), torch.FloatTensor(smplx_mesh[i]), torch.LongTensor(smpl_x.face))

    
if __name__ == "__main__":
    main()
