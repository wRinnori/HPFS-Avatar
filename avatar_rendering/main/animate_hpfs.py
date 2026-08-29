import argparse
import json
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.vis import render_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Animate an HPFS Gaussian avatar with converted SMPL-X motion.")
    parser.add_argument("--avatar_id", type=str, required=True, help="Checkpoint folder name under output/model_dump, e.g. 00028_v3")
    parser.add_argument("--identity_id", type=str, default="", help="Identity data folder under data/XHumans/data, e.g. 00028. Defaults to avatar_id prefix before first underscore.")
    parser.add_argument("--test_epoch", type=str, required=True, help="Checkpoint epoch, e.g. 24")
    parser.add_argument("--motion_path", type=str, required=True, help="Converted motion directory.")
    parser.add_argument("--output_video", type=str, default="", help="Optional output mp4 path.")
    parser.add_argument("--render_height", type=int, default=1920)
    parser.add_argument("--render_width", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save_frames_dir", type=str, default="", help="Optional directory to save rendered frames.")
    parser.add_argument("--render_mode", type=str, default="avatar", choices=["avatar", "mesh+avatar"], help="avatar: only photorealistic render; mesh+avatar: side-by-side.")
    parser.add_argument("--draw-frame-id", action="store_true", help="Overlay frame index text on outputs.")
    args = parser.parse_args()
    return args


def infer_identity_id(avatar_id):
    return avatar_id.split("_")[0]


def load_json_tensor(path, flatten=False):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        t = torch.FloatTensor(v).cuda()
        out[k] = t.view(-1) if flatten else t
    return out


def main():
    args = parse_args()
    identity_id = args.identity_id or infer_identity_id(args.avatar_id)

    cfg.set_args(args.avatar_id)
    tester = Tester(args.test_epoch)

    root_path = osp.join(cfg.data_dir, cfg.dataset, "data", identity_id)
    with open(osp.join(root_path, "smplx_optimized", "shape_param.json")) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "smplx_optimized", "face_offset.json")) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "smplx_optimized", "joint_offset.json")) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "smplx_optimized", "locator_offset.json")) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()

    frame_jsons = sorted(glob(osp.join(args.motion_path, "smplx_optimized", "smplx_params_smoothed", "*.json")), key=lambda x: int(osp.basename(x)[:-5]))
    if not frame_jsons:
        raise RuntimeError(f"No smplx_params_smoothed json found under: {args.motion_path}")
    frame_idx_list = [int(osp.basename(x)[:-5]) for x in frame_jsons]

    render_shape = (args.render_height, args.render_width)
    motion_name = osp.basename(osp.normpath(args.motion_path))
    output_video = args.output_video or (motion_name + ".mp4")
    output_video_dir = osp.dirname(output_video)
    if output_video_dir:
        os.makedirs(output_video_dir, exist_ok=True)
    video_out = None
    panel_multiplier = 2 if args.render_mode == "mesh+avatar" else 1
    video_out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (render_shape[1] * panel_multiplier, render_shape[0]),
    )

    save_frames_dir = args.save_frames_dir
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    for frame_idx in tqdm(frame_idx_list):
        cam_param = load_json_tensor(osp.join(args.motion_path, "cam_params", f"{frame_idx}.json"), flatten=False)
        smplx_param = load_json_tensor(osp.join(args.motion_path, "smplx_optimized", "smplx_params_smoothed", f"{frame_idx}.json"), flatten=True)

        with torch.no_grad():
            _, human_asset_refined, _, _ = tester.model.module.human_gaussian(smplx_param, cam_param, is_world_coord=False)
            human_render = tester.model.module.gaussian_renderer(human_asset_refined, render_shape, cam_param)

        render = (human_render["img"].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).copy().astype(np.uint8)

        if args.render_mode == "mesh+avatar":
            root_pose = smplx_param["root_pose"].view(1, 3)
            body_pose = smplx_param["body_pose"].view(1, (len(smpl_x.joint_part["body"]) - 1) * 3)
            jaw_pose = smplx_param["jaw_pose"].view(1, 3)
            leye_pose = smplx_param["leye_pose"].view(1, 3)
            reye_pose = smplx_param["reye_pose"].view(1, 3)
            lhand_pose = smplx_param["lhand_pose"].view(1, len(smpl_x.joint_part["lhand"]) * 3)
            rhand_pose = smplx_param["rhand_pose"].view(1, len(smpl_x.joint_part["rhand"]) * 3)
            expr = smplx_param["expr"].view(1, smpl_x.expr_param_dim)
            trans = smplx_param["trans"].view(1, 3)
            shape = tester.model.module.human_gaussian.shape_param[None]
            face_offset_cur = smpl_x.face_offset.cuda()[None]
            joint_offset_cur = tester.model.module.human_gaussian.joint_offset[None]
            output = tester.model.module.smplx_layer(
                global_orient=root_pose,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=lhand_pose,
                right_hand_pose=rhand_pose,
                expression=expr,
                betas=shape,
                transl=trans,
                face_offset=face_offset_cur,
                joint_offset=joint_offset_cur,
            )
            mesh = output.vertices[0]
            mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255).astype(np.uint8)
            out = np.concatenate((mesh_render, render), axis=1).astype(np.uint8)
        else:
            out = render

        if args.draw_frame_id:
            cv2.putText(out, str(frame_idx), (int(out.shape[1] * 0.03), int(out.shape[0] * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 2)
        video_out.write(out)
        if save_frames_dir:
            cv2.imwrite(osp.join(save_frames_dir, f"{frame_idx:06d}.png"), out)

    video_out.release()
    print(f"Saved video to: {output_video}")


if __name__ == "__main__":
    main()
