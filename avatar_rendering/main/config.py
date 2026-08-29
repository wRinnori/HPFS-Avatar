import os
import os.path as osp
import sys

class Config:
    
    ## shape
    triplane_shape_3d = (2, 2, 2)
    triplane_face_shape_3d = (0.3, 0.3, 0.3)
    triplane_shape = (32, 128, 128)
    
    ## train
    lr = 1e-3 
    warmup_itr = 100
    use_early_stop = True
    early_stop_patience = 5
    early_stop_min_delta = 0.001
    early_stop_start_epoch = 10
    val_ratio = 0.1
    val_split_seed = 2024
    val_max_batches = -1 # -1 means all validation batches
    use_geometry_supervision = True
    silhouette_loss_weight = 0.05
    depth_loss_weight = 0.1
    depth_loss_beta = 0.01
    deformation_lr = 1e-4
    deformation_start_itr = 1000
    deformation_ramp_end_itr = 8000
    deformation_max_disp_start = 0.005
    deformation_max_disp_end = 0.03
    deformation_use_root_local = True
    smplx_param_lr = 0 # for X-Humans dataset, we do not optimize smplx paraeters as it gives better results
    end_epoch = 80

    ## loss functions
    rgb_loss_weight = 0.8
    ssim_loss_weight = 0.2
    lpips_weight = 0.2
    deformation_residual_weight = 0.1

    ## dataset
    dataset = 'XHumans'

    ## others
    num_thread = 8
    default_num_gpus = 1
    num_gpus = default_num_gpus
    batch_size = 1 # Gaussian splatting renderer only supports batch_size==1
    smplx_gender = 'male' # only use male version as female version is not very good
    is_warmup = False
    cur_itr = deformation_ramp_end_itr

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    ablation = None
    exp_name = None

    def reset_experiment_defaults(self):
        self.use_geometry_supervision = True
        self.silhouette_loss_weight = 0.05
        self.depth_loss_weight = 0.1
        self.depth_loss_beta = 0.01
        self.deformation_lr = 1e-4
        self.deformation_start_itr = 1000
        self.deformation_ramp_end_itr = 8000
        self.deformation_max_disp_start = 0.005
        self.deformation_max_disp_end = 0.03
        self.deformation_use_root_local = True
        self.deformation_residual_weight = 0.1

    def apply_ablation_overrides(self):
        self.reset_experiment_defaults()
        if not self.ablation:
            return

        disable_deformation_forever_itr = 10 ** 12

        if self.ablation == 'full':
            return
        if self.ablation == 'base_protocol':
            self.use_geometry_supervision = False
            self.silhouette_loss_weight = 0.0
            self.depth_loss_weight = 0.0
            self.deformation_start_itr = disable_deformation_forever_itr
            self.deformation_ramp_end_itr = disable_deformation_forever_itr
            self.deformation_residual_weight = 0.0
        elif self.ablation == 'geom_only':
            self.deformation_start_itr = disable_deformation_forever_itr
            self.deformation_ramp_end_itr = disable_deformation_forever_itr
            self.deformation_residual_weight = 0.0
        elif self.ablation == 'deform_only':
            self.use_geometry_supervision = False
            self.silhouette_loss_weight = 0.0
            self.depth_loss_weight = 0.0
        elif self.ablation == 'silhouette_only':
            self.silhouette_loss_weight = 0.05
            self.depth_loss_weight = 0.0
        elif self.ablation == 'depth_only':
            self.silhouette_loss_weight = 0.0
            self.depth_loss_weight = 0.1
        elif self.ablation == 'deform_no_root_local':
            self.deformation_use_root_local = False
        elif self.ablation == 'deform_no_ramp':
            self.deformation_start_itr = 0
            self.deformation_ramp_end_itr = 0
            self.deformation_max_disp_start = self.deformation_max_disp_end
        elif self.ablation == 'deform_no_reg':
            self.deformation_residual_weight = 0.0
        else:
            raise ValueError('Unknown ablation: ' + str(self.ablation))

    def apply_runtime_overrides(self, overrides=None):
        if not overrides:
            return
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(self, key):
                raise AttributeError('Unknown config override: ' + str(key))
            setattr(self, key, value)

    def set_args(self, subject_id, continue_train=False, ablation=None, exp_name=None, overrides=None):
        self.subject_id = subject_id
        self.continue_train = continue_train
        self.ablation = ablation
        self.exp_name = exp_name
        self.apply_ablation_overrides()
        self.apply_runtime_overrides(overrides)
        self.apply_subject_overrides()
        self.cur_itr = self.deformation_ramp_end_itr
        self.is_warmup = False

        model_root = osp.join(self.output_dir, 'model_dump')
        result_root = osp.join(self.output_dir, 'result')
        path_parts = []
        if self.exp_name:
            path_parts.append(self.exp_name)
        if self.ablation:
            path_parts.append(self.ablation)
        path_parts.append(subject_id)
        self.model_dir = osp.join(model_root, *path_parts)
        self.result_dir = osp.join(result_root, *path_parts)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def apply_subject_overrides(self):
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        if visible_devices:
            device_ids = [device_id.strip() for device_id in visible_devices.split(',') if device_id.strip()]
            self.num_gpus = len(device_ids) if device_ids else self.default_num_gpus
        else:
            self.num_gpus = self.default_num_gpus

    def set_stage(self, itr):
        self.cur_itr = itr
        if itr < self.warmup_itr:
            self.is_warmup = True
        else:
            self.is_warmup = False

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
