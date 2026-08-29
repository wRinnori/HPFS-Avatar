import os
import os.path as osp
import math
import glob
import abc

import numpy as np
import torch
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, random_split
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import cfg
from logger import colorlogger
from model import get_model
from timer import Timer

# dynamic dataset import
exec('from ' + cfg.dataset + ' import ' + cfg.dataset)


def _build_run_name():
    name_parts = []
    if getattr(cfg, 'exp_name', None):
        name_parts.append(cfg.exp_name)
    if getattr(cfg, 'ablation', None):
        name_parts.append(cfg.ablation)
    name_parts.append(getattr(cfg, 'subject_id', 'logs'))
    return '_'.join(name_parts)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        log_name = 'train_{}.txt'.format(_build_run_name())
        super(Trainer, self).__init__(log_name=log_name)
        self.val_psnr_metric = PeakSignalNoiseRatio(data_range=1).cuda()
        self.val_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1).cuda()
        self.val_lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

    def get_optimizer(self, optimizable_params):
        optimizer = torch.optim.Adam(optimizable_params, lr=0.0, eps=1e-15)
        return optimizer

    def set_lr(self, cur_itr, tot_itr):
        for param_group in self.optimizer.param_groups:
            name = param_group['name']

            if name in ('deformation_net_human', 'deformation_gate_net_human'):
                base_lr = cfg.deformation_lr
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = base_lr / 10
                elif cur_itr > 0.95 * tot_itr:
                    param_group['lr'] = base_lr / 100
                else:
                    param_group['lr'] = base_lr
            elif 'human' in name:
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = cfg.lr / 10
                elif cur_itr > 0.95 * tot_itr:
                    param_group['lr'] = cfg.lr / 100
                else:
                    param_group['lr'] = cfg.lr
            elif 'smplx' in name:
                lr = cfg.smplx_param_lr
                if 'hand' in name:
                    lr = cfg.smplx_hand_param_lr
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = lr / 10
                elif cur_itr > 0.95 * tot_itr:
                    param_group['lr'] = lr / 100
                else:
                    param_group['lr'] = lr

    def _make_batch_generator(self):
        self.logger.info('Creating dataset...')
        trainset = eval(cfg.dataset)(transforms.ToTensor(), 'train')
        if len(trainset) <= 1:
            raise ValueError('Training set must have at least 2 samples for train/val split.')

        val_size = int(len(trainset) * cfg.val_ratio)
        val_size = max(1, min(val_size, len(trainset) - 1))
        train_size = len(trainset) - val_size
        split_generator = torch.Generator().manual_seed(cfg.val_split_seed)
        train_subset, val_subset = random_split(
            trainset, [train_size, val_size], generator=split_generator
        )

        batch_size = cfg.num_gpus * cfg.batch_size
        self.itr_per_epoch = math.ceil(len(train_subset) / batch_size)
        self.batch_generator = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )
        self.val_batch_generator = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )
        self.smplx_params = trainset.smplx_params
        self.logger.info('Train/Val split: %d/%d' % (len(train_subset), len(val_subset)))

    def _make_model(self, epoch=None):
        if cfg.continue_train:
            ckpt = self.load_model()
            model = get_model(self.smplx_params)
            model = DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()
            model.module.load_state_dict(ckpt['network'], strict=False)
        else:
            ckpt = None
            model = get_model(self.smplx_params)
            model = DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()

        optimizer = self.get_optimizer(model.module.optimizable_params)

        if cfg.continue_train:
            start_epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['optimizer'])
            self.best_val_psnr = ckpt.get('best_val_psnr', -1e10)
            self.best_val_ssim = ckpt.get('best_val_ssim', -1e10)
            self.best_val_lpips = ckpt.get('best_val_lpips', 1e10)
            self.best_epoch = ckpt.get('best_epoch', -1)
            self.no_improve_epoch = ckpt.get('no_improve_epoch', 0)
        else:
            start_epoch = 0
            self.best_val_psnr = -1e10
            self.best_val_ssim = -1e10
            self.best_val_lpips = 1e10
            self.best_epoch = -1
            self.no_improve_epoch = 0

        model.train()
        for module in model.module.eval_modules:
            module.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info('Write snapshot into {}'.format(file_path))

    def save_best_model(self, state):
        file_path = osp.join(cfg.model_dir, 'best.pth')
        torch.save(state, file_path)
        self.logger.info('Write best snapshot into {}'.format(file_path))

    def validate(self, max_batches=-1):
        self.model.eval()
        mse_sum, psnr_sum, ssim_sum, lpips_sum, sample_num = 0.0, 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for itr, data in enumerate(self.val_batch_generator):
                if (max_batches > 0) and (itr >= max_batches):
                    break

                out = self.model(data, 'test')
                pred = out['human_img_refined']
                gt = data['img'].cuda(non_blocking=True)
                mask = data['mask'].cuda(non_blocking=True)
                pred = pred * mask + (1 - mask)
                gt = gt * mask + (1 - mask)
                mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
                mse_sum += mse.sum().item()
                batch_size = pred.shape[0]
                for b in range(batch_size):
                    pred_b = pred[b:b+1]
                    gt_b = gt[b:b+1]
                    psnr_sum += self.val_psnr_metric(pred_b, gt_b).item()
                    self.val_psnr_metric.reset()
                    ssim_sum += self.val_ssim_metric(pred_b, gt_b).item()
                    self.val_ssim_metric.reset()
                    lpips_sum += self.val_lpips_metric(pred_b * 2 - 1, gt_b * 2 - 1).item()
                    self.val_lpips_metric.reset()
                sample_num += batch_size

        mean_mse = mse_sum / max(sample_num, 1)
        val_psnr = -10.0 * math.log10(max(mean_mse, 1e-12))
        val_ssim = ssim_sum / max(sample_num, 1)
        val_lpips = lpips_sum / max(sample_num, 1)

        self.model.train()
        for module in self.model.module.eval_modules:
            module.eval()
        return {'mse': mean_mse, 'psnr': val_psnr, 'ssim': val_ssim, 'lpips': val_lpips}

    def load_model(self):
        model_file_list = glob.glob(osp.join(cfg.model_dir, 'snapshot_*.pth'))
        if not model_file_list:
            raise FileNotFoundError('Cannot find snapshot checkpoint in ' + cfg.model_dir)

        cur_epoch = max(
            int(osp.basename(file_name)[len('snapshot_'):-len('.pth')])
            for file_name in model_file_list
        )
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')
        return ckpt


class Tester(Base):
    def __init__(self, test_epoch):
        log_name = 'test_{}.txt'.format(_build_run_name())
        super(Tester, self).__init__(log_name=log_name)
        self.test_epoch = str(test_epoch)

    def _make_batch_generator(self):
        self.logger.info('Creating dataset...')
        testset_loader = eval(cfg.dataset)(transforms.ToTensor(), 'test')
        batch_generator = DataLoader(
            dataset=testset_loader,
            batch_size=cfg.num_gpus * cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )

        self.testset = testset_loader
        self.batch_generator = batch_generator
        self.smplx_params = testset_loader.smplx_params

    def _make_model(self):
        if self.test_epoch == 'best':
            model_path = os.path.join(cfg.model_dir, 'best.pth')
        else:
            model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth' % int(self.test_epoch))
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')

        self.logger.info('Creating graph...')
        model = get_model(self.smplx_params)
        model = DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()
        model.module.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model
