import argparse
from config import cfg
import torch
from base import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
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
    return args

def main():
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
        args.continue_train,
        args.ablation,
        exp_name=args.exp_name,
        overrides=overrides,
    )

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    best_val_psnr = trainer.best_val_psnr
    best_val_ssim = trainer.best_val_ssim
    best_val_lpips = trainer.best_val_lpips
    best_epoch = trainer.best_epoch
    no_improve_epoch = trainer.no_improve_epoch

    if trainer.start_epoch >= cfg.end_epoch:
        trainer.logger.info(
            'Start epoch %d >= end_epoch %d. Nothing to train. Increase cfg.end_epoch to continue.'
            % (trainer.start_epoch, cfg.end_epoch)
        )
        return

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, data in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            # set stage
            cur_itr = epoch * len(trainer.batch_generator) + itr
            cfg.set_stage(cur_itr)

            # set learning rate
            tot_itr = cfg.end_epoch * len(trainer.batch_generator)
            trainer.set_lr(cur_itr, tot_itr)
           
            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(data, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            
            # update
            trainer.optimizer.step()
            
            # log
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))
            print(cfg.model_dir)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
            cur_itr += 1

        # validation + early stopping
        val_stat = trainer.validate(cfg.val_max_batches)
        trainer.logger.info(
            'Validation epoch %d: val_mse %.8f val_psnr %.4f val_ssim %.4f val_lpips %.4f'
            % (epoch, val_stat['mse'], val_stat['psnr'], val_stat['ssim'], val_stat['lpips'])
        )

        is_best = val_stat['psnr'] > best_val_psnr + cfg.early_stop_min_delta
        if is_best:
            best_val_psnr = val_stat['psnr']
            best_val_ssim = val_stat['ssim']
            best_val_lpips = val_stat['lpips']
            best_epoch = epoch
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        state = {
            'epoch': epoch,
            'network': trainer.model.module.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'val_psnr': val_stat['psnr'],
            'val_ssim': val_stat['ssim'],
            'val_lpips': val_stat['lpips'],
            'best_val_psnr': best_val_psnr,
            'best_val_ssim': best_val_ssim,
            'best_val_lpips': best_val_lpips,
            'best_epoch': best_epoch,
            'no_improve_epoch': no_improve_epoch,
        }
        trainer.save_model(state, epoch)

        if is_best:
            trainer.save_best_model(state)
            trainer.logger.info(
                'Best model updated at epoch %d with val_psnr %.4f val_ssim %.4f val_lpips %.4f'
                % (epoch, best_val_psnr, val_stat['ssim'], val_stat['lpips'])
            )
        else:
            trainer.logger.info(
                'No improvement for %d epoch(s). Best epoch: %d, best val_psnr: %.4f, best val_ssim: %.4f, best val_lpips: %.4f'
                % (no_improve_epoch, best_epoch, best_val_psnr, best_val_ssim, best_val_lpips)
            )

        trainer.best_val_psnr = best_val_psnr
        trainer.best_val_ssim = best_val_ssim
        trainer.best_val_lpips = best_val_lpips
        trainer.best_epoch = best_epoch
        trainer.no_improve_epoch = no_improve_epoch

        stop_ready = (
            cfg.use_early_stop
            and (epoch >= cfg.early_stop_start_epoch)
            and (no_improve_epoch >= cfg.early_stop_patience)
        )
        if stop_ready:
            trainer.logger.info(
                'Early stop triggered at epoch %d. Best epoch: %d, best val_psnr: %.4f, best val_ssim: %.4f, best val_lpips: %.4f '
                '(patience: %d, min_delta: %.4f, start_epoch: %d)'
                % (
                    epoch,
                    best_epoch,
                    best_val_psnr,
                    best_val_ssim,
                    best_val_lpips,
                    cfg.early_stop_patience,
                    cfg.early_stop_min_delta,
                    cfg.early_stop_start_epoch,
                )
            )
            break
   
if __name__ == "__main__":
    main()
