import os
import time
import datetime

import torch
import torchvision

from utils import misc, metrics

best_psnr = 0


def train(train_loader, val_loader, model, optimizer, scheduler, loss_fn, logger, opt):
    total_step = opt.epochs * len(train_loader)

    step_time_log = misc.AverageMeter()
    loss_log = misc.AverageMeter(':6f')
    loss_fg_content_bg_appearance_construct_log = misc.AverageMeter(':6f')
    loss_lut_transform_image_log = misc.AverageMeter(':6f')
    loss_lut_regularize_log = misc.AverageMeter(':6f')

    start_epoch = 0

    "Load pretrained checkpoints"
    if opt.pretrained is not None:
        logger.info(f"Load pretrained weight from {opt.pretrained}")
        load_state = torch.load(opt.pretrained)
        model = model.cpu()
        model.load_state_dict(load_state['model'])
        model = model.to(opt.device)
        optimizer.load_state_dict(load_state['optimizer'])
        scheduler.load_state_dict(load_state['scheduler'])
        start_epoch = load_state['last_epoch'] + 1

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        time_ckp = time.time()
        for step, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + step + 1

            if opt.INRDecode and opt.hr_train:
                "List with 4 elements: [Input to Encoder, three different resolutions' crop to INR Decoder]"
                composite_image = [batch[f'composite_image{name}'].to(opt.device) for name in range(4)]
                real_image = [batch[f'real_image{name}'].to(opt.device) for name in range(4)]
                mask = [batch[f'mask{name}'].to(opt.device) for name in range(4)]
                coordinate_map = [batch[f'coordinate_map{name}'].to(opt.device) for name in range(4)]

                fg_INR_coordinates = coordinate_map[1:]

            else:
                composite_image = batch['composite_image'].to(opt.device)
                real_image = batch['real_image'].to(opt.device)
                mask = batch['mask'].to(opt.device)

                fg_INR_coordinates = batch['fg_INR_coordinates'].to(opt.device)

            fg_content_bg_appearance_construct, fit_lut3d, lut_transform_image = model(
                composite_image, mask, fg_INR_coordinates)

            if opt.INRDecode:
                loss_fg_content_bg_appearance_construct = 0
                """
                    Our LRIP module requires three different resolution layers, thus here 
                    `loss_fg_content_bg_appearance_construct` is calculated in multiple layers. 
                    Besides, when leverage `hr_train`, i.e. use RSC strategy (See Section 3.4), the `real_image`
                    and `mask` are list type, corresponding different resolutions' crop.
                """
                if opt.hr_train:
                    for n in range(3):
                        loss_fg_content_bg_appearance_construct += loss_fn['masked_mse'] \
                            (fg_content_bg_appearance_construct[n], real_image[3 - n], mask[3 - n])
                    loss_fg_content_bg_appearance_construct /= 3
                    loss_lut_transform_image = loss_fn['masked_mse'](lut_transform_image, real_image[1], mask[1])
                else:
                    for n in range(3):
                        loss_fg_content_bg_appearance_construct += loss_fn['MaskWeightedMSE'] \
                            (fg_content_bg_appearance_construct[n],
                             torchvision.transforms.Resize(opt.INR_input_size // 2 ** (3 - n - 1))(real_image),
                             torchvision.transforms.Resize(opt.INR_input_size // 2 ** (3 - n - 1))(mask))
                    loss_fg_content_bg_appearance_construct /= 3
                    loss_lut_transform_image = loss_fn['masked_mse'](lut_transform_image, real_image, mask)
                loss_lut_regularize = loss_fn['regularize_LUT'](fit_lut3d)

            else:
                loss_fg_content_bg_appearance_construct = 0
                loss_lut_transform_image = loss_fn['masked_mse'](lut_transform_image, real_image, mask)
                loss_lut_regularize = 0

            loss = loss_fg_content_bg_appearance_construct + loss_lut_transform_image + loss_lut_regularize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_time_log.update(time.time() - time_ckp)

            loss_fg_content_bg_appearance_construct_log.update(0 if isinstance(loss_fg_content_bg_appearance_construct,
                                                                               int) else loss_fg_content_bg_appearance_construct.item())
            loss_lut_transform_image_log.update(
                0 if isinstance(loss_lut_transform_image, int) else loss_lut_transform_image.item())
            loss_lut_regularize_log.update(0 if isinstance(loss_lut_regularize, int) else loss_lut_regularize.item())
            loss_log.update(loss.item())

            if current_step % opt.print_freq == 0:
                remain_secs = (total_step - current_step) * step_time_log.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))

                log_msg = f'Epoch: [{epoch}/{opt.epochs}]\t' \
                          f'Step: [{step}/{len(train_loader)}]\t' \
                          f'StepTime {step_time_log.val:.3f} ({step_time_log.avg:.3f})\t' \
                          f'lr {optimizer.param_groups[0]["lr"]}\t' \
                          f'Loss {loss_log.val:.4f} ({loss_log.avg:.4f})\t' \
                          f'Loss_fg_bg_cons {loss_fg_content_bg_appearance_construct_log.val:.4f} ({loss_fg_content_bg_appearance_construct_log.avg:.4f})\t' \
                          f'Loss_lut_trans {loss_lut_transform_image_log.val:.4f} ({loss_lut_transform_image_log.avg:.4f})\t' \
                          f'Loss_lut_reg {loss_lut_regularize_log.val:.4f} ({loss_lut_regularize_log.avg:.4f})\t' \
                          f'Remaining Time {remain_time} ({finish_time})'
                logger.info(log_msg)

                if opt.wandb:
                    import wandb
                    wandb.log(
                        {'Train/Epoch': epoch, 'Train/lr': optimizer.param_groups[0]['lr'], 'Train/Step': current_step,
                         'Train/Loss': loss_log.val,
                         'Train/Loss_fg_bg_cons': loss_fg_content_bg_appearance_construct_log.val,
                         'Train/Loss_lut_trans': loss_lut_transform_image_log.val,
                         'Train/Loss_lut_reg': loss_lut_regularize_log.val,
                         })

            time_ckp = time.time()

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'last_epoch': epoch,
                 'scheduler': scheduler.state_dict()}

        """
            As the validation of original resolution Harmonization will have no consistent resolution among images 
            (so fail to form a batch) and also may lead to out-of-memory problem when combined with training phase,
            we here only save the model when `opt.isFullRes` is True, leaving the evaluation in `inference.py`.
        """
        if opt.isFullRes and opt.hr_train:
            if epoch % 5 == 0:
                torch.save(state, os.path.join(opt.save_path, f"epoch{epoch}.pth"))
            else:
                torch.save(state, os.path.join(opt.save_path, "last.pth"))
        else:
            val(val_loader, model, logger, opt, state)


def val(val_loader, model, logger, opt, state):
    global best_psnr
    current_process = 10
    model.eval()

    metric_log = {
        'HAdobe5k': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'HCOCO': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'Hday2night': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'HFlickr': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'All': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
    }

    lut_metric_log = {
        'HAdobe5k': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'HCOCO': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'Hday2night': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'HFlickr': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
        'All': {'Samples': 0, 'MSE': 0, 'fMSE': 0, 'PSNR': 0, 'SSIM': 0},
    }

    for step, batch in enumerate(val_loader):
        composite_image = batch['composite_image'].to(opt.device)
        real_image = batch['real_image'].to(opt.device)
        mask = batch['mask'].to(opt.device)
        category = batch['category']

        fg_INR_coordinates = batch['fg_INR_coordinates'].to(opt.device)
        bg_INR_coordinates = batch['bg_INR_coordinates'].to(opt.device)
        fg_transfer_INR_RGB = batch['fg_transfer_INR_RGB'].to(opt.device)

        with torch.no_grad():
            fg_content_bg_appearance_construct, _, lut_transform_image = model(
                composite_image,
                mask,
                fg_INR_coordinates,
                bg_INR_coordinates)
        if opt.INRDecode:
            pred_fg_image = fg_content_bg_appearance_construct[-1]
        else:
            pred_fg_image = None
        fg_transfer_INR_RGB = misc.lin2img(fg_transfer_INR_RGB,
                                           val_loader.dataset.INR_dataset.size) if fg_transfer_INR_RGB is not None else None

        "For INR"
        mask_INR = torchvision.transforms.Resize(opt.INR_input_size)(mask)

        if not opt.INRDecode:
            pred_harmonized_image = None
        else:
            pred_harmonized_image = pred_fg_image * (mask > 100 / 255.) + real_image * (~(mask > 100 / 255.))
        lut_transform_image = lut_transform_image * (mask > 100 / 255.) + real_image * (~(mask > 100 / 255.))

        "Save the output images. For every 10 epochs, save more results, otherwise, save little. Thus save storage."
        if state['last_epoch'] % 10 == 0:
            misc.visualize(real_image, composite_image, mask, pred_fg_image,
                           pred_harmonized_image, lut_transform_image, opt, state['last_epoch'], show=False,
                           wandb=opt.wandb, isAll=True, step=step)
        elif step == 0:
            misc.visualize(real_image, composite_image, mask, pred_fg_image,
                           pred_harmonized_image, lut_transform_image, opt, state['last_epoch'], show=False,
                           wandb=opt.wandb, step=step)

        if opt.INRDecode:
            mse, fmse, psnr, ssim = metrics.calc_metrics(misc.normalize(pred_harmonized_image, opt, 'inv'),
                                                         misc.normalize(fg_transfer_INR_RGB, opt, 'inv'), mask_INR)

        lut_mse, lut_fmse, lut_psnr, lut_ssim = metrics.calc_metrics(misc.normalize(lut_transform_image, opt, 'inv'),
                                                                     misc.normalize(real_image, opt, 'inv'), mask)

        for idx in range(len(category)):
            if opt.INRDecode:
                metric_log[category[idx]]['Samples'] += 1
                metric_log[category[idx]]['MSE'] += mse[idx]
                metric_log[category[idx]]['fMSE'] += fmse[idx]
                metric_log[category[idx]]['PSNR'] += psnr[idx]
                metric_log[category[idx]]['SSIM'] += ssim[idx]

                metric_log['All']['Samples'] += 1
                metric_log['All']['MSE'] += mse[idx]
                metric_log['All']['fMSE'] += fmse[idx]
                metric_log['All']['PSNR'] += psnr[idx]
                metric_log['All']['SSIM'] += ssim[idx]

            lut_metric_log[category[idx]]['Samples'] += 1
            lut_metric_log[category[idx]]['MSE'] += lut_mse[idx]
            lut_metric_log[category[idx]]['fMSE'] += lut_fmse[idx]
            lut_metric_log[category[idx]]['PSNR'] += lut_psnr[idx]
            lut_metric_log[category[idx]]['SSIM'] += lut_ssim[idx]

            lut_metric_log['All']['Samples'] += 1
            lut_metric_log['All']['MSE'] += lut_mse[idx]
            lut_metric_log['All']['fMSE'] += lut_fmse[idx]
            lut_metric_log['All']['PSNR'] += lut_psnr[idx]
            lut_metric_log['All']['SSIM'] += lut_ssim[idx]

        if (step + 1) / len(val_loader) * 100 >= current_process:
            logger.info(f'Processing: {current_process}')
            current_process += 10

    logger.info('=========================')
    for key in metric_log.keys():
        if opt.INRDecode:
            msg = f"{key}-'MSE': {metric_log[key]['MSE'] / metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'fMSE': {metric_log[key]['fMSE'] / metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'PSNR': {metric_log[key]['PSNR'] / metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'SSIM': {metric_log[key]['SSIM'] / metric_log[key]['Samples']:.4f}\n" \
                  f"{key}-'LUT_MSE': {lut_metric_log[key]['MSE'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_fMSE': {lut_metric_log[key]['fMSE'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_PSNR': {lut_metric_log[key]['PSNR'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_SSIM': {lut_metric_log[key]['SSIM'] / lut_metric_log[key]['Samples']:.4f}\n"
        else:
            msg = f"{key}-'LUT_MSE': {lut_metric_log[key]['MSE'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_fMSE': {lut_metric_log[key]['fMSE'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_PSNR': {lut_metric_log[key]['PSNR'] / lut_metric_log[key]['Samples']:.2f}\n" \
                  f"{key}-'LUT_SSIM': {lut_metric_log[key]['SSIM'] / lut_metric_log[key]['Samples']:.4f}\n"

        logger.info(msg)

        if opt.wandb:
            import wandb
            if opt.INRDecode:
                wandb.log(
                    {f'Val/{key}/Epoch': state['last_epoch'],
                     f'Val/{key}/MSE': metric_log[key]['MSE'] / metric_log[key]['Samples'],
                     f'Val/{key}/fMSE': metric_log[key]['fMSE'] / metric_log[key]['Samples'],
                     f'Val/{key}/PSNR': metric_log[key]['PSNR'] / metric_log[key]['Samples'],
                     f'Val/{key}/SSIM': metric_log[key]['SSIM'] / metric_log[key]['Samples'],
                     f'Val/{key}/LUT_MSE': lut_metric_log[key]['MSE'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_fMSE': lut_metric_log[key]['fMSE'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_PSNR': lut_metric_log[key]['PSNR'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_SSIM': lut_metric_log[key]['SSIM'] / lut_metric_log[key]['Samples']
                     })
            else:
                wandb.log(
                    {f'Val/{key}/Epoch': state['last_epoch'],
                     f'Val/{key}/LUT_MSE': lut_metric_log[key]['MSE'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_fMSE': lut_metric_log[key]['fMSE'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_PSNR': lut_metric_log[key]['PSNR'] / lut_metric_log[key]['Samples'],
                     f'Val/{key}/LUT_SSIM': lut_metric_log[key]['SSIM'] / lut_metric_log[key]['Samples']
                     })

    logger.info('=========================')

    if not opt.INRDecode:
        if lut_metric_log['All']['PSNR'] / lut_metric_log['All']['Samples'] > best_psnr:
            logger.info("Best Save!")
            best_psnr = lut_metric_log['All']['PSNR'] / lut_metric_log['All']['Samples']
            torch.save(state, os.path.join(opt.save_path, "best.pth"))
        else:
            logger.info("Last Save!")
            torch.save(state, os.path.join(opt.save_path, "last.pth"))
    else:
        if metric_log['All']['PSNR'] / metric_log['All']['Samples'] > best_psnr:
            logger.info("Best Save!")
            best_psnr = metric_log['All']['PSNR'] / metric_log['All']['Samples']
            torch.save(state, os.path.join(opt.save_path, "best.pth"))
        else:
            logger.info("Last Save!")
            torch.save(state, os.path.join(opt.save_path, "last.pth"))
