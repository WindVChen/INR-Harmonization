import skimage

import torch
import numpy as np
from pytorch_msssim import ssim
import math


def calc_metrics(harmonized, real, mask_batch):
    n, c, h, w = harmonized.shape

    mse = []
    fmse = []
    psnr = []
    ssim = []
    for id in range(n):
        # fg = (mask_batch[id]).view(-1)
        # fg_pixels = int(torch.sum(fg).cpu().numpy())
        # total_pixels = h * w
        #
        # pred = torch.clamp(harmonized[id] * 255, 0, 255)
        # gt = torch.clamp(real[id] * 255, 0, 255)
        #
        # pred = pred.permute(1, 2, 0).cpu().numpy()
        # gt = gt.permute(1, 2, 0).cpu().numpy()
        # mask = mask_batch[id].permute(1, 2, 0).cpu().numpy()
        #
        # mse.append(skimage.metrics.mean_squared_error(pred, gt))
        # fmse.append(skimage.metrics.mean_squared_error(pred * mask, gt * mask) * total_pixels / fg_pixels)
        # psnr.append(skimage.metrics.peak_signal_noise_ratio(pred, gt, data_range=pred.max() - pred.min()))
        # ssim.append(skimage.metrics.structural_similarity(pred, gt, multichannel=True))
        mse.append(MSE(torch.clamp(harmonized[id] * 255, 0, 255), torch.clamp(real[id] * 255, 0, 255), mask_batch[id]))
        fmse.append(fMSE(torch.clamp(harmonized[id] * 255, 0, 255), torch.clamp(real[id] * 255, 0, 255), mask_batch[id]))
        psnr.append(PSNR(torch.clamp(harmonized[id] * 255, 0, 255), torch.clamp(real[id] * 255, 0, 255), mask_batch[id]))
        ssim.append(SSIM(torch.clamp(harmonized[id] * 255, 0, 255), torch.clamp(real[id] * 255, 0, 255), mask_batch[id]))

    return mse, fmse, psnr, ssim


def SSIM(pred, target_image, mask):
    pred = pred * mask + (target_image) * (1 - mask)
    return ssim(pred.unsqueeze(0), target_image.unsqueeze(0))


def MSE(pred, target_image, mask):
    return (mask * (pred - target_image) ** 2).mean().item()


def fMSE(pred, target_image, mask):
    diff = mask * ((pred - target_image) ** 2)
    return (diff.sum() / (diff.size(0) * mask.sum() + 1e-6)).item()


def PSNR(pred, target_image, mask):
    mse = (mask * (pred - target_image) ** 2).mean().item()
    squared_max = target_image.max().item() ** 2

    return 10 * math.log10(squared_max / (mse + 1e-6))