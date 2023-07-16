import re
from pathlib import Path
import glob
import logging
import numpy as np
import torch
import cv2
import os
import math
from adamp import AdamP
import random
import torch.nn as nn

_logger = None


def increment_path(path):
    # Increment path, i.e. runs/exp1 --> runs/exp{sep}1, runs/exp{sep}2 etc.
    res = re.search("\d+", path)
    if res is None:
        print("Set initial exp number!")
        exit(1)

    if not Path(path).exists():
        return str(path)
    else:
        path = path[:res.start()]
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % Path(path).stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1  # increment number
        return f"{path}{n}"  # update path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, fmt=':f'):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(log_file, level=logging.INFO):
    global _logger
    _logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    _logger.setLevel(level)
    _logger.addHandler(fh)
    _logger.addHandler(sh)

    return _logger


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        if isinstance(image_resolution, int):
            image_resolution = (image_resolution, image_resolution)
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)


def normalize(x, opt, mode='normal'):
    device = x.device
    mean = torch.tensor(np.array(opt.transform_mean), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].to(device)
    var = torch.tensor(np.array(opt.transform_var), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].to(device)
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean


def prepare_cooridinate_input(mask, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if mask.shape[0] == mask.shape[1]:
        sidelen = mask.shape[0]
    else:
        sidelen = mask.shape[:2]

    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    return pixel_coords.squeeze(0).transpose(2, 0, 1)


def visualize(real, composite, mask, pred_fg, pred_harmonized, lut_transform_image, opt, epoch,
              show=False, wandb=True, isAll=False, step=None):
    save_path = os.path.join(opt.save_path, "figs", str(epoch))
    os.makedirs(save_path, exist_ok=True)

    if isAll:
        final_index = 1

        """
            Uncomment the following code if you want to save all the results, otherwise will only save the first image
            of each batch
        """
        # final_index = len(real)
    else:
        final_index = 1

    for id in range(final_index):
        if show:
            cv2.imshow("pred_fg", normalize(pred_fg, opt, 'inv')[id].permute(1, 2, 0).cpu().numpy())
            cv2.imshow("real", normalize(real, opt, 'inv')[id].permute(1, 2, 0).cpu().numpy())
            cv2.imshow("lut_transform", normalize(lut_transform_image, opt, 'inv')[id].permute(1, 2, 0).cpu().numpy())
            cv2.imshow("composite", normalize(composite, opt, 'inv')[id].permute(1, 2, 0).cpu().numpy())
            cv2.imshow("mask", mask[id].permute(1, 2, 0).cpu().numpy())
            cv2.imshow("pred_harmonized_image",
                       normalize(pred_harmonized, opt, 'inv')[id].permute(1, 2, 0).cpu().numpy())
            cv2.waitKey()

        if not opt.INRDecode:
            real_tmp = cv2.cvtColor(
                normalize(real, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(
                    np.uint8),
                cv2.COLOR_RGB2BGR)
            composite_tmp = cv2.cvtColor(
                normalize(composite, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(
                    np.uint8), cv2.COLOR_RGB2BGR)
            mask_tmp = mask[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(np.uint8)
            lut_transform_image_tmp = cv2.cvtColor(
                normalize(lut_transform_image, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(
                    0., 255.).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            pred_fg_tmp = cv2.cvtColor(
                normalize(pred_fg, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(
                    np.uint8), cv2.COLOR_RGB2BGR)
            real_tmp = cv2.cvtColor(
                normalize(real, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(
                    np.uint8),
                cv2.COLOR_RGB2BGR)
            composite_tmp = cv2.cvtColor(
                normalize(composite, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(
                    np.uint8), cv2.COLOR_RGB2BGR)
            lut_transform_image_tmp = cv2.cvtColor(
                normalize(lut_transform_image, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(
                    0., 255.).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            mask_tmp = mask[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(0., 255.).numpy().astype(np.uint8)
            pred_harmonized_tmp = cv2.cvtColor(
                normalize(pred_harmonized, opt, 'inv')[id].permute(1, 2, 0).cpu().mul_(255.).clamp_(
                    0., 255.).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

        if isAll:
            cv2.imwrite(os.path.join(save_path, f"{step}_{id}_composite.jpg"), composite_tmp)
            cv2.imwrite(os.path.join(save_path, f"{step}_{id}_real.jpg"), real_tmp)
            if opt.INRDecode:
                cv2.imwrite(os.path.join(save_path, f"{step}_{id}_pred_harmonized_image.jpg"), pred_harmonized_tmp)
            cv2.imwrite(os.path.join(save_path, f"{step}_{id}_lut_transform_image.jpg"), lut_transform_image_tmp)
            cv2.imwrite(os.path.join(save_path, f"{step}_{id}_mask.jpg"), mask_tmp)
        else:
            if not opt.INRDecode:
                cv2.imwrite(os.path.join(save_path, f"real_{step}_{id}.jpg"), real_tmp)
                cv2.imwrite(os.path.join(save_path, f"composite_{step}_{id}.jpg"), composite_tmp)
                cv2.imwrite(os.path.join(save_path, f"mask_{step}_{id}.jpg"), mask_tmp)
                cv2.imwrite(os.path.join(save_path, f"lut_transform_image_{step}_{id}.jpg"), lut_transform_image_tmp)
            else:
                cv2.imwrite(os.path.join(save_path, f"pred_fg_{step}_{id}.jpg"), pred_fg_tmp)
                cv2.imwrite(os.path.join(save_path, f"real_{step}_{id}.jpg"), real_tmp)
                cv2.imwrite(os.path.join(save_path, f"composite_{step}_{id}.jpg"), composite_tmp)
                cv2.imwrite(os.path.join(save_path, f"mask_{step}_{id}.jpg"), mask_tmp)
                cv2.imwrite(os.path.join(save_path, f"pred_harmonized_image_{step}_{id}.jpg"), pred_harmonized_tmp)
                cv2.imwrite(os.path.join(save_path, f"lut_transform_image_{step}_{id}.jpg"), lut_transform_image_tmp)

        "Only upload images of the first batch of the first epoch to save storage."
        if wandb and id == 0 and step == 0:
            import wandb
            real_tmp = wandb.Image(real_tmp, caption=epoch)
            composite_tmp = wandb.Image(composite_tmp, caption=epoch)
            if opt.INRDecode:
                pred_fg_tmp = wandb.Image(pred_fg_tmp, caption=epoch)
                pred_harmonized_tmp = wandb.Image(pred_harmonized_tmp, caption=epoch)
            lut_transform_image_tmp = wandb.Image(lut_transform_image_tmp, caption=epoch)
            mask_tmp = wandb.Image(mask_tmp, caption=epoch)
            if not opt.INRDecode:
                wandb.log(
                    {"pic/real": real_tmp, "pic/composite": composite_tmp,
                     "pic/mask": mask_tmp,
                     "pic/lut_trans": lut_transform_image_tmp,
                     "pic/epoch": epoch})
            else:
                wandb.log(
                    {"pic/pred_fg": pred_fg_tmp, "pic/real": real_tmp, "pic/composite": composite_tmp,
                     "pic/mask": mask_tmp,
                     "pic/lut_trans": lut_transform_image_tmp,
                     "pic/pred_harmonized": pred_harmonized_tmp,
                     "pic/epoch": epoch})
            wandb.log({})


def get_optimizer(model, opt_name, opt_kwargs):
    params = []
    base_lr = opt_kwargs['lr']
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            # print(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult

        params.append(param_group)

    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'adamp': AdamP
    }[opt_name.lower()](params, **opt_kwargs)

    return optimizer


def improved_efficient_matmul(a, c, index, batch=256):
    """
    Reduce the unneed memory cost, but the speed is very slow.
    :param a: N * I * J
    :param b: N * J * K
    :return:  N * I * K
    """
    "The first can only support when a is not requires_grad_, and have high speed. While the second one supports "
    "whatever situations, but speed is quite slow. More Details in "
    "https://discuss.pytorch.org/t/many-weird-phenomena-about-torch-matmul-operation/158208"

    # out = torch.cat(
    #     [torch.matmul(a[i * batch:i * batch + batch, :, :], c[index[i * batch:i * batch + batch], :, :]) for i in
    #      range(a.shape[0] // batch)], dim=0)

    batch = 1
    out = torch.cat(
        [torch.matmul(a[i * batch:i * batch + batch, :, :], c[index[i * batch], :, :]) for i in
         range(a.shape[0] // batch)], dim=0)

    return out


class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult


def customRandomCrop(objects, crop_height, crop_width, h_start=None, w_start=None):
    if h_start is None:
        h_start = random.random()
    if w_start is None:
        w_start = random.random()
    if isinstance(objects, list):
        out = []
        for obj in objects:
            out.append(random_crop(obj, crop_height, crop_width, h_start, w_start))

    else:
        out = random_crop(objects, crop_height, crop_width, h_start, w_start)

    return out, h_start, w_start


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float,
                           w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


class PadToDivisor:
    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def transform(self, images):

        self._pads = (*self._get_dim_padding(images[0].shape[-1]), *self._get_dim_padding(images[0].shape[-2]))
        self.pad_operation = nn.ZeroPad2d(padding=self._pads)

        out = []
        for im in images:
            out.append(self.pad_operation(im))

        return out

    def inv_transform(self, image):
        assert self._pads is not None,\
            'Something went wrong, inv_transform(...) should be called after transform(...)'
        return self._remove_padding(image)

    def _get_dim_padding(self, dim_size):
        pad = (self.divisor - dim_size % self.divisor) % self.divisor
        pad_upper = pad // 2
        pad_lower = pad - pad_upper

        return pad_upper, pad_lower

    def _remove_padding(self, tensors):
        tensor_h, tensor_w = tensors[0].shape[-2:]
        out = []
        for t in tensors:
            out.append(t[..., self._pads[2]:tensor_h - self._pads[3], self._pads[0]:tensor_w - self._pads[1]])
        return out
