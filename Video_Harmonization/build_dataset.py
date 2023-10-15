import torch
import cv2
import numpy as np
import torchvision
import os
import random

from utils.misc import prepare_cooridinate_input, customRandomCrop

from datasets.build_INR_dataset import Implicit2DGenerator
import albumentations
from albumentations import Resize, RandomResizedCrop, HorizontalFlip
from torch.utils.data import DataLoader


class dataset_generator(torch.utils.data.Dataset):
    def __init__(self, dataset_txt, alb_transforms, torch_transforms, opt, area_keep_thresh=0.2, mode='Train'):
        super().__init__()

        self.opt = opt
        self.root_path = opt.dataset_path
        self.mode = mode

        self.alb_transforms = alb_transforms
        self.torch_transforms = torch_transforms
        self.kp_t = area_keep_thresh

        self.dataset_samples = []

        with open(dataset_txt, 'r') as f:
            for line in f.readlines():
                real_img_name, cur_mask_name, cur_img_name = line.strip().split()
                cur_img_name = cur_img_name.replace('\\', '/')
                cur_mask_name = cur_mask_name.replace('\\', '/')
                real_img_name = real_img_name.replace('\\', '/')
                cur_img_name = os.path.join(self.root_path, cur_img_name)
                cur_mask_name = os.path.join(self.root_path, cur_mask_name)
                real_img_name = os.path.join(self.root_path, real_img_name)

                all_cur_imgs = os.listdir(cur_img_name)
                all_cur_imgs.sort()

                for name in all_cur_imgs:
                    cur_img_name_ = os.path.join(cur_img_name, name)
                    cur_mask_name_ = os.path.join(cur_mask_name, name.replace(".jpg", ".png"))
                    real_img_name_ = os.path.join(real_img_name, name)

                    self.dataset_samples.append([cur_img_name_, cur_mask_name_, real_img_name_])

        self.INR_dataset = Implicit2DGenerator(opt, self.mode)

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        composite_image, mask, real_image = self.dataset_samples[idx]

        if self.opt.hr_train:
            if self.opt.isFullRes:
                "Since in dataset preprocessing, we resize the image in HAdobe5k to a lower resolution for " \
                "quick loading, we need to change the path here to that of the original resolution of HAdobe5k " \
                "if `opt.isFullRes` is set to True."
                composite_image = composite_image.replace("HAdobe5k", "HAdobe5kori")

        composite_image = cv2.imread(composite_image)
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

        real_image = cv2.imread(real_image)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask)
        mask = mask[:, :, 0].astype(np.float32) / 255.

        """
            If set `opt.hr_train` to True:
        
            Apply multi resolution crop for HR image train. Specifically, for 1024/2048 `input_size` (not fullres), 
            the training phase is first to RandomResizeCrop 1024/2048 `input_size`, then to random crop a `base_size` 
            patch to feed in multiINR process. For inference, just resize it.

            While for fullres, the RandomResizeCrop is removed and just do a random crop. For inference, just keep the size.
            
            BTW, we implement LR and HR mixing train. I.e., the following `random.random() < 0.5`
        """
        if self.opt.hr_train:
            if self.mode == 'Train' and self.opt.isFullRes:
                if random.random() < 0.5:  # LR mix training
                    mixTransform = albumentations.Compose(
                        [
                            RandomResizedCrop(self.opt.base_size, self.opt.base_size, scale=(0.5, 1.0)),
                            HorizontalFlip()],
                        additional_targets={'real_image': 'image', 'object_mask': 'image'}
                    )
                    origin_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
                    origin_bg_ratio = 1 - origin_fg_ratio

                    "Ensure fg and bg not disappear after transformation"
                    valid_augmentation = False
                    transform_out = None
                    time = 0
                    while not valid_augmentation:
                        time += 1
                        # There are some extreme ratio pics, this code is to avoid being hindered by them.
                        if time == 20:
                            tmp_transform = albumentations.Compose(
                                [Resize(self.opt.base_size, self.opt.base_size)],
                                additional_targets={'real_image': 'image',
                                                    'object_mask': 'image'})
                            transform_out = tmp_transform(image=composite_image, real_image=real_image,
                                                          object_mask=mask)
                            valid_augmentation = True
                        else:
                            transform_out = mixTransform(image=composite_image, real_image=real_image,
                                                         object_mask=mask)
                            valid_augmentation = check_augmented_sample(transform_out['object_mask'],
                                                                        origin_fg_ratio,
                                                                        origin_bg_ratio,
                                                                        self.kp_t)
                    composite_image = transform_out['image']
                    real_image = transform_out['real_image']
                    mask = transform_out['object_mask']
                else:  # Padding to ensure that the original resolution can be divided by 4. This is for pixel-aligned crop.
                    if real_image.shape[0] < 256:
                        bottom_pad = 256 - real_image.shape[0]
                    else:
                        bottom_pad = (4 - real_image.shape[0] % 4) % 4
                    if real_image.shape[1] < 256:
                        right_pad = 256 - real_image.shape[1]
                    else:
                        right_pad = (4 - real_image.shape[1] % 4) % 4
                    composite_image = cv2.copyMakeBorder(composite_image, 0, bottom_pad, 0, right_pad,
                                                         cv2.BORDER_REPLICATE)
                    real_image = cv2.copyMakeBorder(real_image, 0, bottom_pad, 0, right_pad, cv2.BORDER_REPLICATE)
                    mask = cv2.copyMakeBorder(mask, 0, bottom_pad, 0, right_pad, cv2.BORDER_REPLICATE)

        origin_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
        origin_bg_ratio = 1 - origin_fg_ratio

        "Ensure fg and bg not disappear after transformation"
        valid_augmentation = False
        transform_out = None
        time = 0

        if self.opt.hr_train:
            if self.mode == 'Train':
                if not self.opt.isFullRes:
                    if random.random() < 0.5:  # LR mix training
                        mixTransform = albumentations.Compose(
                            [
                                RandomResizedCrop(self.opt.base_size, self.opt.base_size, scale=(0.5, 1.0)),
                                HorizontalFlip()],
                            additional_targets={'real_image': 'image', 'object_mask': 'image'}
                        )
                        while not valid_augmentation:
                            time += 1
                            # There are some extreme ratio pics, this code is to avoid being hindered by them.
                            if time == 20:
                                tmp_transform = albumentations.Compose(
                                    [Resize(self.opt.base_size, self.opt.base_size)],
                                    additional_targets={'real_image': 'image',
                                                        'object_mask': 'image'})
                                transform_out = tmp_transform(image=composite_image, real_image=real_image,
                                                              object_mask=mask)
                                valid_augmentation = True
                            else:
                                transform_out = mixTransform(image=composite_image, real_image=real_image,
                                                             object_mask=mask)
                                valid_augmentation = check_augmented_sample(transform_out['object_mask'],
                                                                            origin_fg_ratio,
                                                                            origin_bg_ratio,
                                                                            self.kp_t)
                    else:
                        while not valid_augmentation:
                            time += 1
                            # There are some extreme ratio pics, this code is to avoid being hindered by them.
                            if time == 20:
                                tmp_transform = albumentations.Compose(
                                    [Resize(self.opt.input_size, self.opt.input_size)],
                                    additional_targets={'real_image': 'image',
                                                        'object_mask': 'image'})
                                transform_out = tmp_transform(image=composite_image, real_image=real_image,
                                                              object_mask=mask)
                                valid_augmentation = True
                            else:
                                transform_out = self.alb_transforms(image=composite_image, real_image=real_image,
                                                                    object_mask=mask)
                                valid_augmentation = check_augmented_sample(transform_out['object_mask'],
                                                                            origin_fg_ratio,
                                                                            origin_bg_ratio,
                                                                            self.kp_t)
                    composite_image = transform_out['image']
                    real_image = transform_out['real_image']
                    mask = transform_out['object_mask']

                    origin_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

                full_coord = prepare_cooridinate_input(mask).transpose(1, 2, 0)

                tmp_transform = albumentations.Compose([Resize(self.opt.base_size, self.opt.base_size)],
                                                       additional_targets={'real_image': 'image',
                                                                           'object_mask': 'image'})
                transform_out = tmp_transform(image=composite_image, real_image=real_image, object_mask=mask)
                compos_list = [self.torch_transforms(transform_out['image'])]
                real_list = [self.torch_transforms(transform_out['real_image'])]
                mask_list = [
                    torchvision.transforms.ToTensor()(transform_out['object_mask'][..., np.newaxis].astype(np.float32))]
                coord_map_list = []

                valid_augmentation = False
                while not valid_augmentation:
                    #  RSC strategy. To crop different resolutions.
                    transform_out, c_h, c_w = customRandomCrop([composite_image, real_image, mask, full_coord],
                                                               self.opt.base_size, self.opt.base_size)
                    valid_augmentation = check_hr_crop_sample(transform_out[2], origin_fg_ratio)

                compos_list.append(self.torch_transforms(transform_out[0]))
                real_list.append(self.torch_transforms(transform_out[1]))
                mask_list.append(
                    torchvision.transforms.ToTensor()(transform_out[2][..., np.newaxis].astype(np.float32)))
                coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[3]))
                coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[3]))
                for n in range(2):
                    tmp_comp = cv2.resize(composite_image, (
                        composite_image.shape[1] // 2 ** (n + 1), composite_image.shape[0] // 2 ** (n + 1)))
                    tmp_real = cv2.resize(real_image,
                                          (real_image.shape[1] // 2 ** (n + 1), real_image.shape[0] // 2 ** (n + 1)))
                    tmp_mask = cv2.resize(mask, (mask.shape[1] // 2 ** (n + 1), mask.shape[0] // 2 ** (n + 1)))
                    tmp_coord = prepare_cooridinate_input(tmp_mask).transpose(1, 2, 0)

                    transform_out, c_h, c_w = customRandomCrop([tmp_comp, tmp_real, tmp_mask, tmp_coord],
                                                               self.opt.base_size // 2 ** (n + 1),
                                                               self.opt.base_size // 2 ** (n + 1), c_h, c_w)
                    compos_list.append(self.torch_transforms(transform_out[0]))
                    real_list.append(self.torch_transforms(transform_out[1]))
                    mask_list.append(
                        torchvision.transforms.ToTensor()(transform_out[2][..., np.newaxis].astype(np.float32)))
                    coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[3]))
                out_comp = compos_list
                out_real = real_list
                out_mask = mask_list
                out_coord = coord_map_list

                fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
                    self.torch_transforms, transform_out[0], transform_out[1], mask)

                return {
                    'file_path': self.dataset_samples[idx],
                    'category': self.dataset_samples[idx].split("\\")[-1].split("/")[0],
                    'composite_image': out_comp,
                    'real_image': out_real,
                    'mask': out_mask,
                    'coordinate_map': out_coord,
                    'composite_image0': out_comp[0],
                    'real_image0': out_real[0],
                    'mask0': out_mask[0],
                    'coordinate_map0': out_coord[0],
                    'composite_image1': out_comp[1],
                    'real_image1': out_real[1],
                    'mask1': out_mask[1],
                    'coordinate_map1': out_coord[1],
                    'composite_image2': out_comp[2],
                    'real_image2': out_real[2],
                    'mask2': out_mask[2],
                    'coordinate_map2': out_coord[2],
                    'composite_image3': out_comp[3],
                    'real_image3': out_real[3],
                    'mask3': out_mask[3],
                    'coordinate_map3': out_coord[3],
                    'fg_INR_coordinates': fg_INR_coordinates,
                    'bg_INR_coordinates': bg_INR_coordinates,
                    'fg_INR_RGB': fg_INR_RGB,
                    'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
                    'bg_INR_RGB': bg_INR_RGB
                }
            else:
                if not self.opt.isFullRes:
                    tmp_transform = albumentations.Compose([Resize(self.opt.input_size, self.opt.input_size)],
                                                           additional_targets={'real_image': 'image',
                                                                               'object_mask': 'image'})
                    transform_out = tmp_transform(image=composite_image, real_image=real_image, object_mask=mask)

                    coordinate_map = prepare_cooridinate_input(transform_out['object_mask'])

                    "Generate INR dataset."
                    mask = (torchvision.transforms.ToTensor()(
                        transform_out['object_mask']).squeeze() > 100 / 255.).view(-1)
                    mask = np.bool_(mask.numpy())

                    fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
                        self.torch_transforms, transform_out['image'], transform_out['real_image'], mask)

                    return {
                        'file_path': self.dataset_samples[idx],
                        'category': self.dataset_samples[idx].split("\\")[-1].split("/")[0],
                        'composite_image': self.torch_transforms(transform_out['image']),
                        'real_image': self.torch_transforms(transform_out['real_image']),
                        'mask': transform_out['object_mask'][np.newaxis, ...].astype(np.float32),
                        # Can automatically transfer to Tensor.
                        'coordinate_map': coordinate_map,
                        'fg_INR_coordinates': fg_INR_coordinates,
                        'bg_INR_coordinates': bg_INR_coordinates,
                        'fg_INR_RGB': fg_INR_RGB,
                        'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
                        'bg_INR_RGB': bg_INR_RGB
                    }
                else:
                    coordinate_map = prepare_cooridinate_input(mask)

                    "Generate INR dataset."
                    mask_tmp = (torchvision.transforms.ToTensor()(mask).squeeze() > 100 / 255.).view(-1)
                    mask_tmp = np.bool_(mask_tmp.numpy())

                    fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
                        self.torch_transforms, composite_image, real_image, mask_tmp)

                    return {
                        'file_path': self.dataset_samples[idx],
                        'category': self.dataset_samples[idx].split("\\")[-1].split("/")[0],
                        'composite_image': self.torch_transforms(composite_image),
                        'real_image': self.torch_transforms(real_image),
                        'mask': mask[np.newaxis, ...].astype(np.float32),
                        # Can automatically transfer to Tensor.
                        'coordinate_map': coordinate_map,
                        'fg_INR_coordinates': fg_INR_coordinates,
                        'bg_INR_coordinates': bg_INR_coordinates,
                        'fg_INR_RGB': fg_INR_RGB,
                        'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
                        'bg_INR_RGB': bg_INR_RGB
                    }

        while not valid_augmentation:
            time += 1
            # There are some extreme ratio pics, this code is to avoid being hindered by them.
            if time == 20:
                tmp_transform = albumentations.Compose([Resize(self.opt.input_size, self.opt.input_size)],
                                                       additional_targets={'real_image': 'image',
                                                                           'object_mask': 'image'})
                transform_out = tmp_transform(image=composite_image, real_image=real_image, object_mask=mask)
                valid_augmentation = True
            else:
                transform_out = self.alb_transforms(image=composite_image, real_image=real_image, object_mask=mask)
                valid_augmentation = check_augmented_sample(transform_out['object_mask'], origin_fg_ratio,
                                                            origin_bg_ratio,
                                                            self.kp_t)

        coordinate_map = prepare_cooridinate_input(transform_out['object_mask'])

        "Generate INR dataset."
        mask = (torchvision.transforms.ToTensor()(transform_out['object_mask']).squeeze() > 100 / 255.).view(-1)
        mask = np.bool_(mask.numpy())

        fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
            self.torch_transforms, transform_out['image'], transform_out['real_image'], mask)

        return {
            'file_path': self.dataset_samples[idx][0],
            'category': 'HYouTube',
            'composite_image': self.torch_transforms(transform_out['image']),
            'real_image': self.torch_transforms(transform_out['real_image']),
            'mask': transform_out['object_mask'][np.newaxis, ...].astype(np.float32),
            # Can automatically transfer to Tensor.
            'coordinate_map': coordinate_map,
            'fg_INR_coordinates': fg_INR_coordinates,
            'bg_INR_coordinates': bg_INR_coordinates,
            'fg_INR_RGB': fg_INR_RGB,
            'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
            'bg_INR_RGB': bg_INR_RGB
        }


def check_augmented_sample(mask, origin_fg_ratio, origin_bg_ratio, area_keep_thresh):
    current_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
    current_bg_ratio = 1 - current_fg_ratio

    if current_fg_ratio < origin_fg_ratio * area_keep_thresh or current_bg_ratio < origin_bg_ratio * area_keep_thresh:
        return False

    return True


def check_hr_crop_sample(mask, origin_fg_ratio):
    current_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

    if current_fg_ratio < 0.8 * origin_fg_ratio:
        return False

    return True
