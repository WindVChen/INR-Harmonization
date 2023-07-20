import argparse

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.build_model import build_model

import torch
import cv2
import numpy as np
import torchvision
import os
import tqdm
import time

from utils.misc import prepare_cooridinate_input, customRandomCrop

from datasets.build_INR_dataset import Implicit2DGenerator
import albumentations
from albumentations import Resize
from torch.utils.data import DataLoader
from utils.misc import normalize

import math


class single_image_dataset(torch.utils.data.Dataset):
    def __init__(self, opt, composite_image=None, mask=None):
        super().__init__()

        self.opt = opt

        if composite_image is None:
            composite_image = cv2.imread(opt.composite_image)
            composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        self.composite_image = composite_image

        if mask is None:
            mask = cv2.imread(opt.mask)
        mask = mask[:, :, 0].astype(np.float32) / 255.
        self.mask = mask

        self.torch_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        self.INR_dataset = Implicit2DGenerator(opt, 'Val')

        self.split_width_resolution = composite_image.shape[1] // opt.split_num
        self.split_height_resolution = composite_image.shape[0] // opt.split_num

        self.split_width_resolution = self.split_height_resolution = min(self.split_width_resolution,
                                                                         self.split_height_resolution)

        if self.split_width_resolution % 4 != 0:
            self.split_width_resolution = self.split_width_resolution + (4 - self.split_width_resolution % 4)

        if self.split_height_resolution % 4 != 0:
            self.split_height_resolution = self.split_height_resolution + (4 - self.split_height_resolution % 4)

        self.num_w = math.ceil(composite_image.shape[1] / self.split_width_resolution)
        self.num_h = math.ceil(composite_image.shape[0] / self.split_height_resolution)

        self.split_start_point = []

        "Split the image into several parts."
        for i in range(self.num_h):
            for j in range(self.num_w):
                if i == composite_image.shape[0] // self.split_height_resolution:
                    if j == composite_image.shape[1] // self.split_width_resolution:
                        self.split_start_point.append((composite_image.shape[0] - self.split_height_resolution,
                                                       composite_image.shape[1] - self.split_width_resolution))
                    else:
                        self.split_start_point.append(
                            (composite_image.shape[0] - self.split_height_resolution, j * self.split_width_resolution))
                else:
                    if j == composite_image.shape[1] // self.split_width_resolution:
                        self.split_start_point.append(
                            (i * self.split_height_resolution, composite_image.shape[1] - self.split_width_resolution))
                    else:
                        self.split_start_point.append(
                            (i * self.split_height_resolution, j * self.split_width_resolution))

        assert len(self.split_start_point) == self.num_w * self.num_h

        print(
            f"The image will be split into {self.num_h} pieces in height, and {self.num_w} pieces in width. Totally {self.num_h * self.num_w} patches.")
        print(f"The final resolution of each patch is {self.split_height_resolution} x {self.split_width_resolution}")

    def __len__(self):
        return self.num_w * self.num_h

    def __getitem__(self, idx):
        composite_image = self.composite_image

        mask = self.mask

        full_coord = prepare_cooridinate_input(mask).transpose(1, 2, 0)

        tmp_transform = albumentations.Compose([Resize(self.opt.base_size, self.opt.base_size)],
                                               additional_targets={'object_mask': 'image'})
        transform_out = tmp_transform(image=composite_image, object_mask=mask)
        compos_list = [self.torch_transforms(transform_out['image'])]
        mask_list = [
            torchvision.transforms.ToTensor()(transform_out['object_mask'][..., np.newaxis].astype(np.float32))]
        coord_map_list = []

        if composite_image.shape[0] != self.split_height_resolution:
            c_h = self.split_start_point[idx][0] / (composite_image.shape[0] - self.split_height_resolution)
        else:
            c_h = 0
        if composite_image.shape[1] != self.split_width_resolution:
            c_w = self.split_start_point[idx][1] / (composite_image.shape[1] - self.split_width_resolution)
        else:
            c_w = 0
        transform_out, c_h, c_w = customRandomCrop([composite_image, mask, full_coord],
                                                   self.split_height_resolution, self.split_width_resolution, c_h, c_w)

        compos_list.append(self.torch_transforms(transform_out[0]))
        mask_list.append(
            torchvision.transforms.ToTensor()(transform_out[1][..., np.newaxis].astype(np.float32)))
        coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        for n in range(2):
            tmp_comp = cv2.resize(composite_image, (
                composite_image.shape[1] // 2 ** (n + 1), composite_image.shape[0] // 2 ** (n + 1)))
            tmp_mask = cv2.resize(mask, (mask.shape[1] // 2 ** (n + 1), mask.shape[0] // 2 ** (n + 1)))
            tmp_coord = prepare_cooridinate_input(tmp_mask).transpose(1, 2, 0)

            transform_out, c_h, c_w = customRandomCrop([tmp_comp, tmp_mask, tmp_coord],
                                                       self.split_height_resolution // 2 ** (n + 1),
                                                       self.split_width_resolution // 2 ** (n + 1), c_h, c_w)
            compos_list.append(self.torch_transforms(transform_out[0]))
            mask_list.append(
                torchvision.transforms.ToTensor()(transform_out[1][..., np.newaxis].astype(np.float32)))
            coord_map_list.append(torchvision.transforms.ToTensor()(transform_out[2]))
        out_comp = compos_list
        out_mask = mask_list
        out_coord = coord_map_list

        fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB = self.INR_dataset.generator(
            self.torch_transforms, transform_out[0], transform_out[0], mask)

        return {
            'composite_image': out_comp,
            'mask': out_mask,
            'coordinate_map': out_coord,
            'composite_image0': out_comp[0],
            'mask0': out_mask[0],
            'coordinate_map0': out_coord[0],
            'composite_image1': out_comp[1],
            'mask1': out_mask[1],
            'coordinate_map1': out_coord[1],
            'composite_image2': out_comp[2],
            'mask2': out_mask[2],
            'coordinate_map2': out_coord[2],
            'composite_image3': out_comp[3],
            'mask3': out_mask[3],
            'coordinate_map3': out_coord[3],
            'fg_INR_coordinates': fg_INR_coordinates,
            'bg_INR_coordinates': bg_INR_coordinates,
            'fg_INR_RGB': fg_INR_RGB,
            'fg_transfer_INR_RGB': fg_transfer_INR_RGB,
            'bg_INR_RGB': bg_INR_RGB,
            'start_point': self.split_start_point[idx],
        }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_num', type=int, default=4,
                        help='How many pieces do you want to split an image width / height.')

    parser.add_argument('--composite_image', type=str, default=r'./demo/demo_2k_composite.jpg',
                        help='composite image path')

    parser.add_argument('--mask', type=str, default=r'./demo/demo_2k_mask.jpg',
                        help='mask path')

    parser.add_argument('--save_path', type=str, default=r'./demo/',
                        help='save path')

    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--device', type=str, default='cuda',
                        help="Whether use cuda, 'cuda' or 'cpu'.")

    parser.add_argument('--base_size', type=int, default=256,
                        help='Base size. Resolution of the image input into the Encoder')

    parser.add_argument('--input_size', type=int, default=256,
                        help='Input size. Resolution of the image that want to be generated by the Decoder')

    parser.add_argument('--INR_input_size', type=int, default=256,
                        help='INR input size. Resolution of the image that want to be generated by the Decoder. '
                             'Should be the same as `input_size`')

    parser.add_argument('--INR_MLP_dim', type=int, default=32,
                        help='Number of channels for INR linear layer.')

    parser.add_argument('--LUT_dim', type=int, default=7,
                        help='Dim of the output LUT. Refer to https://ieeexplore.ieee.org/abstract/document/9206076')

    parser.add_argument('--activation', type=str, default='leakyrelu_pe',
                        help='INR activation layer type: leakyrelu_pe, sine')

    parser.add_argument('--pretrained', type=str,
                        default=r'.\pretrained_models\Resolution_RAW_iHarmony4.pth',
                        help='Pretrained weight path')

    parser.add_argument('--param_factorize_dim', type=int,
                        default=10,
                        help='The intermediate dimensions of the factorization of the predicted MLP parameters. '
                             'Refer to https://arxiv.org/abs/2011.12026')

    parser.add_argument('--embedding_type', type=str,
                        default="CIPS_embed",
                        help='Which embedding_type to use.')

    parser.add_argument('--INRDecode', action="store_false",
                        help='Whether INR decoder. Set it to False if you want to test the baseline '
                             '(https://github.com/SamsungLabs/image_harmonization)')

    parser.add_argument('--isMoreINRInput', action="store_false",
                        help='Whether to cat RGB and mask. See Section 3.4 in the paper.')

    parser.add_argument('--hr_train', action="store_false",
                        help='Whether use hr_train. See section 3.4 in the paper.')

    parser.add_argument('--isFullRes', action="store_true",
                        help='Whether for original resolution. See section 3.4 in the paper.')

    opt = parser.parse_args()

    return opt

@torch.no_grad()
def inference(model, opt, composite_image=None, mask=None):
    model.eval()

    "dataset here is actually consisted of several patches of a single image."
    singledataset = single_image_dataset(opt, composite_image, mask)

    single_data_loader = DataLoader(singledataset, opt.batch_size, shuffle=False, drop_last=False, pin_memory=True,
                                    num_workers=opt.workers, persistent_workers=False if composite_image is not None else True)

    "Init a pure black image with the same size as the input image."
    init_img = np.zeros_like(singledataset.composite_image)

    time_all = 0

    for step, batch in tqdm.tqdm(enumerate(single_data_loader)):
        composite_image = [batch[f'composite_image{name}'].to(opt.device) for name in range(4)]
        mask = [batch[f'mask{name}'].to(opt.device) for name in range(4)]
        coordinate_map = [batch[f'coordinate_map{name}'].to(opt.device) for name in range(4)]
        start_points = batch['start_point']

        if opt.batch_size == 1:
            start_points = [torch.cat(start_points)]

        fg_INR_coordinates = coordinate_map[1:]

        try:
            if step == 0:  # This is for CUDA Kernel Warm-up, or the first inference step will be quite slow.
                fg_content_bg_appearance_construct, _, lut_transform_image = model(
                    composite_image,
                    mask,
                    fg_INR_coordinates,
                )
            if opt.device == "cuda":
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()
                start_time = time.time()
                torch.cuda.synchronize()
            fg_content_bg_appearance_construct, _, lut_transform_image = model(
                composite_image,
                mask,
                fg_INR_coordinates,
            )
            if opt.device == "cuda":
                torch.cuda.synchronize()
                end_time = time.time()

                end_max_memory = torch.cuda.max_memory_allocated() // 1024 ** 2
                end_memory = torch.cuda.memory_allocated() // 1024 ** 2

                print(f'GPU max memory usage: {end_max_memory} MB')
                print(f'GPU memory usage: {end_memory} MB')
                time_all += (end_time - start_time)
            print(f'progress: {step} / {len(single_data_loader)}')
        except:
            raise Exception(
                f'The image resolution is large. Please increase the `split_num` value. Your current set is {opt.split_num}')

        "Assemble the every patch's harmonized result into the final whole image."
        for id in range(len(fg_INR_coordinates[0])):
            pred_fg_image = fg_content_bg_appearance_construct[-1][id]
            pred_harmonized_image = pred_fg_image * (mask[1][id] > 100 / 255.) + composite_image[1][id] * (
                ~(mask[1][id] > 100 / 255.))

            pred_harmonized_tmp = cv2.cvtColor(
                normalize(pred_harmonized_image.unsqueeze(0), opt, 'inv')[0].permute(1, 2, 0).cpu().mul_(255.).clamp_(
                    0., 255.).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

            init_img[start_points[id][0]:start_points[id][0] + singledataset.split_height_resolution,
            start_points[id][1]:start_points[id][1] + singledataset.split_width_resolution] = pred_harmonized_tmp

    print(f'Inference time: {time_all}')
    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)
        cv2.imwrite(os.path.join(opt.save_path, "pred_harmonized_image.jpg"), init_img)
    return init_img


def main_process(opt, composite_image=None, mask=None):
    cudnn.benchmark = True

    model = build_model(opt).to(opt.device)

    load_dict = torch.load(opt.pretrained)['model']
    for k in load_dict.keys():
        if k not in model.state_dict().keys():
            print(f"Skip {k}")
    model.load_state_dict(load_dict, strict=False)

    return inference(model, opt, composite_image, mask)


if __name__ == '__main__':
    opt = parse_args()
    opt.transform_mean = [.5, .5, .5]
    opt.transform_var = [.5, .5, .5]
    main_process(opt)
