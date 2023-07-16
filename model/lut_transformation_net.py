import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import normalize


class build_lut_transform(nn.Module):

    def __init__(self, input_dim, lut_dim, input_resolution, opt):
        super().__init__()

        self.lut_dim = lut_dim
        self.opt = opt

        # self.compress_layer = nn.Linear(input_resolution, 1)

        self.transform_layers = nn.Sequential(
            nn.Linear(input_dim, 3 * lut_dim ** 3, bias=True),
            # nn.BatchNorm1d(3 * lut_dim ** 3, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(3 * lut_dim ** 3, 3 * lut_dim ** 3, bias=True),
        )
        self.transform_layers[-1].apply(lambda m: hyper_weight_init(m))

    def forward(self, composite_image, fg_appearance_features, bg_appearance_features):
        composite_image = normalize(composite_image, self.opt, 'inv')

        features = fg_appearance_features

        lut_params = self.transform_layers(features)

        fit_3DLUT = lut_params.view(lut_params.shape[0], 3, self.lut_dim, self.lut_dim, self.lut_dim)

        lut_transform_image = torch.stack(
            [TrilinearInterpolation(lut, image)[0] for lut, image in zip(fit_3DLUT, composite_image)], dim=0)

        return fit_3DLUT, normalize(lut_transform_image, self.opt)


def TrilinearInterpolation(LUT, img):
    img = (img - 0.5) * 2.

    img = img.unsqueeze(0).permute(0, 2, 3, 1)[:, None].flip(-1)

    # Note that the coordinates in the grid_sample are inverse to LUT DHW, i.e., xyz is to WHD not DHW.
    LUT = LUT[None]

    # grid sample
    result = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)

    # drop added dimensions and permute back
    result = result[:, :, 0]

    return result


def hyper_weight_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(0., 1.)
