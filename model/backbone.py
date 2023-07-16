import torch.nn as nn

from .hrnetv2.hrnet_ocr import HighResolutionNet
from .hrnetv2.modifiers import LRMult
from .base.basic_blocks import MaxPoolDownSize
from .base.ih_model import IHModelWithBackbone, DeepImageHarmonization


def build_backbone(name, opt):
    return eval(name)(opt)


class baseline(IHModelWithBackbone):
    def __init__(self, opt, ocr=64):
        base_config = {'model': DeepImageHarmonization,
                       'params': {'depth': 7, 'batchnorm_from': 2, 'image_fusion': True, 'opt': opt}}

        params = base_config['params']

        backbone = HRNetV2(opt, ocr=ocr)

        params.update(dict(
            backbone_from=2,
            backbone_channels=backbone.output_channels,
            backbone_mode='cat',
            opt=opt
        ))
        base_model = base_config['model'](**params)

        super(baseline, self).__init__(base_model, backbone, False, 'sum', opt=opt)


class HRNetV2(nn.Module):
    def __init__(
            self, opt,
            cat_outputs=True,
            pyramid_channels=-1, pyramid_depth=4,
            width=18, ocr=128, small=False,
            lr_mult=0.1, pretained=True
    ):
        super(HRNetV2, self).__init__()
        self.opt = opt
        self.cat_outputs = cat_outputs
        self.ocr_on = ocr > 0 and cat_outputs
        self.pyramid_on = pyramid_channels > 0 and cat_outputs

        self.hrnet = HighResolutionNet(width, 2, ocr_width=ocr, small=small, opt=opt)
        self.hrnet.apply(LRMult(lr_mult))
        if self.ocr_on:
            self.hrnet.ocr_distri_head.apply(LRMult(1.0))
            self.hrnet.ocr_gather_head.apply(LRMult(1.0))
            self.hrnet.conv3x3_ocr.apply(LRMult(1.0))

        hrnet_cat_channels = [width * 2 ** i for i in range(4)]
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        elif self.ocr_on:
            self.output_channels = [ocr * 2]
        elif self.cat_outputs:
            self.output_channels = [sum(hrnet_cat_channels)]
        else:
            self.output_channels = hrnet_cat_channels

        if self.pyramid_on:
            downsize_in_channels = ocr * 2 if self.ocr_on else sum(hrnet_cat_channels)
            self.downsize = MaxPoolDownSize(downsize_in_channels, pyramid_channels, pyramid_channels, pyramid_depth)

        if pretained:
            self.load_pretrained_weights(
                r".\pretrained_models/hrnetv2_w18_imagenet_pretrained.pth")

        self.output_resolution = (opt.input_size // 8) ** 2

    def forward(self, image, mask, mask_features=None):
        outputs = list(self.hrnet(image, mask, mask_features))
        return outputs

    def load_pretrained_weights(self, pretrained_path):
        self.hrnet.load_pretrained_weights(pretrained_path)
