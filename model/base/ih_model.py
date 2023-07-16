import torch
import torchvision
import torch.nn as nn

from .conv_autoencoder import ConvEncoder, DeconvDecoder, INRDecoder, onlyVector, onlyMatrix, onlyLastLayer

from .ops import ScaleLayer


class IHModelWithBackbone(nn.Module):
    def __init__(
            self,
            model, backbone,
            downsize_backbone_input=False,
            mask_fusion='sum',
            backbone_conv1_channels=64, opt=None
    ):
        super(IHModelWithBackbone, self).__init__()
        self.downsize_backbone_input = downsize_backbone_input
        self.mask_fusion = mask_fusion

        self.backbone = backbone
        self.model = model
        self.opt = opt

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, backbone_conv1_channels, kernel_size=3, stride=2, padding=1, bias=True),
            ScaleLayer(init_value=0.1, lr_mult=1)
        )

    def forward(self, image, mask, coord=None):
        if self.opt.INRDecode and self.opt.hr_train and self.training:
            backbone_image = torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(image[0])
            backbone_mask = torch.cat(
                (torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask[0]),
                 1.0 - torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask[0])), dim=1)
        else:
            backbone_image = torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(image)
            backbone_mask = torch.cat((torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask),
                                       1.0 - torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask)), dim=1)

        backbone_mask_features = self.mask_conv(backbone_mask[:, :1])
        backbone_features = self.backbone(backbone_image, backbone_mask, backbone_mask_features)

        output = self.model(image, mask, backbone_features, coord=coord)
        return output


class DeepImageHarmonization(nn.Module):
    def __init__(
            self,
            depth,
            norm_layer=nn.BatchNorm2d, batchnorm_from=0,
            attend_from=-1,
            image_fusion=False,
            ch=64, max_channels=512,
            backbone_from=-1, backbone_channels=None, backbone_mode='', opt=None
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode, INRDecode=opt.INRDecode
        )
        self.opt = opt
        if opt.INRDecode:
            "See Table 2 in the paper to test with different INR decoders' structures."
            self.decoder = INRDecoder(depth, self.encoder.blocks_channels, norm_layer, opt, backbone_from)
            # self.decoder = onlyLastLayer(depth, self.encoder.blocks_channels, norm_layer, opt, backbone_from)
            # self.decoder = onlyVector(depth, self.encoder.blocks_channels, norm_layer, opt, backbone_from)
            # self.decoder = onlyMatrix(depth, self.encoder.blocks_channels, norm_layer, opt, backbone_from)
        else:
            "Baseline: https://github.com/SamsungLabs/image_harmonization"
            self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, mask, backbone_features=None, coord=None):
        if self.opt.INRDecode and self.opt.hr_train and self.training:
            x = torch.cat((torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(image[0]),
                           torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask[0])), dim=1)
        else:
            x = torch.cat((torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(image),
                           torchvision.transforms.Resize([self.opt.base_size, self.opt.base_size])(mask)), dim=1)

        intermediates = self.encoder(x, backbone_features)

        if self.opt.INRDecode and self.opt.hr_train and self.training:
            output = self.decoder(intermediates, image[1], mask[1], coord_samples=coord)
        else:
            output = self.decoder(intermediates, image, mask)
        return output
