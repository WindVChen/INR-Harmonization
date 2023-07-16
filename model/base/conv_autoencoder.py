import torch
import torchvision
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .basic_blocks import ConvBlock, lineParams, convParams
from .ops import MaskedChannelAttention, FeaturesConnector
from .ops import PosEncodingNeRF, INRGAN_embed, RandomFourier, CIPS_embed
from utils import misc
from utils.misc import lin2img
from ..lut_transformation_net import build_lut_transform


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class Leaky_relu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.leaky_relu(input, 0.01, inplace=True)


def select_activation(type):
    if type == 'sine':
        return Sine()
    elif type == 'leakyrelu_pe':
        return Leaky_relu()
    else:
        raise NotImplementedError


class ConvEncoder(nn.Module):
    def __init__(
            self,
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels=None, backbone_mode='', INRDecode=False
    ):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.INRDecode = INRDecode
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = ConvBlock(in_channels, out_channels, norm_layer=norm_layer if batchnorm_from == 0 else None)
        self.block1 = ConvBlock(out_channels, out_channels, norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None)
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                if INRDecode:
                    self.blocks_connected[f'block{block_i}_decode'] = ConvBlock(
                        in_channels, out_channels,
                        norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                        padding=int(block_i < depth - 1)
                    )
                    self.blocks_channels += [out_channels]
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels

            self.blocks_connected[f'block{block_i}'] = ConvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=int(block_i < depth - 1)
            )
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            output = outputs[-1]
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                if self.INRDecode:
                    block = self.blocks_connected[f'block{block_i}_decode']
                    outputs += [block(output)]

                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            block = self.blocks_connected[f'block{block_i}']
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attend_from=-1, image_fusion=False):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=0 <= attend_from <= d
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output


class SEDeconvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
            norm_layer=nn.BatchNorm2d, activation=nn.ELU,
            with_se=False
    ):
        super(SEDeconvBlock, self).__init__()
        self.with_se = with_se
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.with_se:
            self.se = MaskedChannelAttention(out_channels)

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_se:
            out = self.se(out, mask)
        return out


class INRDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, opt, attend_from):
        super(INRDecoder, self).__init__()
        self.INR_encoding = None
        if opt.embedding_type == "PosEncodingNeRF":
            self.INR_encoding = PosEncodingNeRF(in_features=2, sidelength=opt.input_size)
        elif opt.embedding_type == "RandomFourier":
            self.INR_encoding = RandomFourier(std_scale=10, embedding_length=64, device=opt.device)
        elif opt.embedding_type == "CIPS_embed":
            self.INR_encoding = CIPS_embed(size=opt.base_size, embedding_length=32)
        elif opt.embedding_type == "INRGAN_embed":
            self.INR_encoding = INRGAN_embed(resolution=opt.INR_input_size)
        else:
            raise NotImplementedError
        encoder_blocks_channels = encoder_blocks_channels[::-1]
        max_hidden_mlp_num = attend_from + 1
        self.opt = opt
        self.max_hidden_mlp_num = max_hidden_mlp_num
        self.content_mlp_blocks = nn.ModuleDict()
        for n in range(max_hidden_mlp_num):
            if n != max_hidden_mlp_num - 1:
                self.content_mlp_blocks[f"block{n}"] = convParams(encoder_blocks_channels.pop(),
                                            [self.INR_encoding.out_dim + opt.INR_MLP_dim + (4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim], opt, n + 1)
            else:
                self.content_mlp_blocks[f"block{n}"] = convParams(encoder_blocks_channels.pop(),
                                            [self.INR_encoding.out_dim + (4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim], opt, n + 1)

        self.deconv_blocks = nn.ModuleList()

        encoder_blocks_channels = encoder_blocks_channels[::-1]
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - attend_from):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=False
            ))
            in_channels = out_channels

        self.appearance_mlps = lineParams(out_channels, [opt.INR_MLP_dim, opt.INR_MLP_dim],
                                          (opt.base_size // (2 ** (max_hidden_mlp_num - 1))) ** 2,
                                          opt, 2, toRGB=True)

        self.lut_transform = build_lut_transform(self.appearance_mlps.output_dim, opt.LUT_dim,
                                                 None, opt)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, encoder_outputs, image=None, mask=None, coord_samples=None):
        """For full resolution, do split."""
        if self.opt.hr_train and not self.training and self.opt.isFullRes:
            return self.forward_fullResInference(encoder_outputs, image=image, mask=mask, coord_samples=coord_samples)

        encoder_outputs = encoder_outputs[::-1]
        mlp_output = None
        waitToRGB = []
        for n in range(self.max_hidden_mlp_num):
            if not self.opt.hr_train:
                coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1)))\
                    .unsqueeze(0).repeat(encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
            else:
                if self.training:
                    coord = coord_samples[self.max_hidden_mlp_num - n - 1].permute(0, 2, 3, 1).view(encoder_outputs[0].shape[0], -1, 2)
                else:
                    coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)

            """Whether to leverage multiple input to INR decoder. See Section 3.4 in the paper."""
            if self.opt.isMoreINRInput:
                if not self.opt.isFullRes or self.training:
                    res_h = res_w = np.sqrt(coord.shape[1]).astype(int)
                else:
                    res_h = image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1))
                    res_w = image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))

                res_image = torchvision.transforms.Resize([res_h, res_w])(image)
                res_mask = torchvision.transforms.Resize([res_h, res_w])(mask)
                coord = torch.cat([self.INR_encoding(coord), res_image.view(*res_image.shape[:2], -1).permute(0, 2, 1),
                                   res_mask.view(*res_mask.shape[:2], -1).permute(0, 2, 1)], dim=-1)
            else:
                coord = self.INR_encoding(coord)

            """============ LRIP structure, see Section 3.3 =============="""

            """Local MLPs."""
            if n == 0:
                mlp_output = self.mlp_process(coord, self.INR_encoding.out_dim + (4 if self.opt.isMoreINRInput else 0),
                                              self.opt, content_mlp=self.content_mlp_blocks[f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n)))
                waitToRGB.append(mlp_output[1])
            else:
                mlp_output = self.mlp_process(coord, self.opt.INR_MLP_dim + self.INR_encoding.out_dim + (
                    4 if self.opt.isMoreINRInput else 0), self.opt, base_feat=mlp_output[0], content_mlp=self.content_mlp_blocks[
                                                  f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n)))
                waitToRGB.append(mlp_output[1])

        encoder_outputs = encoder_outputs[::-1]
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output)
            output = output + skip_output
        output = self.deconv_blocks[-1](output)

        """Global MLPs."""
        app_mlp, app_params = self.appearance_mlps(output)
        harm_out = []
        for id in range(len(waitToRGB)):
            output = self.mlp_process(None, self.opt.INR_MLP_dim, self.opt, base_feat=waitToRGB[id],
                                      appearance_mlp=app_mlp)
            harm_out.append(output[0])

        """Optional 3D LUT prediction."""
        fit_lut3d, lut_transform_image = self.lut_transform(image, app_params, None)

        return harm_out, fit_lut3d, lut_transform_image

    def mlp_process(self, coorinates, INR_input_dim, opt, base_feat=None, content_mlp=None, appearance_mlp=None, resolution=None):

        activation = select_activation(opt.activation)

        output = None

        if content_mlp is not None:
            if base_feat is not None:
                coorinates = torch.cat([coorinates, base_feat], dim=2)
            coorinates = lin2img(coorinates, resolution)

            k_h = coorinates.shape[2] // content_mlp[0][0].shape[1]
            k_w = coorinates.shape[3] // content_mlp[0][0].shape[1]
            bs = coorinates.shape[0]
            h_lr = w_lr = content_mlp[0][0].shape[1]
            nci = INR_input_dim

            """(evaluation or not HR training) and not fullres evaluation"""
            if (not self.opt.hr_train or not self.training) and not (not self.training and self.opt.isFullRes and self.opt.hr_train):
                coorinates = coorinates.unfold(2, k_h, k_h).unfold(3, k_w, k_w)
                coorinates = coorinates.permute(0, 2, 3, 4, 5, 1).contiguous().view(
                    bs, h_lr, w_lr, int(k_h * k_w), nci)

                for id, layer in enumerate(content_mlp):
                    if id == 0:
                        output = torch.matmul(coorinates, layer[0]) + layer[1]
                        output = activation(output)
                    else:
                        output = torch.matmul(output, layer[0]) + layer[1]
                        output = activation(output)

                output = output.view(bs, h_lr, w_lr, k_h, k_w, opt.INR_MLP_dim).permute(
                    0, 1, 3, 2, 4, 5).contiguous().view(bs, -1, opt.INR_MLP_dim)

                output_large = self.up(lin2img(output))

                return output_large.view(bs, -1, opt.INR_MLP_dim), output
            else:
                coorinates = coorinates.permute(0, 2, 3, 1)
                for id, layer in enumerate(content_mlp):
                    weigt_shape = layer[0].shape
                    bias_shape = layer[1].shape
                    layer[0] = layer[0].view(*layer[0].shape[:-2], -1).permute(0, 3, 1, 2).contiguous()
                    layer[1] = layer[1].view(*layer[1].shape[:-2], -1).permute(0, 3, 1, 2).contiguous()
                    layer[0] = F.grid_sample(layer[0], coorinates[..., :2].flip(-1), mode='nearest' if True
                    else 'bilinear', padding_mode='border', align_corners=False)
                    layer[1] = F.grid_sample(layer[1], coorinates[..., :2].flip(-1), mode='nearest' if True
                    else 'bilinear', padding_mode='border', align_corners=False)
                    layer[0] = layer[0].permute(0, 2, 3, 1).contiguous().view(*coorinates.shape[:-1], *weigt_shape[-2:])
                    layer[1] = layer[1].permute(0, 2, 3, 1).contiguous().view(*coorinates.shape[:-1], *bias_shape[-2:])

                    if id == 0:
                        output = torch.matmul(coorinates.unsqueeze(-2), layer[0]) + layer[1]
                        output = activation(output)
                    else:
                        output = torch.matmul(output, layer[0]) + layer[1]
                        output = activation(output)

                output = output.squeeze(-2).view(bs, -1, opt.INR_MLP_dim)

                output_large = self.up(lin2img(output, resolution))

                return output_large.view(bs, -1, opt.INR_MLP_dim), output

        elif appearance_mlp is not None:
            output = base_feat
            genMask = None
            for id, layer in enumerate(appearance_mlp):
                if id != len(appearance_mlp) - 1:
                    output = torch.matmul(output, layer[0]) + layer[1]
                    output = activation(output)
                else:
                    output = torch.matmul(output, layer[0]) + layer[1]  # last layer
                    if opt.activation == 'leakyrelu_pe':
                        output = torch.tanh(output)
            return lin2img(output, resolution), None

    def forward_fullResInference(self, encoder_outputs, image=None, mask=None, coord_samples=None):
        encoder_outputs = encoder_outputs[::-1]
        mlp_output = None
        res_w = image.shape[-1]
        res_h = image.shape[-2]
        coord = misc.get_mgrid([image.shape[-2], image.shape[-1]]).unsqueeze(0).repeat(
            encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)

        if self.opt.isMoreINRInput:
            coord = torch.cat(
                [self.INR_encoding(coord, (res_h, res_w)), image.view(*image.shape[:2], -1).permute(0, 2, 1),
                 mask.view(*mask.shape[:2], -1).permute(0, 2, 1)], dim=-1)
        else:
            coord = self.INR_encoding(coord, (res_h, res_w))

        total = coord.clone()

        interval = 10
        all_intervals = math.ceil(res_h / interval)
        divisible = True
        if res_h / interval != res_h // interval:
            divisible = False

        for n in range(self.max_hidden_mlp_num):
            accum_mlp_output = []
            for line in range(all_intervals):
                if not divisible and line == all_intervals - 1:
                    coord = total[:, line * interval * res_w:, :]
                else:
                    coord = total[:, line * interval * res_w: (line + 1) * interval * res_w, :]
                if n == 0:
                    accum_mlp_output.append(self.mlp_process(coord,
                                                  self.INR_encoding.out_dim + (4 if self.opt.isMoreINRInput else 0),
                                                  self.opt, content_mlp=self.content_mlp_blocks[
                            f"block{self.max_hidden_mlp_num - 1 - n}"](
                            encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n) if line == all_intervals - 1 else encoder_outputs[self.max_hidden_mlp_num - 1 - n]),
                                                  resolution=(interval, res_w) if divisible or line != all_intervals - 1 else (res_h - interval * (all_intervals - 1), res_w))[1])

                else:
                    accum_mlp_output.append(self.mlp_process(coord, self.opt.INR_MLP_dim + self.INR_encoding.out_dim + (
                        4 if self.opt.isMoreINRInput else 0), self.opt, base_feat=mlp_output[0][:, line * interval * res_w: (line + 1) * interval * res_w, :]
                                    if divisible or line != all_intervals - 1 else mlp_output[0][:, line * interval * res_w:, :], content_mlp=self.content_mlp_blocks[
                                  f"block{self.max_hidden_mlp_num - 1 - n}"](
                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n) if line == all_intervals - 1 else encoder_outputs[self.max_hidden_mlp_num - 1 - n]),
                                                  resolution=(interval, res_w) if divisible or line != all_intervals - 1 else (res_h - interval * (all_intervals - 1), res_w))[1])

            accum_mlp_output = torch.cat(accum_mlp_output, dim=1)
            mlp_output = [accum_mlp_output, accum_mlp_output]

        encoder_outputs = encoder_outputs[::-1]
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output)
            output = output + skip_output
        output = self.deconv_blocks[-1](output)

        app_mlp, app_params = self.appearance_mlps(output)
        harm_out = []

        accum_mlp_output = []
        for line in range(all_intervals):
            if not divisible and line == all_intervals - 1:
                base = mlp_output[1][:, line * interval * res_w:, :]
            else:
                base = mlp_output[1][:, line * interval * res_w: (line + 1) * interval * res_w, :]

            accum_mlp_output.append(self.mlp_process(None, self.opt.INR_MLP_dim, self.opt, base_feat=base,
                                      appearance_mlp=app_mlp,
                resolution=(interval, res_w) if divisible or line != all_intervals - 1 else (res_h - interval * (all_intervals - 1), res_w))[0])

        accum_mlp_output = torch.cat(accum_mlp_output, dim=2)
        harm_out.append(accum_mlp_output)

        fit_lut3d, lut_transform_image = self.lut_transform(image, app_params, None)

        return harm_out, fit_lut3d, lut_transform_image


class onlyVector(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, opt, attend_from):
        super(onlyVector, self).__init__()
        self.INR_encoding = None
        if opt.embedding_type == "PosEncodingNeRF":
            self.INR_encoding = PosEncodingNeRF(in_features=2, sidelength=opt.input_size)
        elif opt.embedding_type == "RandomFourier":
            self.INR_encoding = RandomFourier(std_scale=10, embedding_length=64, device=opt.device)
        elif opt.embedding_type == "CIPS_embed":
            self.INR_encoding = CIPS_embed(size=opt.base_size, embedding_length=32)
        elif opt.embedding_type == "INRGAN_embed":
            self.INR_encoding = INRGAN_embed(resolution=opt.INR_input_size)
        else:
            raise NotImplementedError

        encoder_blocks_channels = encoder_blocks_channels[::-1]
        max_hidden_mlp_num = attend_from + 1
        self.opt = opt
        self.max_hidden_mlp_num = max_hidden_mlp_num
        self.base_mlp_blocks = nn.ModuleDict()
        for n in range(max_hidden_mlp_num):
            if n != max_hidden_mlp_num - 1:
                self.base_mlp_blocks[f"block{n}"] = lineParams(encoder_blocks_channels.pop(),
                        [self.INR_encoding.out_dim + opt.INR_MLP_dim + (4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim],
                       (opt.INR_input_size // (2 ** (n + 1))) ** 2,
                       opt, n + 1, toRGB=False)
            else:
                self.base_mlp_blocks[f"block{n}"] = lineParams(encoder_blocks_channels.pop(),
                                                               [self.INR_encoding.out_dim + (4 if opt.isMoreINRInput else 0),
                                                                opt.INR_MLP_dim],
                                                               (opt.INR_input_size // (2 ** (n + 1))) ** 2,
                                                               opt, n + 1, toRGB=False)

        self.deconv_blocks = nn.ModuleList()

        encoder_blocks_channels = encoder_blocks_channels[::-1]
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - attend_from):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=False
            ))
            in_channels = out_channels

        self.appearance_mlps = lineParams(out_channels, [opt.INR_MLP_dim, opt.INR_MLP_dim],
                                          (opt.base_size // (2 ** (max_hidden_mlp_num - 1))) ** 2,
                                          opt, 2, toRGB=True)

        self.lut_transform = build_lut_transform(self.appearance_mlps.output_dim, opt.LUT_dim,
                                                 None, opt)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, encoder_outputs, image=None, mask=None, coord_samples=None):
        encoder_outputs = encoder_outputs[::-1]
        mlp_output = None
        waitToRGB = []
        for n in range(self.max_hidden_mlp_num):
            if not self.opt.hr_train:
                coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))) \
                    .unsqueeze(0).repeat(encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
            else:
                if self.training:
                    coord = coord_samples[self.max_hidden_mlp_num - n - 1].permute(0, 2, 3, 1).view(
                        encoder_outputs[0].shape[0], -1, 2)
                elif not self.opt.isFullRes:
                    coord = misc.get_mgrid(
                        self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
                else:
                    coord = misc.get_mgrid([image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1)),
                                            image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))]).unsqueeze(
                        0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)

            if self.opt.isMoreINRInput:
                if not self.opt.isFullRes or self.training:
                    res_h = res_w = np.sqrt(coord.shape[1]).astype(int)
                else:
                    res_h = image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1))
                    res_w = image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))

                res_image = torchvision.transforms.Resize([res_h, res_w])(image)
                res_mask = torchvision.transforms.Resize([res_h, res_w])(mask)
                coord = torch.cat([self.INR_encoding(coord), res_image.view(*res_image.shape[:2], -1).permute(0, 2, 1),
                                   res_mask.view(*res_mask.shape[:2], -1).permute(0, 2, 1)], dim=-1)
            else:
                coord = self.INR_encoding(coord)

            if n == 0:
                mlp_output = self.mlp_process(None, self.opt, base_feat=coord, appearance_mlp=self.base_mlp_blocks[f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n))[0])
                waitToRGB.append(mlp_output[1])
            else:
                mlp_output = self.mlp_process(coord, self.opt, base_feat=mlp_output[0], appearance_mlp=self.base_mlp_blocks[
                                                  f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n))[0])
                waitToRGB.append(mlp_output[1])

        encoder_outputs = encoder_outputs[::-1]
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output)
            output = output + skip_output
        output = self.deconv_blocks[-1](output)

        app_mlp, app_params = self.appearance_mlps(output)
        harm_out = []
        lastMask = None
        for id in range(len(waitToRGB)):
            output = self.mlp_process(None, self.opt, base_feat=waitToRGB[id],
                                      appearance_mlp=app_mlp, isRGB=True)
            harm_out.append(output[0])

        fit_lut3d, lut_transform_image = self.lut_transform(image, app_params, None)

        return harm_out, fit_lut3d, lut_transform_image

    def mlp_process(self, coorinates, opt, base_feat=None, appearance_mlp=None, isRGB=False):

        activation = select_activation(opt.activation)
        bs = base_feat.shape[0]
        output = base_feat
        if coorinates is not None:
            output = torch.cat([coorinates, base_feat], dim=2)

        if not isRGB:
            for id, layer in enumerate(appearance_mlp):
                output = torch.matmul(output, layer[0]) + layer[1]
                output = activation(output)
            output_large = self.up(lin2img(output))

            return output_large.view(bs, -1, opt.INR_MLP_dim), output
        else:
            for id, layer in enumerate(appearance_mlp):
                if id != len(appearance_mlp) - 1:
                    output = torch.matmul(output, layer[0]) + layer[1]
                    output = activation(output)
                else:
                    output = torch.matmul(output, layer[0]) + layer[1]  # last layer
                    if opt.activation == 'leakyrelu_pe':
                        output = torch.tanh(output)

            return lin2img(output), lin2img(output)


class onlyMatrix(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, opt, attend_from):
        super(onlyMatrix, self).__init__()
        self.INR_encoding = None
        if opt.embedding_type == "PosEncodingNeRF":
            self.INR_encoding = PosEncodingNeRF(in_features=2, sidelength=opt.input_size)
        elif opt.embedding_type == "RandomFourier":
            self.INR_encoding = RandomFourier(std_scale=10, embedding_length=64, device=opt.device)
        elif opt.embedding_type == "CIPS_embed":
            self.INR_encoding = CIPS_embed(size=opt.base_size, embedding_length=32)
        elif opt.embedding_type == "INRGAN_embed":
            self.INR_encoding = INRGAN_embed(resolution=opt.INR_input_size)
        else:
            raise NotImplementedError
        encoder_blocks_channels = encoder_blocks_channels[::-1]
        max_hidden_mlp_num = attend_from + 1
        self.opt = opt
        self.max_hidden_mlp_num = max_hidden_mlp_num
        self.content_mlp_blocks = nn.ModuleDict()
        for n in range(max_hidden_mlp_num):
            if n != max_hidden_mlp_num - 1:
                self.content_mlp_blocks[f"block{n}"] = convParams(encoder_blocks_channels.pop(),
                                            [self.INR_encoding.out_dim + opt.INR_MLP_dim + (4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim], opt, n + 1)
            else:
                self.content_mlp_blocks[f"block{n}"] = convParams(encoder_blocks_channels.pop(),
                                            [self.INR_encoding.out_dim + (4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim], opt, n + 1)

        self.deconv_blocks = nn.ModuleList()

        encoder_blocks_channels = encoder_blocks_channels[::-1]
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - attend_from):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=False
            ))
            in_channels = out_channels

        self.base_mlps = convParams(out_channels, [opt.INR_MLP_dim, opt.INR_MLP_dim],
                                          opt, 2, toRGB=True)

        self.compress_layer = nn.Sequential(
            nn.Linear(32 ** 2, 64, bias=False),
            nn.BatchNorm1d(self.base_mlps.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=True)
        )

        self.lut_transform = build_lut_transform(self.base_mlps.output_dim, opt.LUT_dim,
                                                 None, opt)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, encoder_outputs, image=None, mask=None, coord_samples=None):
        encoder_outputs = encoder_outputs[::-1]
        mlp_output = None
        waitToRGB = []
        for n in range(self.max_hidden_mlp_num):
            if not self.opt.hr_train:
                coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1)))\
                    .unsqueeze(0).repeat(encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
            else:
                if self.training:
                    coord = coord_samples[self.max_hidden_mlp_num - n - 1].permute(0, 2, 3, 1).view(encoder_outputs[0].shape[0], -1, 2)
                elif not self.opt.isFullRes:
                    coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
                else:
                    coord = misc.get_mgrid([image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1)),
                                            image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))]).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)

            if self.opt.isMoreINRInput:
                if not self.opt.isFullRes or self.training:
                    res_h = res_w = np.sqrt(coord.shape[1]).astype(int)
                else:
                    res_h = image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1))
                    res_w = image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))

                res_image = torchvision.transforms.Resize([res_h, res_w])(image)
                res_mask = torchvision.transforms.Resize([res_h, res_w])(mask)
                coord = torch.cat([self.INR_encoding(coord), res_image.view(*res_image.shape[:2], -1).permute(0, 2, 1),
                                   res_mask.view(*res_mask.shape[:2], -1).permute(0, 2, 1)], dim=-1)
            else:
                coord = self.INR_encoding(coord)

            if n == 0:
                mlp_output = self.mlp_process(coord, self.INR_encoding.out_dim + (4 if self.opt.isMoreINRInput else 0),
                                              self.opt, content_mlp=self.content_mlp_blocks[f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n)))
                waitToRGB.append(mlp_output[1])
            else:
                mlp_output = self.mlp_process(coord, self.opt.INR_MLP_dim + self.INR_encoding.out_dim + (
                    4 if self.opt.isMoreINRInput else 0), self.opt, base_feat=mlp_output[0], content_mlp=self.content_mlp_blocks[
                                                  f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n)))
                waitToRGB.append(mlp_output[1])

        encoder_outputs = encoder_outputs[::-1]
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output)
            output = output + skip_output
        output = self.deconv_blocks[-1](output)

        cont_mlp, cont_params = self.base_mlps(output, outMore=True)
        harm_out = []
        lastMask = None
        for id in range(len(waitToRGB)):
            output = self.mlp_process(None, self.opt.INR_MLP_dim, self.opt, base_feat=waitToRGB[id],
                                      content_mlp=cont_mlp, isRGB=True)
            harm_out.append(lin2img(output[1]))
        cont_params = self.compress_layer(torch.flatten(cont_params, 2)).squeeze(-1)
        fit_lut3d, lut_transform_image = self.lut_transform(image, cont_params, None)

        return harm_out, fit_lut3d, lut_transform_image

    def mlp_process(self, coorinates, INR_input_dim, opt, base_feat=None, content_mlp=None, resolution=None, isRGB=False):

        activation = select_activation(opt.activation)

        output = None

        if content_mlp is not None:
            if base_feat is not None:
                if coorinates is not None:
                    coorinates = torch.cat([coorinates, base_feat], dim=2)
                else:
                    coorinates = base_feat
            coorinates = lin2img(coorinates, resolution)

            k_h = coorinates.shape[2] // content_mlp[0][0].shape[1]
            k_w = coorinates.shape[3] // content_mlp[0][0].shape[1]
            bs = coorinates.shape[0]
            h_lr = w_lr = content_mlp[0][0].shape[1]
            nci = INR_input_dim

            coorinates = coorinates.unfold(2, k_h, k_h).unfold(3, k_w, k_w)
            coorinates = coorinates.permute(0, 2, 3, 4, 5, 1).contiguous().view(
                bs, h_lr, w_lr, int(k_h * k_w), nci)

            for id, layer in enumerate(content_mlp):
                if id == 0:
                    output = torch.matmul(coorinates, layer[0]) + layer[1]
                    output = activation(output)
                else:
                    output = torch.matmul(output, layer[0]) + layer[1]
                    output = activation(output)

            if isRGB:
                output = output.view(bs, h_lr, w_lr, k_h, k_w, 3).permute(
                    0, 1, 3, 2, 4, 5).contiguous().view(bs, -1, 3)
            else:
                output = output.view(bs, h_lr, w_lr, k_h, k_w, opt.INR_MLP_dim).permute(
                    0, 1, 3, 2, 4, 5).contiguous().view(bs, -1, opt.INR_MLP_dim)

            output_large = self.up(lin2img(output))

            return output_large.view(bs, -1, opt.INR_MLP_dim), output


class onlyLastLayer(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, opt, attend_from):
        super(onlyLastLayer, self).__init__()
        self.INR_encoding = None
        if opt.embedding_type == "PosEncodingNeRF":
            self.INR_encoding = PosEncodingNeRF(in_features=2, sidelength=opt.input_size)
        elif opt.embedding_type == "RandomFourier":
            self.INR_encoding = RandomFourier(std_scale=10, embedding_length=64, device=opt.device)
        elif opt.embedding_type == "CIPS_embed":
            self.INR_encoding = CIPS_embed(size=opt.base_size, embedding_length=32)
        elif opt.embedding_type == "INRGAN_embed":
            self.INR_encoding = INRGAN_embed(resolution=opt.INR_input_size)
        else:
            raise NotImplementedError
        encoder_blocks_channels = encoder_blocks_channels[::-1]
        max_hidden_mlp_num = attend_from + 1
        self.opt = opt
        self.max_hidden_mlp_num = max_hidden_mlp_num

        for n in range(max_hidden_mlp_num):
            encoder_blocks_channels.pop()

        self.deconv_blocks = nn.ModuleList()

        encoder_blocks_channels = encoder_blocks_channels[::-1]
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - attend_from):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            self.deconv_blocks.append(SEDeconvBlock(
                in_channels, out_channels,
                norm_layer=norm_layer,
                padding=0 if d == 0 else 1,
                with_se=False
            ))
            in_channels = out_channels

        self.content_mlp_blocks = nn.ModuleDict()
        for n in range(max_hidden_mlp_num):
            if n != max_hidden_mlp_num - 1:
                self.content_mlp_blocks[f"block{n}"] = convParams(out_channels,
                                                                  [self.INR_encoding.out_dim + opt.INR_MLP_dim + (
                                                                      4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim],
                                                                  opt, n + 1)
            else:
                self.content_mlp_blocks[f"block{n}"] = convParams(out_channels,
                                                                  [self.INR_encoding.out_dim + (
                                                                      4 if opt.isMoreINRInput else 0), opt.INR_MLP_dim],
                                                                  opt, n + 1)

        self.appearance_mlps = lineParams(out_channels, [opt.INR_MLP_dim, opt.INR_MLP_dim],
                                          (opt.base_size // (2 ** (max_hidden_mlp_num - 1))) ** 2,
                                          opt, 2, toRGB=True)

        self.lut_transform = build_lut_transform(self.appearance_mlps.output_dim, opt.LUT_dim,
                                                 None, opt)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, encoder_outputs, image=None, mask=None, coord_samples=None):
        encoder_outputs = encoder_outputs[::-1]
        mlp_output = None
        waitToRGB = []
        for n in range(self.max_hidden_mlp_num):
            encoder_outputs.pop(self.max_hidden_mlp_num - 1 - n)

        encoder_outputs = encoder_outputs[::-1]
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output)
            output = output + skip_output
        output = self.deconv_blocks[-1](output)

        for n in range(self.max_hidden_mlp_num):
            if not self.opt.hr_train:
                coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1)))\
                    .unsqueeze(0).repeat(encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
            else:
                if self.training:
                    coord = coord_samples[self.max_hidden_mlp_num - n - 1].permute(0, 2, 3, 1).view(encoder_outputs[0].shape[0], -1, 2)
                elif not self.opt.isFullRes:
                    coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)
                else:
                    coord = misc.get_mgrid([image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1)),
                                            image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))]).unsqueeze(0).repeat(
                        encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)

            if self.opt.isMoreINRInput:
                if not self.opt.isFullRes or self.training:
                    res_h = res_w = np.sqrt(coord.shape[1]).astype(int)
                else:
                    res_h = image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1))
                    res_w = image.shape[-1] // (2 ** (self.max_hidden_mlp_num - n - 1))

                res_image = torchvision.transforms.Resize([res_h, res_w])(image)
                res_mask = torchvision.transforms.Resize([res_h, res_w])(mask)
                coord = torch.cat([self.INR_encoding(coord), res_image.view(*res_image.shape[:2], -1).permute(0, 2, 1),
                                   res_mask.view(*res_mask.shape[:2], -1).permute(0, 2, 1)], dim=-1)
            else:
                coord = self.INR_encoding(coord)

            if n == 0:
                mlp_output = self.mlp_process(coord, self.INR_encoding.out_dim + (4 if self.opt.isMoreINRInput else 0),
                                              self.opt, content_mlp=self.content_mlp_blocks[f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  output))
                waitToRGB.append(mlp_output[1])
            else:
                mlp_output = self.mlp_process(coord, self.opt.INR_MLP_dim + self.INR_encoding.out_dim + (
                    4 if self.opt.isMoreINRInput else 0), self.opt, base_feat=mlp_output[0], content_mlp=self.content_mlp_blocks[
                                                  f"block{self.max_hidden_mlp_num - 1 - n}"](
                                                  output))
                waitToRGB.append(mlp_output[1])

        app_mlp, app_params = self.appearance_mlps(output)
        harm_out = []
        lastMask = None
        for id in range(len(waitToRGB)):
            output = self.mlp_process(None, self.opt.INR_MLP_dim, self.opt, base_feat=waitToRGB[id],
                                      appearance_mlp=app_mlp)
            harm_out.append(output[0])

        fit_lut3d, lut_transform_image = self.lut_transform(image, app_params, None)

        return harm_out, fit_lut3d, lut_transform_image

    def mlp_process(self, coorinates, INR_input_dim, opt, base_feat=None, content_mlp=None, appearance_mlp=None, resolution=None):

        activation = select_activation(opt.activation)

        output = None

        if content_mlp is not None:
            if base_feat is not None:
                coorinates = torch.cat([coorinates, base_feat], dim=2)
            coorinates = lin2img(coorinates, resolution)

            k_h = coorinates.shape[2] // content_mlp[0][0].shape[1]
            k_w = coorinates.shape[3] // content_mlp[0][0].shape[1]
            bs = coorinates.shape[0]
            h_lr = w_lr = content_mlp[0][0].shape[1]
            nci = INR_input_dim

            if (not self.opt.hr_train or not self.training) and not (not self.training and self.opt.isFullRes and self.opt.hr_train):
                coorinates = coorinates.unfold(2, k_h, k_h).unfold(3, k_w, k_w)
                coorinates = coorinates.permute(0, 2, 3, 4, 5, 1).contiguous().view(
                    bs, h_lr, w_lr, int(k_h * k_w), nci)

                for id, layer in enumerate(content_mlp):
                    if id == 0:
                        output = torch.matmul(coorinates, layer[0]) + layer[1]
                        output = activation(output)
                    else:
                        output = torch.matmul(output, layer[0]) + layer[1]
                        output = activation(output)

                output = output.view(bs, h_lr, w_lr, k_h, k_w, opt.INR_MLP_dim).permute(
                    0, 1, 3, 2, 4, 5).contiguous().view(bs, -1, opt.INR_MLP_dim)

                output_large = self.up(lin2img(output))

                return output_large.view(bs, -1, opt.INR_MLP_dim), output
            else:
                coorinates = coorinates.permute(0, 2, 3, 1)
                for id, layer in enumerate(content_mlp):
                    weigt_shape = layer[0].shape
                    bias_shape = layer[1].shape
                    layer[0] = layer[0].view(*layer[0].shape[:-2], -1).permute(0, 3, 1, 2).contiguous()
                    layer[1] = layer[1].view(*layer[1].shape[:-2], -1).permute(0, 3, 1, 2).contiguous()
                    layer[0] = F.grid_sample(layer[0], coorinates[..., :2].flip(-1), mode='nearest' if True
                    else 'bilinear', padding_mode='border', align_corners=False)
                    layer[1] = F.grid_sample(layer[1], coorinates[..., :2].flip(-1), mode='nearest' if True
                    else 'bilinear', padding_mode='border', align_corners=False)
                    layer[0] = layer[0].permute(0, 2, 3, 1).contiguous().view(*coorinates.shape[:-1], *weigt_shape[-2:])
                    layer[1] = layer[1].permute(0, 2, 3, 1).contiguous().view(*coorinates.shape[:-1], *bias_shape[-2:])

                    if id == 0:
                        output = torch.matmul(coorinates.unsqueeze(-2), layer[0]) + layer[1]
                        output = activation(output)
                    else:
                        output = torch.matmul(output, layer[0]) + layer[1]
                        output = activation(output)

                output = output.squeeze(-2).view(bs, -1, opt.INR_MLP_dim)

                output_large = self.up(lin2img(output, resolution))

                return output_large.view(bs, -1, opt.INR_MLP_dim), output

        elif appearance_mlp is not None:
            output = base_feat
            genMask = None
            for id, layer in enumerate(appearance_mlp):
                if id != len(appearance_mlp) - 1:
                    output = torch.matmul(output, layer[0]) + layer[1]
                    output = activation(output)
                else:
                    output = torch.matmul(output, layer[0]) + layer[1]  # last layer
                    if opt.activation == 'leakyrelu_pe':
                        output = torch.tanh(output)
            return lin2img(output, resolution), None