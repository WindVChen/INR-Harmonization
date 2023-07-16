import torch
from torch import nn as nn
import numpy as np
import math
import torch.nn.functional as F


class SimpleInputFusion(nn.Module):
    def __init__(self, add_ch=1, rgb_ch=3, ch=8, norm_layer=nn.BatchNorm2d):
        super(SimpleInputFusion, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=add_ch + rgb_ch, out_channels=ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            norm_layer(ch),
            nn.Conv2d(in_channels=ch, out_channels=rgb_ch, kernel_size=1),
        )

    def forward(self, image, additional_input):
        return self.fusion_conv(torch.cat((image, additional_input), dim=1))


class MaskedChannelAttention(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super(MaskedChannelAttention, self).__init__()
        self.global_max_pool = MaskedGlobalMaxPool2d()
        self.global_avg_pool = FastGlobalAvgPool2d()

        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(3 * in_channels, intermediate_channels_count),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        if mask.shape[2:] != x.shape[:2]:
            mask = nn.functional.interpolate(
                mask, size=x.size()[-2:],
                mode='bilinear', align_corners=True
            )
        pooled_x = torch.cat([
            self.global_max_pool(x, mask),
            self.global_avg_pool(x)
        ], dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]

        return channel_attention_weights * x


class MaskedGlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_max_pool = FastGlobalMaxPool2d()

    def forward(self, x, mask):
        return torch.cat((
            self.global_max_pool(x * mask),
            self.global_max_pool(x * (1.0 - mask))
        ), dim=1)


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(FastGlobalAvgPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).mean(dim=2)


class FastGlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(FastGlobalMaxPool2d, self).__init__()

    def forward(self, x):
        in_size = x.size()
        return x.view((in_size[0], in_size[1], -1)).max(dim=2)[0]


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class FeaturesConnector(nn.Module):
    def __init__(self, mode, in_channels, feature_channels, out_channels):
        super(FeaturesConnector, self).__init__()
        self.mode = mode if feature_channels else ''

        if self.mode == 'catc':
            self.reduce_conv = nn.Conv2d(in_channels + feature_channels, out_channels, kernel_size=1)
        elif self.mode == 'sum':
            self.reduce_conv = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

        self.output_channels = out_channels if self.mode != 'cat' else in_channels + feature_channels

    def forward(self, x, features):
        if self.mode == 'cat':
            return torch.cat((x, features), 1)
        if self.mode == 'catc':
            return self.reduce_conv(torch.cat((x, features), 1))
        if self.mode == 'sum':
            return self.reduce_conv(features) + x
        return x

    def extra_repr(self):
        return self.mode


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class RandomFourier(nn.Module):
    def __init__(self, std_scale, embedding_length, device):
        super().__init__()

        self.embed = torch.normal(0, 1, (2, embedding_length)) * std_scale
        self.embed = self.embed.to(device)

        self.out_dim = embedding_length * 2 + 2

    def forward(self, coords):
        coords_pos_enc = torch.cat([torch.sin(torch.matmul(2 * np.pi * coords, self.embed)),
                                    torch.cos(torch.matmul(2 * np.pi * coords, self.embed))], dim=-1)

        return torch.cat([coords, coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)], dim=-1)


class CIPS_embed(nn.Module):
    def __init__(self, size, embedding_length):
        super().__init__()
        self.fourier_embed = ConstantInput(size, embedding_length)
        self.predict_embed = Predict_embed(embedding_length)
        self.out_dim = embedding_length * 2 + 2

    def forward(self, coord, res=None):
        x = self.predict_embed(coord)
        y = self.fourier_embed(x, coord, res)

        return torch.cat([coord, x, y], dim=-1)


class Predict_embed(nn.Module):
    def __init__(self, embedding_length):
        super(Predict_embed, self).__init__()
        self.ffm = nn.Linear(2, embedding_length, bias=True)
        nn.init.uniform_(self.ffm.weight, -np.sqrt(9 / 2), np.sqrt(9 / 2))

    def forward(self, x):
        x = self.ffm(x)
        x = torch.sin(x)
        return x


class ConstantInput(nn.Module):
    def __init__(self, size, channel):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, size ** 2, channel))

    def forward(self, input, coord, resolution=None):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1)

        if coord.shape[1] != self.input.shape[1]:
            x = out.permute(0, 2, 1).contiguous().view(batch, self.input.shape[-1],
                                                       int(self.input.shape[1] ** 0.5), int(self.input.shape[1] ** 0.5))

            if resolution is None:
                grid = coord.view(coord.shape[0], int(coord.shape[1] ** 0.5), int(coord.shape[1] ** 0.5), coord.shape[-1])
            else:
                grid = coord.view(coord.shape[0], *resolution, coord.shape[-1])

            out = F.grid_sample(x, grid.flip(-1), mode='bilinear', padding_mode='border', align_corners=True)

            out = out.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.input.shape[-1])

        return out


class INRGAN_embed(nn.Module):
    def __init__(self, resolution: int, w_dim=None):
        super().__init__()

        self.resolution = resolution
        self.res_cfg = {"log_emb_size": 32,
                        "random_emb_size": 32,
                        "const_emb_size": 64,
                        "use_cosine": True}
        self.log_emb_size = self.res_cfg.get('log_emb_size', 0)
        self.random_emb_size = self.res_cfg.get('random_emb_size', 0)
        self.shared_emb_size = self.res_cfg.get('shared_emb_size', 0)
        self.predictable_emb_size = self.res_cfg.get('predictable_emb_size', 0)
        self.const_emb_size = self.res_cfg.get('const_emb_size', 0)
        self.fourier_scale = self.res_cfg.get('fourier_scale', np.sqrt(10))
        self.use_cosine = self.res_cfg.get('use_cosine', False)

        if self.log_emb_size > 0:
            self.register_buffer('log_basis', generate_logarithmic_basis(
                resolution, self.log_emb_size, use_diagonal=self.res_cfg.get('use_diagonal', False)))

        if self.random_emb_size > 0:
            self.register_buffer('random_basis', self.sample_w_matrix((2, self.random_emb_size), self.fourier_scale))

        if self.shared_emb_size > 0:
            self.shared_basis = nn.Parameter(self.sample_w_matrix((2, self.shared_emb_size), self.fourier_scale))

        if self.predictable_emb_size > 0:
            self.W_size = self.predictable_emb_size * self.cfg.coord_dim
            self.b_size = self.predictable_emb_size
            self.affine = nn.Linear(w_dim, self.W_size + self.b_size)

        if self.const_emb_size > 0:
            self.const_embs = nn.Parameter(torch.randn(1, resolution ** 2, self.const_emb_size))

        self.out_dim = self.get_total_dim() + 2

    def sample_w_matrix(self, shape, scale: float):
        return torch.randn(shape) * scale

    def get_total_dim(self) -> int:
        total_dim = 0
        if self.log_emb_size > 0:
            total_dim += self.log_basis.shape[0] * (2 if self.use_cosine else 1)
        total_dim += self.random_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.shared_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.predictable_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.const_emb_size

        return total_dim

    def forward(self, raw_coords, w=None):
        batch_size, img_size, in_channels = raw_coords.shape

        raw_embs = []

        if self.log_emb_size > 0:
            log_bases = self.log_basis.unsqueeze(0).repeat(batch_size, 1, 1).permute(0, 2, 1)
            raw_log_embs = torch.matmul(raw_coords, log_bases)
            raw_embs.append(raw_log_embs)

        if self.random_emb_size > 0:
            random_bases = self.random_basis.unsqueeze(0).repeat(batch_size, 1, 1)
            raw_random_embs = torch.matmul(raw_coords, random_bases)
            raw_embs.append(raw_random_embs)

        if self.shared_emb_size > 0:
            shared_bases = self.shared_basis.unsqueeze(0).repeat(batch_size, 1, 1)
            raw_shared_embs = torch.matmul(raw_coords, shared_bases)
            raw_embs.append(raw_shared_embs)

        if self.predictable_emb_size > 0:
            mod = self.affine(w)
            W = self.fourier_scale * mod[:, :self.W_size]
            W = W.view(batch_size, self.cfg.coord_dim, self.predictable_emb_size)
            bias = mod[:, self.W_size:].view(batch_size, 1, self.predictable_emb_size)
            raw_predictable_embs = (torch.matmul(raw_coords, W) + bias)
            raw_embs.append(raw_predictable_embs)

        if len(raw_embs) > 0:
            raw_embs = torch.cat(raw_embs, dim=-1)
            raw_embs = raw_embs.contiguous()
            out = raw_embs.sin()

            if self.use_cosine:
                out = torch.cat([out, raw_embs.cos()], dim=-1)

        if self.const_emb_size > 0:
            const_embs = self.const_embs.repeat([batch_size, 1, 1])
            const_embs = const_embs
            out = torch.cat([out, const_embs], dim=-1)

        return torch.cat([raw_coords, out], dim=-1)


def generate_logarithmic_basis(
        resolution,
        max_num_feats,
        remove_lowest_freq: bool = False,
        use_diagonal: bool = True):
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    # while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
    #     bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis


def generate_horizontal_basis(num_feats: int):
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int):
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int):
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int):
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_wavefront_basis(num_feats: int, basis_block, period_length: float):
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1)  # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1)  # [num_feats, 1]
    result = basis * powers * period_coef  # [num_feats, 2]

    return result.float()