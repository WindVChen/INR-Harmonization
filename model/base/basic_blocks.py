import torch
from torch import nn as nn
import numpy as np


def hyper_weight_init(m, in_features_main_net, activation):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            if activation == 'sine':
                m.bias.uniform_(-np.sqrt(6 / in_features_main_net) / 30, np.sqrt(6 / in_features_main_net) / 30)
            elif activation == 'leakyrelu_pe':
                m.bias.uniform_(-np.sqrt(6 / in_features_main_net), np.sqrt(6 / in_features_main_net))
            else:
                raise NotImplementedError


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
            norm_layer=nn.BatchNorm2d, activation=nn.ELU,
            bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class MaxPoolDownSize(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super(MaxPoolDownSize, self).__init__()
        self.depth = depth
        self.reduce_conv = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.convs = nn.ModuleList([
            ConvBlock(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for conv_i in range(depth)
        ])
        self.pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        outputs = []

        output = self.reduce_conv(x)

        for conv_i, conv in enumerate(self.convs):
            output = output if conv_i == 0 else self.pool2d(output)
            outputs.append(conv(output))

        return outputs


class convParams(nn.Module):
    def __init__(self, input_dim, INR_in_out, opt, hidden_mlp_num, hidden_dim=512, toRGB=False):
        super(convParams, self).__init__()
        self.INR_in_out = INR_in_out
        self.cont_split_weight = []
        self.cont_split_bias = []
        self.hidden_mlp_num = hidden_mlp_num
        self.param_factorize_dim = opt.param_factorize_dim
        output_dim = self.cal_params_num(INR_in_out, hidden_mlp_num, toRGB)
        self.output_dim = output_dim
        self.toRGB = toRGB
        self.cont_extraction_net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.cont_extraction_net[-1].apply(lambda m: hyper_weight_init(m, INR_in_out[0], opt.activation))

        self.basic_params = nn.ParameterList()
        if opt.param_factorize_dim > 0:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                self.basic_params.append(nn.Parameter(torch.randn(1, 1, 1, inp, outp)))

            if toRGB:
                self.basic_params.append(nn.Parameter(torch.randn(1, 1, 1, self.INR_in_out[1], 3)))

    def forward(self, feat, outMore=False):
        cont_params = self.cont_extraction_net(feat)
        out_mlp = self.to_mlp(cont_params)
        if outMore:
            return out_mlp, cont_params
        return out_mlp

    def cal_params_num(self, INR_in_out, hidden_mlp_num, toRGB=False):
        cont_params = 0
        start = 0
        if self.param_factorize_dim == -1:
            cont_params += INR_in_out[0] * INR_in_out[1] + INR_in_out[1]
            self.cont_split_weight.append([start, cont_params - INR_in_out[1]])
            self.cont_split_bias.append([cont_params - INR_in_out[1], cont_params])
            start = cont_params

            for id in range(hidden_mlp_num):
                cont_params += INR_in_out[1] * INR_in_out[1] + INR_in_out[1]
                self.cont_split_weight.append([start, cont_params - INR_in_out[1]])
                self.cont_split_bias.append([cont_params - INR_in_out[1], cont_params])
                start = cont_params

            if toRGB:
                cont_params += INR_in_out[1] * 3 + 3
                self.cont_split_weight.append([start, cont_params - 3])
                self.cont_split_bias.append([cont_params - 3, cont_params])

        elif self.param_factorize_dim > 0:
            cont_params += INR_in_out[0] * self.param_factorize_dim + self.param_factorize_dim * INR_in_out[1] + \
                           INR_in_out[1]
            self.cont_split_weight.append(
                [start, start + INR_in_out[0] * self.param_factorize_dim, cont_params - INR_in_out[1]])
            self.cont_split_bias.append([cont_params - INR_in_out[1], cont_params])
            start = cont_params

            for id in range(hidden_mlp_num):
                cont_params += INR_in_out[1] * self.param_factorize_dim + self.param_factorize_dim * INR_in_out[1] + \
                               INR_in_out[1]
                self.cont_split_weight.append(
                    [start, start + INR_in_out[1] * self.param_factorize_dim, cont_params - INR_in_out[1]])
                self.cont_split_bias.append([cont_params - INR_in_out[1], cont_params])
                start = cont_params

            if toRGB:
                cont_params += INR_in_out[1] * self.param_factorize_dim + self.param_factorize_dim * 3 + 3
                self.cont_split_weight.append(
                    [start, start + INR_in_out[1] * self.param_factorize_dim, cont_params - 3])
                self.cont_split_bias.append([cont_params - 3, cont_params])

        return cont_params

    def to_mlp(self, params):
        all_weight_bias = []
        if self.param_factorize_dim == -1:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                weight = params[:, self.cont_split_weight[id][0]:self.cont_split_weight[id][1], :, :]
                weight = weight.permute(0, 2, 3, 1).contiguous().view(weight.shape[0], *weight.shape[2:],
                                                                      inp, outp)

                bias = params[:, self.cont_split_bias[id][0]:self.cont_split_bias[id][1], :, :]
                bias = bias.permute(0, 2, 3, 1).contiguous().view(bias.shape[0], *bias.shape[2:], 1, outp)
                all_weight_bias.append([weight, bias])

            if self.toRGB:
                inp, outp = self.INR_in_out[1], 3
                weight = params[:, self.cont_split_weight[-1][0]:self.cont_split_weight[-1][1], :, :]
                weight = weight.permute(0, 2, 3, 1).contiguous().view(weight.shape[0], *weight.shape[2:],
                                                                      inp, outp)

                bias = params[:, self.cont_split_bias[-1][0]:self.cont_split_bias[-1][1], :, :]
                bias = bias.permute(0, 2, 3, 1).contiguous().view(bias.shape[0], *bias.shape[2:], 1, outp)
                all_weight_bias.append([weight, bias])

            return all_weight_bias

        else:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                weight1 = params[:, self.cont_split_weight[id][0]:self.cont_split_weight[id][1], :, :]
                weight1 = weight1.permute(0, 2, 3, 1).contiguous().view(weight1.shape[0], *weight1.shape[2:],
                                                                        inp, self.param_factorize_dim)

                weight2 = params[:, self.cont_split_weight[id][1]:self.cont_split_weight[id][2], :, :]
                weight2 = weight2.permute(0, 2, 3, 1).contiguous().view(weight2.shape[0], *weight2.shape[2:],
                                                                        self.param_factorize_dim, outp)

                bias = params[:, self.cont_split_bias[id][0]:self.cont_split_bias[id][1], :, :]
                bias = bias.permute(0, 2, 3, 1).contiguous().view(bias.shape[0], *bias.shape[2:], 1, outp)

                all_weight_bias.append([torch.tanh(torch.matmul(weight1, weight2)) * self.basic_params[id], bias])

            if self.toRGB:
                inp, outp = self.INR_in_out[1], 3
                weight1 = params[:, self.cont_split_weight[-1][0]:self.cont_split_weight[-1][1], :, :]
                weight1 = weight1.permute(0, 2, 3, 1).contiguous().view(weight1.shape[0], *weight1.shape[2:],
                                                                        inp, self.param_factorize_dim)

                weight2 = params[:, self.cont_split_weight[-1][1]:self.cont_split_weight[-1][2], :, :]
                weight2 = weight2.permute(0, 2, 3, 1).contiguous().view(weight2.shape[0], *weight2.shape[2:],
                                                                        self.param_factorize_dim, outp)

                bias = params[:, self.cont_split_bias[-1][0]:self.cont_split_bias[-1][1], :, :]
                bias = bias.permute(0, 2, 3, 1).contiguous().view(bias.shape[0], *bias.shape[2:], 1, outp)

                all_weight_bias.append([torch.tanh(torch.matmul(weight1, weight2)) * self.basic_params[-1], bias])

            return all_weight_bias


class lineParams(nn.Module):
    def __init__(self, input_dim, INR_in_out, input_resolution, opt, hidden_mlp_num, toRGB=False,
                 hidden_dim=512):
        super(lineParams, self).__init__()
        self.INR_in_out = INR_in_out
        self.app_split_weight = []
        self.app_split_bias = []
        self.toRGB = toRGB
        self.hidden_mlp_num = hidden_mlp_num
        self.param_factorize_dim = opt.param_factorize_dim
        output_dim = self.cal_params_num(INR_in_out, hidden_mlp_num)
        self.output_dim = output_dim

        self.compress_layer = nn.Sequential(
            nn.Linear(input_resolution, 64, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=True)
        )

        self.app_extraction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

        self.app_extraction_net[-1].apply(lambda m: hyper_weight_init(m, INR_in_out[0], opt.activation))

        self.basic_params = nn.ParameterList()
        if opt.param_factorize_dim > 0:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                self.basic_params.append(nn.Parameter(torch.randn(1, inp, outp)))
            if toRGB:
                self.basic_params.append(nn.Parameter(torch.randn(1, self.INR_in_out[1], 3)))

    def forward(self, feat):
        app_params = self.app_extraction_net(self.compress_layer(torch.flatten(feat, 2)).squeeze(-1))
        out_mlp = self.to_mlp(app_params)
        return out_mlp, app_params

    def cal_params_num(self, INR_in_out, hidden_mlp_num):
        app_params = 0
        start = 0
        if self.param_factorize_dim == -1:
            app_params += INR_in_out[0] * INR_in_out[1] + INR_in_out[1]
            self.app_split_weight.append([start, app_params - INR_in_out[1]])
            self.app_split_bias.append([app_params - INR_in_out[1], app_params])
            start = app_params

            for id in range(hidden_mlp_num):
                app_params += INR_in_out[1] * INR_in_out[1] + INR_in_out[1]
                self.app_split_weight.append([start, app_params - INR_in_out[1]])
                self.app_split_bias.append([app_params - INR_in_out[1], app_params])
                start = app_params

            if self.toRGB:
                app_params += INR_in_out[1] * 3 + 3
                self.app_split_weight.append([start, app_params - 3])
                self.app_split_bias.append([app_params - 3, app_params])

        elif self.param_factorize_dim > 0:
            app_params += INR_in_out[0] * self.param_factorize_dim + self.param_factorize_dim * INR_in_out[1] + \
                          INR_in_out[1]
            self.app_split_weight.append([start, start + INR_in_out[0] * self.param_factorize_dim,
                                          app_params - INR_in_out[1]])
            self.app_split_bias.append([app_params - INR_in_out[1], app_params])
            start = app_params

            for id in range(hidden_mlp_num):
                app_params += INR_in_out[1] * self.param_factorize_dim + self.param_factorize_dim * INR_in_out[1] + \
                              INR_in_out[1]
                self.app_split_weight.append(
                    [start, start + INR_in_out[1] * self.param_factorize_dim, app_params - INR_in_out[1]])
                self.app_split_bias.append([app_params - INR_in_out[1], app_params])
                start = app_params

            if self.toRGB:
                app_params += INR_in_out[1] * self.param_factorize_dim + self.param_factorize_dim * 3 + 3
                self.app_split_weight.append([start, start + INR_in_out[1] * self.param_factorize_dim,
                                              app_params - 3])
                self.app_split_bias.append([app_params - 3, app_params])

        return app_params

    def to_mlp(self, params):
        all_weight_bias = []
        if self.param_factorize_dim == -1:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                weight = params[:, self.app_split_weight[id][0]:self.app_split_weight[id][1]]
                weight = weight.view(weight.shape[0], inp, outp)

                bias = params[:, self.app_split_bias[id][0]:self.app_split_bias[id][1]]
                bias = bias.view(bias.shape[0], 1, outp)

                all_weight_bias.append([weight, bias])

            if self.toRGB:
                id = -1
                inp, outp = self.INR_in_out[1], 3
                weight = params[:, self.app_split_weight[id][0]:self.app_split_weight[id][1]]
                weight = weight.view(weight.shape[0], inp, outp)

                bias = params[:, self.app_split_bias[id][0]:self.app_split_bias[id][1]]
                bias = bias.view(bias.shape[0], 1, outp)

                all_weight_bias.append([weight, bias])

            return all_weight_bias

        else:
            for id in range(self.hidden_mlp_num + 1):
                if id == 0:
                    inp, outp = self.INR_in_out[0], self.INR_in_out[1]
                else:
                    inp, outp = self.INR_in_out[1], self.INR_in_out[1]
                weight1 = params[:, self.app_split_weight[id][0]:self.app_split_weight[id][1]]
                weight1 = weight1.view(weight1.shape[0], inp, self.param_factorize_dim)

                weight2 = params[:, self.app_split_weight[id][1]:self.app_split_weight[id][2]]
                weight2 = weight2.view(weight2.shape[0], self.param_factorize_dim, outp)

                bias = params[:, self.app_split_bias[id][0]:self.app_split_bias[id][1]]
                bias = bias.view(bias.shape[0], 1, outp)

                all_weight_bias.append([torch.tanh(torch.matmul(weight1, weight2)) * self.basic_params[id], bias])

            if self.toRGB:
                id = -1
                inp, outp = self.INR_in_out[1], 3
                weight1 = params[:, self.app_split_weight[id][0]:self.app_split_weight[id][1]]
                weight1 = weight1.view(weight1.shape[0], inp, self.param_factorize_dim)

                weight2 = params[:, self.app_split_weight[id][1]:self.app_split_weight[id][2]]
                weight2 = weight2.view(weight2.shape[0], self.param_factorize_dim, outp)

                bias = params[:, self.app_split_bias[id][0]:self.app_split_bias[id][1]]
                bias = bias.view(bias.shape[0], 1, outp)

                all_weight_bias.append([torch.tanh(torch.matmul(weight1, weight2)) * self.basic_params[id], bias])

            return all_weight_bias
