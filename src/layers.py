import torch
import torch.nn as nn
import numpy as np

from src.forward_operator.operators import *


class U_PDGH(nn.Module):
    def __init__(self, N, cfa, spectral_stencil, nb_channels, kernel_size) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfa = cfa
        self.spectral_stencil = spectral_stencil
        self.data = dict()

        layers = []
        second_layer = dual_layer(len(spectral_stencil), nb_channels, kernel_size)

        for _ in range(N):
            layers.append(primal_layer())
            layers.append(second_layer)

        self.layers = nn.Sequential(*layers)

    def setup_operator(self, x):
        x_0 = x[:, :, :, 1:]
        x = x[:, :, :, :1].squeeze(axis=-1)

        self.data['shape'] = (x.shape[0], x.shape[1], x.shape[2], len(self.spectral_stencil))

        op = cfa_operator(self.cfa, self.data['shape'][1:], self.spectral_stencil, 'dirac')
        A_scipy = op.matrix.tocoo()
        A = torch.sparse_coo_tensor(np.vstack((A_scipy.row, A_scipy.col)), A_scipy.data, A_scipy.shape, dtype=x.dtype, device=self.device).coalesce()

        self.data['A'] = A
        self.data['AT'] = self.data['A'].T
        self.data['AAT'] = torch.tensor((A_scipy @ A_scipy.T).todia().data, dtype=x.dtype, device=self.device).squeeze()

        self.data['ATy'] = (self.data['AT'] @ x.view(x.shape[0], -1).T).T
        self.data['x'] = x_0.reshape(x.shape[0], -1)
        self.data['z'] = torch.zeros_like(self.data['x'])

    def forward(self, x):
        self.setup_operator(x)

        self.data = self.layers(self.data)

        return torch.clamp(self.data['x'].view(x.shape[0], x.shape[1], x.shape[2], -1), 0, 1)
    

class primal_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tau = nn.Parameter(torch.tensor(0.1))

    def forward(self, data):
        data['x_prev'] = data['x'].clone()
        data['x'] = data['x'] + self.tau * (data['ATy'] - data['z'])

        tmp = (data['A'] @ data['x'].T).T
        tmp /= (1 + self.tau * data['AAT'])
        tmp = self.tau * (data['AT'] @ tmp.T).T

        data['x'] -= tmp

        return data


class dual_layer(nn.Module):
    def __init__(self, C, mid_channel_nb, kernel_size) -> None:
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.conv_1 = nn.Conv2d(C, mid_channel_nb, kernel_size, padding=1)
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv2d(mid_channel_nb, C, kernel_size, padding=1)
        self.relu_2 = nn.LeakyReLU(inplace=True)

    def forward(self, data):
        input_shape = (data['shape'][0], data['shape'][3], data['shape'][1], data['shape'][2])
        output_shape = (data['shape'][0], -1)

        data['z'] = data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])
        data['z'] = self.relu_1(self.conv_1(data['z'].view(input_shape)))
        data['z'] = self.relu_2(self.conv_2(data['z'])).view(output_shape)

        return data


class dual_layer_(nn.Module):
    def __init__(self, in_channels, kernel_size) -> None:
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(0.1))

        self.conv_0 = nn.Conv2d(in_channels, 16, kernel_size, padding=1)
        self.conv_1 = conv_block(16, kernel_size)
        self.down_1 = down_block(16, 32, kernel_size)
        self.conv_2 = conv_block(32, kernel_size)
        self.down_2 = down_block(32, 64, kernel_size)
        self.conv_3 = conv_block(64, kernel_size)
        self.down_3 = down_block(64, 128, kernel_size)

        self.conv_4 = conv_block(128, kernel_size)

        self.up_1 = up_block(128, 64, kernel_size)
        self.tconv_1 = conv_block(64, kernel_size)
        self.up_2 = up_block(64, 32, kernel_size)
        self.tconv_2 = conv_block(32, kernel_size)
        self.up_3 = up_block(32, 16, kernel_size)
        self.tconv_3 = conv_block(16, kernel_size)

        self.conv_5 = nn.Conv2d(16, 3, kernel_size, padding=1)

    def forward(self, data):
        input_shape = (data['shape'][0], data['shape'][3], data['shape'][1], data['shape'][2])
        output_shape = (data['shape'][0], -1)

        data['z'] = (data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])).view(input_shape)

        z_0 = self.conv_0(data['z'])

        z_1 = self.down_1(self.conv_1(z_0))
        z_2 = self.down_2(self.conv_2(z_1))
        z_3 = self.down_3(self.conv_3(z_2))

        z_4 = self.conv_4(z_3)

        z_5 = self.tconv_1(self.up_1(z_4 + z_3))
        z_6 = self.tconv_2(self.up_2(z_5 + z_2))
        z_7 = self.tconv_3(self.up_3(z_6 + z_1))

        data['z'] = self.conv_5(z_7 + z_0).view(output_shape)

        return data


class conv_block(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        res = []
        res.append(nn.Conv2d(channels, channels, kernel_size, padding=1))
        res.append(nn.LeakyReLU(inplace=True))
        res.append(nn.Conv2d(channels, channels, kernel_size, padding=1))

        self.res = nn.Sequential(*res)

    def forward(self, x):
        return x + self.res(x)


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()

        self.res = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        return self.res(x)


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()

        self.res = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        return self.res(x)