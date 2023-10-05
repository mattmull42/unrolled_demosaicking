import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.forward_operator.operators import cfa_operator


class U_PDHG(nn.Module):
    def __init__(self, N, cfa, spectral_stencil, nb_channels) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = dict()
        self.cfa = cfa
        self.spectral_stencil = spectral_stencil

        layers = []
        first_layer = PrimalBlock()
        second_layer = DualBlock(3, nb_channels)

        for _ in range(N):
            layers.append(first_layer)
            layers.append(DualBlock(3, nb_channels))

        self.layers = nn.Sequential(*layers)

    def setup_operator(self, x):
        self.data['shape'] = (x.shape[0], x.shape[1], x.shape[2], 3)

        op = cfa_operator(self.cfa, self.data['shape'][1:], self.spectral_stencil, 'dirac')
        A_scipy = op.matrix.tocoo()
        A = torch.sparse_coo_tensor(np.vstack((A_scipy.row, A_scipy.col)), A_scipy.data, A_scipy.shape, dtype=x.dtype, device=self.device).coalesce()

        self.data['A'] = A
        self.data['AT'] = self.data['A'].T
        self.data['AAT'] = torch.tensor((A_scipy @ A_scipy.T).todia().data, dtype=x.dtype, device=self.device).squeeze()

        self.data['ATy'] = (self.data['AT'] @ x.view(self.data['shape'][0], -1).T).T
        self.data['x'] = self.data['ATy'].clone()
        self.data['z'] = torch.zeros_like(self.data['x'])

    def forward(self, x):
        self.setup_operator(x)

        self.data = self.layers(self.data)

        return self.data['x'].view(self.data['shape'])


class PrimalBlock(nn.Module):
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


class DualBlock_(nn.Module):
    def __init__(self, C, nb_channels) -> None:
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.conv_1 = nn.Conv2d(C, nb_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nb_channels)
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv2d(nb_channels, C, 3, padding=1, bias=False)
        self.relu_2 = nn.LeakyReLU(inplace=True)

    def forward(self, data):
        input_shape = (data['shape'][0], data['shape'][3], data['shape'][1], data['shape'][2])
        output_shape = (data['shape'][0], -1)

        tmp = (data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])).view(input_shape)
        tmp = self.relu_1(self.bn(self.conv_1(tmp)))
        data['z'] += self.relu_2(self.conv_2(tmp)).view(output_shape)

        return data
    


class DualBlock(nn.Module):
    def __init__(self, C, nb_channels) -> None:
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.inc = DoubleConv(C, nb_channels)
        self.down = Down(nb_channels, nb_channels)
        self.up = Up(nb_channels * 2, nb_channels)
        self.outc = DoubleConv(nb_channels, C)

    def forward(self, data):
        input_shape = (data['shape'][0], data['shape'][3], data['shape'][1], data['shape'][2])
        output_shape = (data['shape'][0], -1)

        x1 = (data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])).view(input_shape)
        x1 = self.inc(x1)
        x2 = self.down(x1)
        data['z'] = self.up(x2, x1)
        data['z'] = self.outc(data['z']).view(output_shape)

        return data


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
