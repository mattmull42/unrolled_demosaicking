import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forward_operator.operators import cfa_operator


class U_ADMM(nn.Module):
    def __init__(self, N, cfa, spectral_stencil, nb_channels) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = dict()
        self.cfa = cfa
        self.spectral_stencil = spectral_stencil

        layers = []
        first_layer = PrimalBlock()
        third_layer = MultiplierBlock()

        for _ in range(N - 1):
            layers.append(first_layer)
            layers.append(AuxiliaryBlock(3, nb_channels))
            layers.append(third_layer)

        layers.append(first_layer)

        self.layers = nn.ModuleList(layers)

    def setup_operator(self, x):
        mask = cfa_operator(self.cfa, (x.shape[1], x.shape[2], 3), self.spectral_stencil, 'dirac').cfa_mask
        mask = torch.tensor(mask, dtype=x.dtype, device=self.device).permute(2, 0, 1)[None]

        self.data['mask'] = mask
        ones = torch.ones_like(x)
        self.data['AAT'] = direct(mask, adjoint(mask, ones)) / ones
        self.data['ATy'] = adjoint(mask, x)
        self.data['x'] = self.data['ATy'].clone()
        self.data['z'] = self.data['ATy'].clone()
        self.data['beta'] = torch.zeros_like(self.data['x'])

    def forward(self, x):
        self.setup_operator(x)
        res = []

        for i in range(len(self.layers)):
            self.data = self.layers[i](self.data)

            if i % 3 == 0:
                res.append(self.data['x'].permute(0, 2, 3, 1))

        return res


class PrimalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(0.1))

    def forward(self, data):
        res = data['ATy'] + self.rho * (data['z'] - data['beta'])

        tmp = direct(data['mask'], res)
        tmp /= (self.rho + data['AAT'])
        tmp = adjoint(data['mask'], tmp)

        res -= tmp
        res /= self.rho
        data['x'] += res

        return data


class MultiplierBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, data):
        data['beta'] = data['beta'] + self.eta * (data['x'] - data['z'])

        return data


class AuxiliaryBlock_(nn.Module):
    def __init__(self, C, nb_channels) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(C, nb_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nb_channels)
        self.relu_1 = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv2d(nb_channels, C, 3, padding=1, bias=False)
        self.relu_2 = nn.LeakyReLU(inplace=True)

    def forward(self, data):
        tmp = (data['x'] + data['beta'])
        tmp = self.relu_1(self.bn(self.conv_1(tmp)))
        data['z'] += self.relu_2(self.conv_2(tmp))

        return data


class AuxiliaryBlock(nn.Module):
    def __init__(self, C, nb_channels) -> None:
        super().__init__()

        self.inc = DoubleConv(C, nb_channels)
        self.down = Down(nb_channels, nb_channels)
        self.up = Up(nb_channels * 2, nb_channels)
        self.outc = DoubleConv(nb_channels, C)

    def forward(self, data):
        res = (data['x'] + data['beta'])
        x1 = self.inc(res)
        x2 = self.down(x1)
        res = self.up(x2, x1)
        res = self.outc(res)

        data['z'] += res

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


def direct(A, x):
    return torch.sum(A * x, dim=1)


def adjoint(A, x):
    return A * x[:, None, :, :]