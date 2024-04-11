import torch
import torch.nn as nn
import torch.nn.functional as F


class U_ADMM(nn.Module):
    def __init__(self, N, nb_channels) -> None:
        super().__init__()

        first_layer = PrimalBlock()
        third_layer = MultiplierBlock()
        layers = [first_layer]

        for _ in range(N - 1):
            layers.append(AuxiliaryBlock(3, nb_channels))
            layers.append(third_layer)
            layers.append(first_layer)

        self.layers = nn.ModuleList(layers)
        self.data = {}

    def forward(self, y, mask):
        self.data['mask'] = mask
        self.data['y'] = y
        self.data['x'] = adjoint(mask, y)
        self.data['z'] = self.data['x'].clone()
        self.data['beta'] = torch.zeros_like(self.data['x'])

        res = []

        for i in range(len(self.layers)):
            self.data = self.layers[i](self.data)

            if i % 3 == 0:
                res.append(self.data['x'].clone())

        return torch.stack(res)


class PrimalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(0.1))
        self.kernel = nn.Parameter(torch.rand((3, 1, 3, 3)))
        self.P = lambda x: F.conv2d(x[:, None], self.kernel, padding=1)
        self.PT = lambda x: F.conv_transpose2d(x, self.kernel, padding=1)[:, 0]

    def forward(self, data):
        A = lambda x: self.rho * x + adjoint(data['mask'], self.PT(self.P(direct(data['mask'], x))))
        b = adjoint(data['mask'], self.PT(self.P(data['y']))) + self.rho * (data['z'] - data['beta'])

        data['x'] = cg(A, b)

        return data


class MultiplierBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, data):
        data['beta'] = data['beta'] + self.eta * (data['x'] - data['z'])

        return data


class AuxiliaryBlock(nn.Module):
    def __init__(self, C, nb_channels) -> None:
        super().__init__()

        self.inc = DoubleConv(C, nb_channels)
        self.down1 = Down(nb_channels, nb_channels * 2)
        self.down2 = Down(nb_channels * 2, nb_channels * 4)
        self.down3 = Down(nb_channels * 4, nb_channels * 4)
        self.up1 = Up(nb_channels * 8, nb_channels * 2)
        self.up2 = Up(nb_channels * 4, nb_channels)
        self.up3 = Up(nb_channels * 2, nb_channels)
        self.outc = DoubleConv(nb_channels, C)

    def forward(self, data):
        res = (data['x'] + data['beta'])
        x1 = self.inc(res)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        res = self.up1(x4, x3)
        res = self.up2(res, x2)
        res = self.up3(res, x1)
        res = self.outc(res)

        data['z'] += res

        return data


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


def direct(mask, x):
    return torch.sum(mask * x, dim=1)


def adjoint(mask, x):
    return mask * x[:, None, :, :]


def cg(A, b, nb_iter=100, tol=1e-6):
    x = b
    r = b - A(x)
    p = r.clone()
    crit = torch.sum(r * r)
    i = 0

    while i <= nb_iter and crit >= tol:
        Ap = A(p)
        alpha = crit / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        crit_old = crit
        crit = torch.sum(r * r)
        beta = crit / crit_old
        p = r + beta * p
        i += 1

    return x
