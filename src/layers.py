from scipy.sparse import coo_array
import torch
import torch.nn as nn
import numpy as np

from src.forward_operator.operators import *
from src.forward_operator.forward_operator import forward_operator


class U_PDGH(nn.Module):
    def __init__(self, N, spectral_stencil, conv_channels, kernel_size) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.spectral_stencil = spectral_stencil
        self.data = dict()

        layers = []

        for _ in range(N):
            layers.append(primal_layer())
            layers.append(dual_layer(len(spectral_stencil), conv_channels, kernel_size))

        self.layers = nn.Sequential(*layers)


    def setup_operator(self, x):
        self.data['shape'] = (x.shape[0], x.shape[1], x.shape[2], len(self.spectral_stencil))

        op = cfa_operator('sparse_3', self.data['shape'][1:], self.spectral_stencil, 'dirac')
        forward_op = forward_operator([op])
        A_scipy = forward_op.matrix.tocoo()
        A = torch.sparse_coo_tensor(np.vstack((A_scipy.row, A_scipy.col)), A_scipy.data, A_scipy.shape, dtype=torch.float, device=self.device).coalesce()

        self.data['A'] = A
        self.data['AT'] = self.data['A'].T
        self.data['AAT'] = torch.tensor((A_scipy @ A_scipy.T).todia().data, device=self.device).squeeze()

        self.data['ATy'] = (self.data['AT'] @ x.view(x.shape[0], -1).T).T
        self.data['x'] = self.data['ATy'].clone()
        self.data['z'] = self.data['x'].clone()


    def forward(self, x):
        self.setup_operator(x)

        self.data = self.layers(self.data)

        return self.data['x'].view(x.shape[0], x.shape[1], x.shape[2], -1)
    

class primal_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tau = nn.Parameter(torch.tensor(0.1))

    
    def forward(self, data):
        data['x_prev'] = data['x'].clone()
        data['x'] = data['x'] + self.tau * (data['ATy'] - data['z'])

        tmp = (data['A'] @ data['x'].T).T
        tmp /= (torch.ones_like(data['AAT']) + self.tau * data['AAT'])
        tmp = self.tau * (data['AT'] @ tmp.T).T

        data['x'] -= tmp

        return data


class dual_layer(nn.Module):
    def __init__(self, C, conv_channels, kernel_size) -> None:
        super().__init__()

        self.in_channel_nb = C
        self.mid_channel_nb = conv_channels
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.conv_1 = nn.Conv2d(C, self.mid_channel_nb, self.kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.mid_channel_nb, C, self.kernel_size, padding='same')
    

    def forward(self, data):
        input_shape = (data['shape'][0], data['shape'][3], data['shape'][1], data['shape'][2])
        output_shape = (data['shape'][0], -1)

        data['z'] = data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])
        data['z'] = self.conv_2(self.relu(self.conv_1(data['z'].view(input_shape)))).view(output_shape)

        return data