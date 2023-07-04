from scipy.sparse import coo_array
import torch
import torch.nn as nn


class U_PDGH(nn.Module):
    def __init__(self, A, N) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = dict()

        self.data['A'] = A
        self.data['AT'] = self.data['A'].T

        tmp = A.cpu()
        A_scipy = coo_array((tmp.values(), tmp.indices()), tmp.shape)

        self.data['AAT'] = torch.tensor((A_scipy @ A_scipy.T).todia().data, device=self.device).squeeze()

        layers = []

        for _ in range(N):
            layers.append(primal_layer())
            layers.append(dual_layer())

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        self.data['shape'] = x.shape
        self.data['ATy'] = (self.data['AT'] @ x.view(x.shape[0], -1).T).T
        self.data['x'] = self.data['ATy'].clone()
        self.data['z'] = self.data['x'].clone()

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
    def __init__(self) -> None:
        super().__init__()

        self.mid_channel_nb = 24
        self.kernel_size = 5
        self.sigma = nn.Parameter(torch.tensor(0.1))
        self.conv_1 = nn.Conv2d(3, self.mid_channel_nb, self.kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.mid_channel_nb, 3, self.kernel_size, padding='same')
    

    def forward(self, data):
        input_shape = (data['z'].shape[0], 3, data['shape'][1], data['shape'][2])
        output_shape = (data['z'].shape[0], -1)

        data['z'] = data['z'] + self.sigma * (2 * data['x'] - data['x_prev'])
        data['z'] = self.conv_2(self.relu(self.conv_1(data['z'].view(input_shape)))).view(output_shape)

        return data