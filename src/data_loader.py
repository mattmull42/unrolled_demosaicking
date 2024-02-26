from os import listdir, path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.forward_model.operators import cfa_operator


RGB_SPECTRAL_STENCIL = np.array([650, 525, 480])


def data_loader_rgb(input_dir, patch_size=None, stride=None):
    res = []
    images = [read_image(path.join(input_dir, image_path)) / 255 for image_path in listdir(input_dir)]

    if patch_size is None and stride is None:
        return torch.stack(images)

    for image in images:
        res.append(image[None].unfold(2, patch_size, stride)
                   .unfold(3, patch_size, stride)
                   .permute(2, 3, 0, 1, 4, 5)
                   .reshape(-1, image.shape[0], patch_size, patch_size))

    return torch.cat(res)


class RGBDataset(Dataset):
    def __init__(self, images_dir, cfas, patch_size=None, stride=None):
        self.images_dir = images_dir
        self.cfas = []
        self.data = data_loader_rgb(images_dir, patch_size, stride)

        for cfa in cfas:
            matrix = cfa_operator(cfa, (self.data[0].shape[1], self.data[0].shape[2], 3), RGB_SPECTRAL_STENCIL).mask
            self.cfas.append(torch.Tensor(matrix).permute(2, 0, 1))

        self.l_i = len(self.data)
        self.l_c = len(self.cfas)

    def __len__(self):
        return self.l_i * self.l_c

    def __getitem__(self, index):
        gt = self.data[index // self.l_c]
        cfa = self.cfas[index % self.l_c]
        x = torch.sum(cfa * gt, axis=0)

        return x, cfa, gt
