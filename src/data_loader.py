from os import listdir, path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.forward_model.cfa_operator import cfa_operator
from src.utils import get_variants


RGB_STENCIL = [650, 525, 480]


def data_loader_rgb(input_dir, patch_size=None, stride=None):
    res = []

    if input_dir.endswith('.png') or input_dir.endswith('.jpg'):
        images = [read_image(input_dir) / 255]

    else:
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
    def __init__(self, images_dir, cfas, cfa_variants=0, patch_size=None, stride=None, std=None):
        self.images_dir = images_dir
        self.cfas = []
        self.cfa_idx = []
        self.data = data_loader_rgb(images_dir, patch_size, stride)
        self.std = std

        for i, cfa in enumerate(cfas):
            pattern = cfa_operator(cfa, (self.data[0].shape[1], self.data[0].shape[2], self.data[0].shape[0]), RGB_STENCIL).pattern
            variants = get_variants(torch.Tensor(pattern).permute(2, 0, 1), self.data[0].shape, depth=cfa_variants)
            self.cfas += variants
            self.cfa_idx += [i] * len(variants)

        self.l_i = len(self.data)
        self.l_c = len(self.cfas)
        self.l = self.l_i * self.l_c

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        gt = self.data[index // self.l_c]
        cfa = self.cfas[index % self.l_c]
        x = torch.sum(cfa * gt, axis=0)

        if self.std is not None:
            x += torch.normal(0, self.std, size=x.shape)

        return x, cfa, gt
