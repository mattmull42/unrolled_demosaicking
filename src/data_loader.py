from os import listdir, path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.forward_model.cfa_operator import cfa_operator
from src.utils import get_variants


def data_loader_rgb(input_dir, patch_size=None, stride=None):
    if path.isfile(input_dir):
        res = torch.stack([read_image(input_dir) / 255])

    else:
        res = torch.stack([read_image(path.join(input_dir, image_path)) / 255
                           for image_path in listdir(input_dir)])

    if patch_size is None:
        return res

    return (res.unfold(2, patch_size, stride)
               .unfold(3, patch_size, stride)
               .permute(2, 3, 0, 1, 4, 5)
               .reshape(-1, res.shape[1], patch_size, patch_size))


class RGBDataset(Dataset):
    def __init__(self, images_dir, cfas, cfa_variants=0, patch_size=None, stride=None):
        self.cfas = []
        self.cfa_idx = []
        self.gts = data_loader_rgb(images_dir, patch_size, stride)

        for i, cfa in enumerate(cfas):
            pattern = cfa_operator(cfa, (*self.gts[0].shape[1:], self.gts[0].shape[0])).pattern
            variants = get_variants(torch.Tensor(pattern).permute(2, 0, 1), self.gts[0].shape, depth=cfa_variants)
            self.cfas += variants
            self.cfa_idx += [i] * len(variants)

        self.l_i = len(self.gts)
        self.l_c = len(self.cfas)
        self.l = self.l_i * self.l_c

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        gt = self.gts[index // self.l_c]
        cfa = self.cfas[index % self.l_c]

        return (cfa * gt).sum(axis=0), cfa, gt
