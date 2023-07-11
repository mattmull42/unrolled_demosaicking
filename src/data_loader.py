import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


RGB_SPECTRAL_STENCIL = np.array([480, 525, 650])


def data_loader_rgb(input_dir: str):
    """load rgb dataset from the input directory.

    Args:
        input_dir (str): data location

    Yields:
        (np.ndarray, str) : (image as an array, image name)
    """
    res = []
    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    for image_path in images:
        image = Image.open(image_path)
        res.append(np.array(image) / 255)

    return res


class RGBDataset(Dataset):
    def __init__(self, images_dir, device, transform=None):
        self.images_dir = images_dir
        self.device = device
        self.transform = transform
        self.data = data_loader_rgb(images_dir)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gt = self.data[index]
        if gt.shape == (321, 481, 3):
            gt = gt.transpose((1, 0, 2))

        if self.transform is not None:
            x = self.transform(gt)

        return torch.tensor(x, dtype=torch.float, device=self.device), torch.tensor(gt, dtype=torch.float, device=self.device)