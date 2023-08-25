import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


RGB_SPECTRAL_STENCIL = np.array([650, 525, 480])


def data_loader_rgb(input_dir: str, scale: int):
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
        image = image.resize((image.size[0] // scale, image.size[1] // scale))
        res.append(np.array(image) / 255)

    return res


class RGBDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = data_loader_rgb(images_dir, 10)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        gt = self.data[index]
        if gt.shape[0] > gt.shape[1]:
            gt = gt.transpose((1, 0, 2))

        if self.transform is not None:
            x = self.transform(gt)

        return torch.tensor(x, dtype=torch.float), torch.tensor(gt, dtype=torch.float)