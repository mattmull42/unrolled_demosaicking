import torch
from torch.utils.data import Dataset, DataLoader


class Test_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    
    def __len__(self):
        pass