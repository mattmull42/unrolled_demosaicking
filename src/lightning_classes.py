import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, LightningDataModule
from os import cpu_count

from src.layers import U_PDGH


class U_PDHG_system(LightningModule):
    def __init__(self, lr, N, cfa, spectral_stencil, kernel_size) -> None:
        super().__init__()

        self.model = U_PDGH(N, cfa, spectral_stencil, kernel_size)
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)

        self.log('train_loss', nn.functional.mse_loss(gt, res), prog_bar=True)

        return nn.functional.mse_loss(gt, res)
    
    def validation_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)

        self.log('val_loss', nn.functional.mse_loss(gt, res), prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)

        self.log('test_loss', nn.functional.mse_loss(gt, res))

    def train_dataloader(self):
        return super().train_dataloader()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    

class DataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=16) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=cpu_count())