from torch.nn.functional import mse_loss
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, LightningDataModule
from os import cpu_count

from src.layers_ADMM import U_ADMM


class UnrolledSystem(LightningModule):
    def __init__(self, lr, N, nb_channels) -> None:
        super().__init__()

        self.model = U_ADMM(N, nb_channels)
        self.lr = lr
        self.loss = mse_loss
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        return loss

    def validation_step(self, batch):
        x, gt = batch
        res = self.model(x)[-1]
        
        self.log('Loss/Val', self.loss(gt, res))

    def test_step(self, batch):
        x, gt = batch
        res = self.model(x)[-1]
        
        self.log('Loss/Test', self.loss(gt, res))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=1e-5)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'Loss/Val'}]


class DataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size) -> None:
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
    