import torch
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lightning.pytorch import LightningModule, LightningDataModule
from os import sched_getaffinity

from src.layers_ADMM import U_ADMM


class UnrolledSystem(LightningModule):
    def __init__(self, lr, N, nb_channels) -> None:
        super().__init__()

        self.model = U_ADMM(N, nb_channels)
        self.lr = lr
        self.loss_mse = MeanSquaredError()
        self.loss_psnr = PeakSignalNoiseRatio(data_range=1, reduction=None, dim=(1, 2, 3))
        self.loss_ssim = StructuralSimilarityIndexMeasure(data_range=1, reduction=None)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss_mse(gt, output)

        return loss

    def validation_step(self, batch):
        x, gt = batch
        res = self.model(x)[-1]
        
        self.log('Loss/Val', self.loss_mse(gt, res), prog_bar=True)

    def test_step(self, batch):
        x, gt = batch
        res = self.model(x)[-1]

        psnr = self.loss_psnr(gt, res)
        ssim = self.loss_ssim(gt, res)

        self.log('Loss/Test_psnr', torch.mean(psnr))
        self.log('Loss/Test_psnr_std', torch.std(psnr))

        self.log('Loss/Test_ssim', torch.mean(ssim))
        self.log('Loss/Test_ssim_std', torch.std(ssim))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'Loss/Val'}]


class DataModule(LightningDataModule):
    def __init__(self, batch_size, train_dataset=None, val_dataset=None, test_dataset=None) -> None:
        super().__init__()

        num_cpu = len(sched_getaffinity(0))

        if train_dataset is not None:
            self.train_datal = DataLoader(train_dataset, batch_size, True, num_workers=num_cpu)

        if val_dataset is not None:
            self.val_datal = DataLoader(val_dataset, batch_size, num_workers=num_cpu)

        if test_dataset is not None:
            self.test_datal = DataLoader(test_dataset, batch_size, num_workers=num_cpu)

    def train_dataloader(self):
        return self.train_datal

    def val_dataloader(self):
        return self.val_datal

    def test_dataloader(self):
        return self.test_datal
    