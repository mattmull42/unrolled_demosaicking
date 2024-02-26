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

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch):
        x, mask, gt = batch
        res = self.model(x, mask)
        loss = 0

        for output in res:
            loss += self.loss_mse(gt, output)

        return loss

    def validation_step(self, batch):
        x, mask, gt = batch
        res = self.model(x, mask)[-1]
        
        self.log('Loss/Val', self.loss_mse(gt, res), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, mask, gt = batch
        res = self.model(x, mask)[-1]

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

        if train_dataset is not None:
            self.train_datal = get_dataloarder(train_dataset, batch_size, True)

        if val_dataset is not None:
            self.val_datal = get_dataloarder(val_dataset, batch_size)

        if test_dataset is not None:
            self.test_datal = get_dataloarder(test_dataset, batch_size)

    def train_dataloader(self):
        return self.train_datal

    def val_dataloader(self):
        return self.val_datal

    def test_dataloader(self):
        return self.test_datal
    

def get_dataloarder(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=len(sched_getaffinity(0)))
