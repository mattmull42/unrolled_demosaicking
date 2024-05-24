import torch
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lightning.pytorch import LightningModule

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
        
        return sum(self.loss_mse(gt, output) for output in res)

    def validation_step(self, batch):
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask)[-1], 0, 1)
        
        self.log('Loss/Val', self.loss_mse(gt, res), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask)[-1], 0, 1)
        gt, res = gt[..., 2:-2, 2:-2], res[..., 2:-2, 2:-2]

        psnr = self.loss_psnr(gt, res)
        ssim = self.loss_ssim(gt, res)

        self.log('Loss/Test_psnr', torch.mean(psnr))
        self.log('Loss/Test_psnr_std', torch.std(psnr))

        self.log('Loss/Test_ssim', torch.mean(ssim))
        self.log('Loss/Test_ssim_std', torch.std(ssim))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, mask, gt = batch
        res = torch.clip(self.model(x, mask), 0, 1)

        return torch.concat((gt[None], res))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'Loss/Val'}]
