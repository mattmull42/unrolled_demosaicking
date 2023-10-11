import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.utilities import grad_norm
from os import cpu_count

from src.layers_PDHG import U_PDHG
from src.layers_ADMM import U_ADMM


class UnrolledSystem(LightningModule):
    def __init__(self, lr, algorithm, N, cfa, spectral_stencil, nb_channels) -> None:
        super().__init__()

        if algorithm == 'U_PDHG':
            self.model = U_PDHG(N, cfa, spectral_stencil, nb_channels)

        elif algorithm == 'U_ADMM':
            self.model = U_ADMM(N, cfa, spectral_stencil, nb_channels)

        self.lr = lr
        self.loss = nn.functional.mse_loss
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        self.logger.experiment.add_scalar('Loss/Train', loss, self.current_epoch)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        self.log('val_loss', loss, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('Loss/Val', loss, self.current_epoch)

        for name, params in self.named_parameters():
            name_list = name.split('.')

            if name.endswith('tau'):
                self.logger.experiment.add_scalar(f'Tau/{name_list[2]}', params, self.current_epoch)

            elif name.endswith('sigma'):
                self.logger.experiment.add_scalar(f'Sigma/{name_list[2]}', params, self.current_epoch)

            else:
                self.logger.experiment.add_histogram(f'{".".join(name_list[:-1])}/{name_list[-1]}', params, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        self.log('test_loss', loss, logger=False)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)

        for key, value in norms.items():
            if 'grad_2.0_norm_total' != key:
                name_list = key.split('/')[1].split('.')

                if 'tau' in key:
                    self.logger.experiment.add_scalar(f'Tau/{name_list[-2]}_grad', value, self.current_epoch)

                elif 'sigma' in key:
                    self.logger.experiment.add_scalar(f'Sigma/{name_list[-2]}_grad', value, self.current_epoch)

                else:
                    self.logger.experiment.add_scalar(f'{".".join(name_list[:-1])}/{name_list[-1]}_grad', value, self.current_epoch)


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