from torch.nn.functional import mse_loss
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule, LightningDataModule
import numpy as np
import matplotlib.pyplot as plt
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

        self.logger.experiment.add_scalar('Loss/Train', loss, self.current_epoch)

        return loss

    def validation_step(self, batch):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        self.log('val_loss', loss, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('Loss/Val', loss, self.current_epoch)

    def test_step(self, batch):
        x, gt = batch
        res = self.model(x)
        loss = 0

        for output in res:
            loss += self.loss(gt, output)

        self.log('test_loss', loss, logger=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-5)

        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]

    def on_before_optimizer_step(self, optimizer):
        nb_batches = 394

        if self.global_step % (nb_batches * 2) == 0:
            for name, params in self.named_parameters():
                name_list = name.split('.')

                if name.endswith('rho'):
                    self.logger.experiment.add_scalar(f'Rho/{name_list[2]}', params, self.current_epoch)
                    self.logger.experiment.add_scalar(f'Rho/{name_list[2]}_grad', params.grad.norm(2), self.current_epoch)

                elif name.endswith('eta'):
                    self.logger.experiment.add_scalar(f'Eta/{name_list[2]}', params, self.current_epoch)
                    self.logger.experiment.add_scalar(f'Eta/{name_list[2]}_grad', params.grad.norm(2), self.current_epoch)

                else:
                    self.logger.experiment.add_histogram(f'{".".join(name_list[:-1])}/{name_list[-1]}', params, self.current_epoch)
                    self.logger.experiment.add_scalar(f'{".".join(name_list[:-1])}/{name_list[-1]}_grad', params.grad.norm(2), self.current_epoch)


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


def ratios(net, it):
    ratios = [p.norm(2).numpy(force=True) / (p.grad.norm(2).numpy(force=True) + 1e-8) for _, p in net.named_parameters()]
    ratios = np.array(ratios) + 1e-8
    total = len(ratios)

    low = np.count_nonzero(ratios < 1e-3)
    high = np.count_nonzero(ratios > 1e3)
    mid = total - low - high

    title = (f'Weight / grad ratio of the different layers at iteration {it:04d}\n'
             f'Sane gradients:  {mid} ({(mid / total):.2%})    '
             f'Exploding gradients: {low} ({(low / total):.2%})     '
             f'Vanishing gradients: {high} ({(high / total):.2%})'
    )

    plt.figure(figsize=(20, 4))
    plt.semilogy(ratios, 'x')
    plt.axhline(y=1e3, color='r', linestyle='-', label='Vanishing gradient threshold')
    plt.axhline(y=1e-3, color='b', linestyle='-', label='Exploding gradient threshold')
    plt.title(title)
    plt.xlabel('Layer number')
    plt.ylabel('Weight / grad ratio')
    plt.legend()
    plt.show()
