# %%
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.lightning_classes import UnrolledSystem, DataModule
from src.data_loader import RGBDataset

# %%
if torch.cuda.get_device_name() == 'NVIDIA A100-PCIE-40GB':
    torch.set_float32_matmul_precision('high')

# %%
CFAS = sorted(['bayer_GRBG', 'quad_bayer', 'sony', 'kodak', 'sparse_3', 'chakrabarti', 'honda', 'kaizu', 'yamagami'])
TRAIN_DIR = 'images/train'
VAL_DIR = 'images/val'
PATCH_SIZE = 64
NB_STAGES = 4
NB_CHANNELS = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
NB_EPOCHS = 200

# %%
train_dataset = RGBDataset(TRAIN_DIR, CFAS, PATCH_SIZE, PATCH_SIZE // 2)
val_dataset = RGBDataset(VAL_DIR, CFAS)
data_module = DataModule(BATCH_SIZE, train_dataset, val_dataset)

model = UnrolledSystem(lr=LEARNING_RATE, N=NB_STAGES, nb_channels=NB_CHANNELS)

early_stop = EarlyStopping(monitor='Loss/Val', min_delta=1e-6, patience=20)
save_best = ModelCheckpoint(filename='best', monitor='Loss/Val')
logger = CSVLogger(save_dir='logs', name='-'.join(CFAS) + f'-{NB_STAGES}')

trainer = pl.Trainer(logger=logger, callbacks=[early_stop, save_best], max_epochs=NB_EPOCHS)

lr_finder = pl.tuner.Tuner(trainer).lr_find(model, datamodule=data_module, num_training=200)

# %%
trainer.fit(model, datamodule=data_module)
