import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.lightning_classes import UnrolledSystem
from src.data_loader import RGBDataset
from src.utils import get_dataloader, set_matmul_precision

set_matmul_precision()


CFAS = sorted(['bayer_GRBG', 'gindele', 'chakrabarti', 'hamilton', 'honda', 'kaizu', 'kodak', 'sparse_3', 'wang', 'yamagami', 'yamanaka'])
CFA_VARIANTS = 1
TRAIN_DIR = 'images/train'
VAL_DIR = 'images/val'
NOISE_STD = 0
PATCH_SIZE = 64
NB_STAGES = 4
NB_CHANNELS = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
NB_EPOCHS = 200


train_dataset = RGBDataset(TRAIN_DIR, CFAS, cfa_variants=CFA_VARIANTS, patch_size=PATCH_SIZE, stride=PATCH_SIZE // 2, std=NOISE_STD)
train_dataloader = get_dataloader(train_dataset, BATCH_SIZE, shuffle=True)

val_dataset = RGBDataset(VAL_DIR, CFAS, cfa_variants=CFA_VARIANTS, std=NOISE_STD)
val_dataloader = get_dataloader(val_dataset, BATCH_SIZE)

model = UnrolledSystem(lr=LEARNING_RATE, N=NB_STAGES, nb_channels=NB_CHANNELS)

early_stop = EarlyStopping(monitor='Loss/Val', min_delta=1e-6, patience=20)
save_best = ModelCheckpoint(filename='best', monitor='Loss/Val')
logger = CSVLogger(save_dir='logs', name='-'.join(CFAS) + f'-{NB_STAGES}{"V" if CFA_VARIANTS else ""}')

trainer = pl.Trainer(logger=logger, callbacks=[early_stop, save_best], max_epochs=NB_EPOCHS)

lr_finder = pl.tuner.Tuner(trainer).lr_find(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, num_training=200)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
