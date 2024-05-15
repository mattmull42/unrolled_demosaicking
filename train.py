import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.lightning_classes import UnrolledSystem
from src.data_loader import RGBDataset
from src.utils import get_dataloader, set_matmul_precision

set_matmul_precision()

# Declares the hyperparameters
CFAS_TRAIN = sorted(['bayer_GRBG', 'quad_bayer', 'gindele', 'chakrabarti', 'hamilton', 'binning', 'honda', 'kaizu', 'kodak', 'sparse_3', 'wang', 'yamagami', 'yamanaka', 'random'])
CFAS_TEST = sorted(['luo', 'lukac', 'xtrans', 'honda2', 'bayer_RGGB', 'sony'])
CFAS = sorted(CFAS_TRAIN + CFAS_TEST)
CFA_VARIANTS = 2
TRAIN_DIR = 'input/train'
VAL_DIR = 'input/val'
PATCH_SIZE = 64
NB_STAGES = 6
NB_CHANNELS = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
MAX_TIME = '00:47:30:00'

# Declares the datasets
train_dataset = RGBDataset(TRAIN_DIR, CFAS_TRAIN, cfa_variants=CFA_VARIANTS, patch_size=PATCH_SIZE, stride=PATCH_SIZE // 2)
train_dataloader = get_dataloader(train_dataset, BATCH_SIZE, shuffle=True)

val_dataset = RGBDataset(VAL_DIR, CFAS, cfa_variants=0)
val_dataloader = get_dataloader(val_dataset, BATCH_SIZE)

# Initializes the network and its trainer
model = UnrolledSystem(lr=LEARNING_RATE, N=NB_STAGES, nb_channels=NB_CHANNELS)

early_stop = EarlyStopping(monitor='Loss/Val', min_delta=1e-6, patience=20)
save_best = ModelCheckpoint(filename='best', monitor='Loss/Val')
logger = CSVLogger(save_dir='weights', name='-'.join(CFAS) + f'-{NB_STAGES}{"V" if CFA_VARIANTS else ""}')

trainer = pl.Trainer(logger=logger, callbacks=[early_stop, save_best], max_time=MAX_TIME)

lr_finder = pl.tuner.Tuner(trainer).lr_find(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Run the training
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
