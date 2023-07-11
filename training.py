import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from time import perf_counter
import numpy as np

from src.forward_operator.operators import cfa_operator
from src.layers import U_PDGH
from src.data_loader import RGBDataset, RGB_SPECTRAL_STENCIL


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_FLAG = True
CFA = 'kodak'
TRAIN_DIR = 'src/images/train'
VAL_DIR = 'src/images/val'
NB_STAGES = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 12
NB_EPOCHS = 1

OP = cfa_operator(CFA, [481, 321], RGB_SPECTRAL_STENCIL, 'dirac').direct

train_dataset = RGBDataset(TRAIN_DIR, DEVICE, OP)
val_dataset = RGBDataset(VAL_DIR, DEVICE, OP)
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, True)


model = U_PDGH(NB_STAGES, RGB_SPECTRAL_STENCIL, 24, 7).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_batch(model, criterion, optimizer, x, gt):
    optimizer.zero_grad()
    loss = criterion(torch.clip(model(x), 0, 1), gt)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model, criterion, optimizer, loader):
    loss = 0

    for x, gt in loader:
        loss += train_batch(model, criterion, optimizer, x, gt)

    return loss / len(loader)


def train(model, criterion, optimizer, loader):
    for epoch in range(NB_EPOCHS):
        start = perf_counter()
        loss = 0

        loss += train_epoch(model, criterion, optimizer, loader)
            
        print(f'Epoch {epoch}:')
        print(f'    Time: {perf_counter() - start:.2f}s')
        print(f'    Loss: {loss / (epoch + 1):.4f}')


def check_accuracy(loader, model):
    model.eval()
    ssim = []
    psnr = []

    with torch.no_grad():
        for x, gt in loader:
            res = torch.clip(model(x), 0, 1).cpu().numpy()
            gt = gt.cpu().numpy()

            ssim.append(np.mean([structural_similarity(i, j, data_range=1, channel_axis=2) for i, j in zip(gt, res)]))
            psnr.append(peak_signal_noise_ratio(gt, res))

        print(f'SSIM: {np.mean(ssim):.4f}')
        print(f'PSNR: {np.mean(psnr):.2f}')


if TRAIN_FLAG:
    train(model, criterion, optimizer, train_loader)
    torch.save(model.state_dict(), 'weights.pt')

else:
    model.load_state_dict(torch.load('weights.pt', map_location=DEVICE))

# print('Checking accuracy on the training set:')
# check_accuracy(train_loader, model)

# print('Checking accuracy on the test set:')
# check_accuracy(val_loader, model)