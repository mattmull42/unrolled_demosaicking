import torch
import torchinfo
import numpy as np
import matplotlib.pyplot as plt

from src.forward_operator.operators import *
from src.forward_operator.forward_operator import forward_operator
from src.layers import U_PDGH


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NB_STAGES = 8
B, M, N, C = 128, 128, 128, 3

x = np.arange(M * N * C).reshape(M, N, C) / (M * N * C)

op = cfa_operator('sparse_3', x.shape, [480, 525, 650], 'dirac')
forward_op = forward_operator([op])
A = forward_op.matrix.tocoo()
A = torch.sparse_coo_tensor(np.vstack((A.row, A.col)), A.data, A.shape, dtype=torch.float, device=DEVICE).coalesce()

model_input = forward_op.direct(x)
model_input_torch = torch.tensor(np.repeat(model_input[None, ...], B, axis=0), dtype=torch.float, device=DEVICE)

model = U_PDGH(A, NB_STAGES).to(DEVICE)

y = model(model_input_torch)

torchinfo.summary(model=model, input_size=(B, M, N))

# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(y[0])
# axs[1].imshow(y[1])
# plt.show()