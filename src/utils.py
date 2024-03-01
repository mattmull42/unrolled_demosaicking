import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import vertical_flip, horizontal_flip, rotate
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from os import sched_getaffinity


def format_output(output, crop=True):
    output = torch.clip(torch.cat(output, dim=1).permute(0, 1, 3, 4, 2), 0, 1).numpy(force=True)

    if crop:
        output = output[..., 2:-2, 2:-2, :]
    
    return output[0], output[1:]


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=len(sched_getaffinity(0)))


def is_not_in_list(list, tensor):
    return not any([(tensor == variant).all() for variant in list if tensor.shape == variant.shape])


def plot_psnr_stages(gt_list, x_hat_list, cfa, cfa_idx):
    if len(cfa) != len(cfa_idx):
        for i in range(len(gt_list)):
            plt.plot([psnr(gt_list[i], x_hat, data_range=1) for x_hat in x_hat_list[:, i]], 
                     label=f'{cfa[cfa_idx[i]]} v{i - cfa_idx.index(cfa_idx[i])}')

    else:
        for i in range(len(gt_list)):
            plt.plot([psnr(gt_list[i], x_hat, data_range=1) for x_hat in x_hat_list[:, i]], label=cfa[i])

    plt.title('PSNR in function of the stages')
    plt.xlabel('Stages')
    plt.ylabel('PSNR (dB)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
    plt.show()


def plot_results(gt_list, x_hat_list, cfa, cfa_idx, stage):
    nb_images = len(gt_list)
    nb_cols = int(nb_images**0.5)
    nb_rows = nb_images // nb_cols + (nb_images % nb_cols != 0)

    fig = plt.figure(1, figsize=(20, 20))

    for i in range(nb_images):
        ax = fig.add_subplot(nb_rows, nb_cols, i + 1)
        ax.imshow(x_hat_list[stage, i])
        ax.axis('off')

        if len(cfa) != len(cfa_idx):
            ax.set_title((f'CFA: {cfa[cfa_idx[i]]} v{i - cfa_idx.index(cfa_idx[i])}, '
                          f'PSNR: {psnr(gt_list[i], x_hat_list[stage, i], data_range=1):.2f}dB'))

        else:
            ax.set_title(f'CFA: {cfa[i]}, PSNR: {psnr(gt_list[i], x_hat_list[stage, i], data_range=1):.2f}dB')

    plt.show()


def plot_error_maps(gt_list, x_hat_list, cfa, cfa_idx, stage, gain):
    nb_images = len(gt_list)
    nb_cols = int(nb_images**0.5)
    nb_rows = nb_images // nb_cols + (nb_images % nb_cols != 0)

    fig = plt.figure(1, figsize=(20, 20))

    for i in range(nb_images):
        ax = fig.add_subplot(nb_rows, nb_cols, i + 1)
        ax.imshow(np.clip(np.abs(gt_list[i] - x_hat_list[stage, i]) * gain, 0, 1))
        ax.axis('off')

        if len(cfa) != len(cfa_idx):
            ax.set_title((f'CFA: {cfa[cfa_idx[i]]} v{i - cfa_idx.index(cfa_idx[i])}, '
                          f'PSNR: {psnr(gt_list[i], x_hat_list[stage, i], data_range=1):.2f}dB'))

        else:
            ax.set_title(f'CFA: {cfa[i]}, PSNR: {psnr(gt_list[i], x_hat_list[stage, i], data_range=1):.2f}dB')

    plt.show()


def set_matmul_precision():
    if torch.cuda.get_device_name() == 'NVIDIA A100-PCIE-40GB':
        torch.set_float32_matmul_precision('high')


def get_mask(pattern, out_shape):
    n = out_shape[-2] // pattern.shape[-2] + (out_shape[-2] % pattern.shape[-2] != 0)
    m = out_shape[-1] // pattern.shape[-1] + (out_shape[-1] % pattern.shape[-1] != 0)

    return torch.tile(pattern, (1, n, m))[:, :out_shape[-2], :out_shape[-1]]


def translation(pattern, out_shape, bottom, right):
    trans = get_mask(pattern, (out_shape[0], out_shape[1] + bottom, out_shape[2] + right))

    return trans[:, bottom:, right:]


def reflection(pattern, out_shape, mode):
    if mode == 'v':
        ref = vertical_flip(pattern)
    elif mode == 'h':
        ref = horizontal_flip(pattern)
    else:
        ref = rotate(vertical_flip(pattern), 90, expand=True)

    return get_mask(ref, out_shape)


def rotation(pattern, out_shape, angle):
    return get_mask(rotate(pattern, angle, expand=True), out_shape)


def get_variants(pattern, out_shape):
    trans_row = max(4, pattern.shape[-2] // 8)
    trans_col = max(4, pattern.shape[-1] // 8)
    reflect = ('v', 'h', 'd')
    rot = (90, 180, 270)

    res = []

    for i in range(trans_row):
        for j in range(trans_col):
            translated = translation(pattern, out_shape, i, j)

            if is_not_in_list(res, translated):
                    res.append(translated)

    for mode in reflect:
        reflected = reflection(pattern, out_shape, mode)

        if is_not_in_list(res, reflected):
            res.append(reflected)

    for angle in rot:
        rotated = rotation(pattern, out_shape, angle)

        if is_not_in_list(res, rotated):
            res.append(rotated)

    return res
