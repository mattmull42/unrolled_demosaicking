import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import vertical_flip, horizontal_flip, rotate
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from os import sched_getaffinity


psnr = lambda x, y: peak_signal_noise_ratio(x, y, data_range=1)


def format_output(output, crop=True):
    output = torch.clip(torch.cat(output, dim=1).permute(0, 1, 3, 4, 2), 0, 1).numpy(force=True)

    if crop:
        output = output[..., 2:-2, 2:-2, :]
    
    return output[0], output[1:]


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=len(sched_getaffinity(0)))


def plot_psnr_stages(gt_list, x_hat_list, cfa, cfa_idx):
    if len(cfa) != len(cfa_idx):
        for i in range(len(gt_list)):
            plt.plot([psnr(gt_list[i], x_hat) for x_hat in x_hat_list[:, i]], 
                     label=f'{cfa[cfa_idx[i]]} v{i - cfa_idx.index(cfa_idx[i])}')

    else:
        for i in range(len(gt_list)):
            plt.plot([psnr(gt_list[i], x_hat) for x_hat in x_hat_list[:, i]], label=cfa[i])

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
                          f'PSNR: {psnr(gt_list[i], x_hat_list[stage, i]):.2f}dB'))

        else:
            ax.set_title(f'CFA: {cfa[i]}, PSNR: {psnr(gt_list[i], x_hat_list[stage, i]):.2f}dB')

    plt.show()


def set_matmul_precision():
    if 'NVIDIA A100-PCIE-40GB' in torch.cuda.get_device_name():
        torch.set_float32_matmul_precision('high')


def get_mask(pattern, out_shape):
    n = out_shape[-2] // pattern.shape[-2] + (out_shape[-2] % pattern.shape[-2] != 0)
    m = out_shape[-1] // pattern.shape[-1] + (out_shape[-1] % pattern.shape[-1] != 0)

    return torch.tile(pattern, (1, n, m))[:, :out_shape[-2], :out_shape[-1]]


def translation(pattern, bottom, right):
    trans = get_mask(pattern, (pattern.shape[0], pattern.shape[1] + bottom, pattern.shape[2] + right))

    return trans[:, bottom:, right:]


def reflection(pattern, mode):
    if mode == 'v':
        return vertical_flip(pattern)

    return horizontal_flip(pattern)


def rotation(pattern, angle):
    return rotate(pattern, angle, expand=True)


def get_variants(pattern, out_shape, depth):
    res = [pattern]

    for _ in range(depth):
        for pattern_idx in range(len(res)):
            res += [translation(res[pattern_idx], i, j) for i in range(pattern.shape[-2]) for j in range(pattern.shape[-1])]
            res += [reflection(res[pattern_idx], mode) for mode in ['v', 'h']]
            res += [rotation(res[pattern_idx], angle) for angle in (90, 180, 270)]

        res = list(torch.unique(torch.stack(res), dim=0))

    return [get_mask(item, out_shape) for item in res]
