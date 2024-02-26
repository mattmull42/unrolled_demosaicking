import torch
from torchvision.transforms.v2.functional import vertical_flip, horizontal_flip, rotate


def is_not_in_list(list, tensor):
    return not any([(tensor == variant).all() for variant in list if tensor.shape == variant.shape])


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
        ref = rotate(vertical_flip(pattern), 90)

    return get_mask(ref, out_shape)


def rotation(pattern, out_shape, angle):
    return get_mask(rotate(pattern, angle), out_shape)


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
