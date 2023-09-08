import numpy as np


def extract_padded(M, size, i, j):
    N_i, N_j = M.shape
    res = np.zeros((size, size))
    middle_size = int((size - 1) / 2)

    for ii in range(- middle_size, middle_size + 1):
        for jj in range(- middle_size, middle_size + 1):
            if i + ii >= 0 and i + ii < N_i and j + jj >= 0 and j + jj < N_j:
                res[middle_size + ii, middle_size + jj] = M[i + ii, j + jj]

    return res


def varying_kernel_convolution(M, K_list):
    N_i, N_j = M.shape
    res = np.zeros_like(M)

    for i in range(N_i):
        for j in range(N_j):
            res[i, j] = np.sum(extract_padded(M, K_list[4 * (i % 4) + j % 4].shape[0], i, j) * K_list[4 * (i % 4) + j % 4])

    np.clip(res, 0, 1, res)

    return res


K_identity = np.zeros((5, 5))
K_identity[2, 2] = 1

K_red_0 = np.zeros((5, 5))
K_red_0[2, :] = np.array([-3, 13, 0, 0, 2]) / 12

K_red_1 = np.zeros((5, 5))
K_red_1[2, :] = np.array([2, 0, 0, 13, -3]) / 12

K_red_8 = np.zeros((5, 5))
K_red_8[:2, :2] = np.array([[-1, -1], [-1, 9]]) / 6

K_red_9 = np.zeros((5, 5))
K_red_9[:2, 3:] = np.array([[-1, -1], [9, -1]]) / 6

K_red_10 = np.zeros((5, 5))
K_red_10[:, 2] = np.array([-3, 13, 0, 0, 2]) / 12

K_red_12 = np.zeros((5, 5))
K_red_12[3:, :2] = np.array([[-1, 9], [-1, -1]]) / 6

K_red_13 = np.zeros((5, 5))
K_red_13[3:, 3:] = np.array([[9, -1], [-1, -1]]) / 6

K_red_14 = np.zeros((5, 5))
K_red_14[:, 2] = np.array([2, 0, 0, 13, -3]) / 12

K_list_red = [K_red_0, K_red_1, K_identity, K_identity, K_red_0, K_red_1, K_identity, K_identity, K_red_8, K_red_9, K_red_10, K_red_10, K_red_12, K_red_13, K_red_14, K_red_14]


K_green_2 = np.zeros((5, 5))
K_green_2[2, :] = [-3, 13, 0, 0, 2]
K_green_2[:, 2] = [-3, 13, 0, 0, 2]
K_green_2 = K_green_2 / 24

K_green_3 = np.zeros((5, 5))
K_green_3[2, :] = [2, 0, 0, 13, -3]
K_green_3[:, 2] = [-3, 13, 0, 0, 2]
K_green_3 = K_green_3 / 24

K_green_6 = np.zeros((5, 5))
K_green_6[2, :] = [-3, 13, 0, 0, 2]
K_green_6[:, 2] = [2, 0, 0, 13, -3]
K_green_6 = K_green_6 / 24

K_green_7 = np.zeros((5, 5))
K_green_7[2, :] = [2, 0, 0, 13, -3]
K_green_7[:, 2] = [2, 0, 0, 13, -3]
K_green_7 = K_green_7 / 24

K_list_green = [K_identity, K_identity, K_green_2, K_green_3, K_identity, K_identity, K_green_6, K_green_7, K_green_2, K_green_3, K_identity, K_identity, K_green_6, K_green_7, K_identity, K_identity]


K_list_blue = [K_red_10, K_red_10, K_red_8, K_red_9, K_red_14, K_red_14, K_red_12, K_red_13, K_identity, K_identity, K_red_0, K_red_1, K_identity, K_identity, K_red_0, K_red_1]



ker_bayer_red_blue = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4

ker_bayer_green = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4