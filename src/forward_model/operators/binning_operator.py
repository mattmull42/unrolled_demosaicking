"""The file describing the binning operator.
"""

from scipy.sparse import csr_array
import numpy as np
from scipy.signal import convolve2d

from .abstract_operator import abstract_operator


class binning_operator(abstract_operator):
    def __init__(self, cfa: str, input_shape: tuple) -> None:
        """Creates an instane of the binning_operator class.

        Args:
            cfa (str): The name of the CFA to be used.
            input_shape (tuple): The shape of the object the operator takes in input.
        """
        self.cfa = cfa

        if self.cfa == 'quad_bayer':
            self.l = 2

        self.P_i = int(np.ceil(input_shape[0] / self.l))
        self.P_j = int(np.ceil(input_shape[1] / self.l))

        super().__init__(input_shape, (self.P_i, self.P_j))

    def direct(self, x: np.ndarray) -> np.ndarray:
        """A method method performing the computation of the operator.

        Args:
            x (np.ndarray): The input array. Must be of shape self.input_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.output_shape.
        """
        if self.cfa == 'quad_bayer':
            kernel = np.array([[1, 1], [1, 1]]) / 4

        return convolve2d(np.pad(x, ((0, self.l * self.P_i - x.shape[0]), (0, self.l * self.P_j - x.shape[1])), 'symmetric'), kernel, 'valid')[::self.l, ::self.l]

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """A method method performing the computation of the adjoint of the operator.

        Args:
            y (np.ndarray): The input array. Must be of shape self.output_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.input_shape.
        """
        res = np.repeat(np.repeat(y, self.l, axis=0), self.l, axis=1) / self.l**2

        tmp_i = self.input_shape[0] % self.l

        if tmp_i:
            res = res[:tmp_i - self.l]
            res[-tmp_i:] *= self.l

        tmp_j = self.input_shape[1] % self.l

        if tmp_j:
            res = res[:, :tmp_j - self.l]
            res[:, -tmp_j:] *= self.l

        return res

    @property
    def matrix(self) -> csr_array:
        """A method method giving the sparse matrix representation of the operator.

        Returns:
            csr_array: The sparse matrix representing the operator.
        """
        N_ij = self.input_shape[0] * self.input_shape[1]

        if self.cfa == 'quad_bayer':
            P_ij = self.P_i * self.P_j

            binning_i, binning_j, binning_data = [], [], []

            for i in range(P_ij):
                tmp_i, tmp_j = 2 * (i // self.P_j), 2 * (i % self.P_j)
                binning_j.append(tmp_j + self.input_shape[1] * tmp_i)

                divider = 1

                if tmp_i + 1 < self.input_shape[0]:
                    binning_j.append(tmp_j + self.input_shape[1] * (tmp_i + 1))

                    divider = 2

                    if tmp_j + 1 < self.input_shape[1]:
                        binning_j.append(tmp_j + 1 + self.input_shape[1] * tmp_i)
                        binning_j.append(tmp_j + 1 + self.input_shape[1] * (tmp_i + 1))

                        divider = 4

                elif tmp_j + 1 < self.input_shape[1]:
                    binning_j.append(tmp_j + 1 + self.input_shape[1] * tmp_i)

                    divider = 2

                binning_i += [i for _ in range(divider)]
                binning_data += [1 / divider for _ in range(divider)]

        return csr_array((binning_data, (binning_i, binning_j)), shape=(P_ij, N_ij))
