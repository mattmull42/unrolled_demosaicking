"""Main class describing a full forward operator.
"""

import numpy as np
from scipy.sparse import csr_array

from operators.abstract_operator import abstract_operator


class forward_operator(abstract_operator):
    def __init__(self, operator_list: list, name: str = None) -> None:
        """Creates an instane of the forward_operator class.

        Args:
            operator_list (list): A list of operators to compose the forward model.
            name (str, optional): A simple nametag for the operator. Defaults to None. Defaults to None.
        """
        self.operator_list = operator_list
        self.name = 'forward' if name is None else name

        super().__init__(operator_list[0].input_shape, operator_list[-1].output_shape, self.name)

    def direct(self, x: np.ndarray) -> np.ndarray:
        """A method method performing the computation of the operator.

        Args:
            x (np.ndarray): The input array. Must be of shape self.input_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.output_shape.
        """
        res = x

        for operator in self.operator_list:
            res = operator.direct(res)

        return res


    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """A method performing the computation of the adjoint of the operator.

        Args:
            y (np.ndarray): The input array. Must be of shape self.output_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.input_shape.
        """
        res = y

        for operator in reversed(self.operator_list):
            res = operator.adjoint(res)

        return res

    @property
    def matrix(self) -> csr_array:
        """A method giving the sparse matrix representation of the operator.

        Returns:
            csr_array: The sparse matrix representing the operator.
        """
        mat = self.operator_list[0].matrix

        for operator in self.operator_list[1:]:
            mat = operator.matrix @ mat

        return mat

    def __str__(self) -> str:
        """Gives a simple description of the operator.

        Returns:
            str: A string describing the operator.
        """
        res = f'{self.name} operator of {type(self)} from {self.input_shape} to {self.output_shape} with the operators:'

        for operator in self.operator_list:
            res += '\n   ' + operator.__str__()

        return res
