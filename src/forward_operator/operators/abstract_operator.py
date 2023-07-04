"""The file descibing of the abstract operator.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse.linalg as sp_lin
from scipy.sparse import csr_array


class abstract_operator(ABC):
    def __init__(self, input_shape: tuple, output_shape: tuple, name: str) -> None:
        """Creates an instane of the abstract_operator class.

        Args:
            input_shape (tuple): The shape of the object the operator takes in input.
            output_shape (tuple): The shape of the object the operator gives in output.
            name (str): A simple nametag for the operator.
        """

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.name = name


    @abstractmethod
    def direct(self, x: np.ndarray) -> np.ndarray:
        """An abstract method performing the computation of the operator.

        Args:
            x (np.ndarray): The input array. Must be of shape self.input_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.output_shape.
        """

        pass


    @abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """An abstract method performing the computation of the adjoint of the operator.

        Args:
            y (np.ndarray): The input array. Must be of shape self.output_shape.

        Returns:
            np.ndarray: The output array. Must be of shape self.input_shape.
        """

        pass


    @property
    def matrix(self) -> csr_array:
        """An abstract method giving the sparse matrix representation of the operator.

        Returns:
            csr_array: The sparse matrix representing the operator.
        """

        pass


    @property
    def norm(self) -> float:
        """An default method giving the spectral norm of the operator.

        Returns:
            float: The spectral norm of the operator.
        """

        return sp_lin.norm(self.matrix, 2)


    def __str__(self) -> str:
        """Gives a simple description of the operator.

        Returns:
            str: A string describing the operator.
        """

        return f'{self.name} operator of {type(self)} from {self.input_shape} to {self.output_shape}'