"""Test file with tools to the that the adjoint operator corresponds to the real adjoint of each operator.
"""

from time import perf_counter
from colorama import Fore
import numpy as np

from operators.abstract_operator import abstract_operator
from forward_operator import forward_operator
from operators import cfa_operator, binning_operator


INPUT_SHAPE = (513, 1043, 11)
SPECTRAL_STENCIL = np.linspace(440, 650, INPUT_SHAPE[2])


def run_test(operator: abstract_operator, x: np.ndarray, y: np.ndarray) -> bool:
    """Checks if the direct operation matches the adjoint one for a given operator.

    Args:
        operator (abstract_operator): The operator to test.
        x (np.ndarray): The direct input.
        y (np.ndarray): The adjoint output.

    Returns:
        bool: A boolean set to True if the test passes, set to False if it fails.
    """

    return np.abs(np.sum(y * operator.direct(x)) - np.sum(operator.adjoint(y) * x)) < 1e-9


def cfa_test(cfa: cfa_operator) -> None:
    """Runs the adjoint test on a CFA operator.

    Args:
        cfa (cfa_operator): The operator to test.
    """

    start = perf_counter()

    operator = cfa_operator(cfa, INPUT_SHAPE, SPECTRAL_STENCIL, 'dirac')

    x = np.random.rand(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    y = np.random.rand(INPUT_SHAPE[0], INPUT_SHAPE[1])

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'CFA operator adjoint test passed in {duration:.2f} seconds for the CFA {cfa}.' + Fore.WHITE)

    else:
        print(Fore.RED + f'CFA operator adjoint test failed for the CFA {cfa}.' + Fore.WHITE)


def binning_test(binning: binning_operator) -> None:
    """Runs the adjoint test on a binning operator.

    Args:
        binning (binning_operator): The operator to test.
    """

    start = perf_counter()

    operator = binning_operator(binning, INPUT_SHAPE[:2])

    x = np.random.rand(INPUT_SHAPE[0], INPUT_SHAPE[1])
    y = np.random.rand(operator.P_i, operator.P_j)

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Binning operator adjoint test passed in {duration:.2f} seconds for the CFA {binning}.' + Fore.WHITE)

    else:
        print(Fore.RED + f'Binning operator adjoint test failed for the CFA {binning}.' + Fore.WHITE)


def forward_test(cfa: cfa_operator, binning: binning_operator) -> None:
    """Runs the adjoint test on a full operator.

    Args:
        cfa (cfa_operator): The CFA operator to test.
        binning (binning_operator): The binning operator to test.
    """

    start = perf_counter()

    list_op = [cfa_operator(cfa, INPUT_SHAPE, SPECTRAL_STENCIL, 'dirac')]
    if binning:
        list_op.append(binning_operator(cfa, list_op[0].output_shape))

    operator = forward_operator(list_op)

    x = np.random.rand(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    y = np.random.rand(operator.output_shape[0], operator.output_shape[1])

    if binning:
        postfix = ' with binning.'

    else:
        postfix = ' without binning.'

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Forward operator adjoint test passed in {duration:.2f} seconds for the CFA {cfa}' + postfix + Fore.WHITE)

    else:
        print(Fore.RED + f'Forward operator adjoint test failed for the CFA {cfa}' + postfix + Fore.WHITE)


def run_adjoint_tests() -> None:
    """A wrapper to test all the possible configurations.
    """

    print(Fore.YELLOW + '######################## Beginning of the adjoint tests ########################' + Fore.WHITE)

    cfa_test('bayer_VRBV')
    forward_test('bayer_VRBV', False)

    cfa_test('quad_bayer')
    forward_test('quad_bayer', False)

    binning_test('quad_bayer')
    forward_test('quad_bayer', True)

    cfa_test('sparse_3')
    forward_test('sparse_3', False)

    print(Fore.YELLOW + '########################### End of the adjoint tests ###########################' + Fore.WHITE)