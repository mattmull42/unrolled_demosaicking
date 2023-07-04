"""A file containing all the CFA masks and related functions.
"""

import numpy as np

from .spectral_responses.get_spectral_responses import get_filter_response


def get_rbgp_bands(file_name: str) -> tuple:
    """Returns the positions of the red, green, blue and panchromatic bands in a specific filter set.

    Args:
        file_name (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        tuple: Tuple indicating the position of the RGBW filters.
    """

    if file_name == 'dirac':
        return 'red', 'green', 'blue', 'pan'

    elif file_name == 'WV34bands_Spectral_Responses.npz':
        return 2, 1, 0, 4


def get_bayer_mask(input_shape: tuple, spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Bayer CFA mask using the specified filters.

    Args:
        input_shape (tuple): The shape of the input. Will also be the shape of the mask.
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Bayer mask.
    """

    band_r, band_g, band_b, _ = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), green_filter)

    cfa_mask[::2, 1::2] = red_filter
    cfa_mask[1::2, ::2] = blue_filter

    return cfa_mask


def get_quad_mask(input_shape: tuple, spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Quad-Bayer CFA mask using the specified filters.

    Args:
        input_shape (tuple): The shape of the input. Will also be the shape of the mask.
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Quad-Bayer mask.
    """

    band_r, band_g, band_b, _ = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), green_filter)

    cfa_mask[::4, 2::4] = red_filter
    cfa_mask[::4, 3::4] = red_filter
    cfa_mask[1::4, 2::4] = red_filter
    cfa_mask[1::4, 3::4] = red_filter

    cfa_mask[2::4, ::4] = blue_filter
    cfa_mask[2::4, 1::4] = blue_filter
    cfa_mask[3::4, ::4] = blue_filter
    cfa_mask[3::4, 1::4] = blue_filter

    return cfa_mask


def get_sparse_3_mask(input_shape: tuple, spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Sparse3 CFA mask using the specified filters.

    Args:
        input_shape (tuple): The shape of the input. Will also be the shape of the mask.
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Sparse3 mask.
    """

    band_r, band_g, band_b, band_p = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), pan_filter)

    cfa_mask[::8, ::8] = red_filter

    cfa_mask[::8, 4::8] = green_filter
    cfa_mask[4::8, ::8] = green_filter

    cfa_mask[4::8, 4::8] = blue_filter

    return cfa_mask


def get_kodak_mask(input_shape: tuple, spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Kodak CFA mask using the specified filters.

    Args:
        input_shape (tuple): The shape of the input. Will also be the shape of the mask.
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Kodak mask.
    """

    band_r, band_g, band_b, band_p = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), pan_filter)

    cfa_mask[3::4, 2::4] = red_filter
    cfa_mask[2::4, 3::4] = red_filter

    cfa_mask[3::4, ::4] = green_filter
    cfa_mask[2::4, 1::4] = green_filter
    cfa_mask[1::4, 2::4] = green_filter
    cfa_mask[::4, 3::4] = green_filter

    cfa_mask[1::4, ::4] = blue_filter
    cfa_mask[::4, 1::4] = blue_filter

    return cfa_mask


def get_sony_mask(input_shape: tuple, spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Sony CFA mask using the specified filters.

    Args:
        input_shape (tuple): The shape of the input. Will also be the shape of the mask.
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Sony mask.
    """

    band_r, band_g, band_b, band_p = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), pan_filter)

    cfa_mask[3::4, 2::4] = red_filter
    cfa_mask[1::4, 0::4] = red_filter

    cfa_mask[3::4, ::4] = green_filter
    cfa_mask[2::4, 1::4] = green_filter
    cfa_mask[1::4, 2::4] = green_filter
    cfa_mask[::4, 3::4] = green_filter

    cfa_mask[2::4, 3::4] = blue_filter
    cfa_mask[::4, 1::4] = blue_filter

    return cfa_mask