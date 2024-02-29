"""A file containing all the CFA patterns and related functions.
"""

import numpy as np

from .spectral_responses.get_spectral_responses import get_filter_response


def get_rgbp_bands(file_name: str) -> tuple:
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


def get_bayer_GRBG_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Bayer GRBG CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Bayer GRBG pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((2, 2, 1)), green_filter)

    pattern[0, 1] = red_filter
    pattern[1, 0] = blue_filter

    return pattern


def get_bayer_RGGB_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Bayer RGGB CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Bayer RGGB pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((2, 2, 1)), green_filter)

    pattern[0, 0] = red_filter
    pattern[1, 1] = blue_filter

    return pattern


def get_quad_bayer_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Quad-Bayer CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Quad-Bayer pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((4, 4, 1)), green_filter)

    pattern[0, 2] = red_filter
    pattern[0, 3] = red_filter
    pattern[1, 2] = red_filter
    pattern[1, 3] = red_filter

    pattern[2, 0] = blue_filter
    pattern[2, 1] = blue_filter
    pattern[3, 0] = blue_filter
    pattern[3, 1] = blue_filter

    return pattern


def get_sparse_3_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Sparse3 CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Sparse3 pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((8, 8, 1)), pan_filter)

    pattern[0, 0] = red_filter

    pattern[0, 4] = green_filter
    pattern[4, 0] = green_filter

    pattern[4, 4] = blue_filter

    return pattern


def get_kodak_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Kodak CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Kodak pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((4, 4, 1)), pan_filter)

    pattern[3, 2] = red_filter
    pattern[2, 3] = red_filter

    pattern[3, 0] = green_filter
    pattern[2, 1] = green_filter
    pattern[1, 2] = green_filter
    pattern[0, 3] = green_filter

    pattern[1, 0] = blue_filter
    pattern[0, 1] = blue_filter

    return pattern


def get_sony_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Sony CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Sony pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((4, 4, 1)), pan_filter)

    pattern[2, 3] = red_filter
    pattern[0, 1] = red_filter

    pattern[3, 0] = green_filter
    pattern[2, 1] = green_filter
    pattern[1, 2] = green_filter
    pattern[0, 3] = green_filter

    pattern[3, 2] = blue_filter
    pattern[1, 0] = blue_filter

    return pattern


def get_chakrabarti_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Chakrabarti CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Chakrabarti pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((6, 6, 1)), pan_filter)

    pattern[2, 3] = red_filter
    pattern[2, 2] = green_filter
    pattern[3, 3] = green_filter
    pattern[3, 2] = blue_filter

    return pattern


def get_honda_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Honda CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Honda pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((4, 4, 1)), pan_filter)

    pattern[1, 3] = red_filter

    pattern[1, 1] = green_filter
    pattern[3, 3] = green_filter

    pattern[3, 1] = blue_filter

    return pattern


def get_kaizu_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Kaizu CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Kaizu pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((6, 6, 1)), pan_filter)

    pattern[0, 0] = red_filter
    pattern[1, 1] = red_filter
    pattern[4, 2] = red_filter
    pattern[5, 3] = red_filter
    pattern[2, 4] = red_filter
    pattern[3, 5] = red_filter

    pattern[0, 2] = green_filter
    pattern[1, 3] = green_filter
    pattern[2, 0] = green_filter
    pattern[3, 1] = green_filter
    pattern[4, 4] = green_filter
    pattern[5, 5] = green_filter

    pattern[4, 0] = blue_filter
    pattern[5, 1] = blue_filter
    pattern[2, 2] = blue_filter
    pattern[3, 3] = blue_filter
    pattern[0, 4] = blue_filter
    pattern[1, 5] = blue_filter

    return pattern


def get_yamagami_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Yamagami CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Yamagami pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((4, 4, 1)), pan_filter)

    pattern[0, 2] = red_filter
    pattern[2, 0] = red_filter

    pattern[1, 1] = green_filter
    pattern[1, 3] = green_filter
    pattern[3, 1] = green_filter
    pattern[3, 3] = green_filter

    pattern[0, 0] = blue_filter
    pattern[2, 2] = blue_filter

    return pattern


def get_gindele_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Gindele CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Gindele pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((2, 2, 1)), green_filter)

    pattern[0, 1] = red_filter
    pattern[1, 0] = blue_filter
    pattern[1, 1] = pan_filter

    return pattern


def get_hamilton_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Hamilton CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Hamilton pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((8, 8, 1)), pan_filter)

    pattern[0, 0] = red_filter
    pattern[1, 1] = red_filter
    pattern[2, 2] = red_filter
    pattern[3, 3] = red_filter
    pattern[2, 0] = red_filter
    pattern[3, 1] = red_filter
    pattern[0, 2] = red_filter
    pattern[1, 3] = red_filter

    pattern[0, 4] = green_filter
    pattern[0, 6] = green_filter
    pattern[1, 5] = green_filter
    pattern[1, 7] = green_filter
    pattern[2, 4] = green_filter
    pattern[2, 6] = green_filter
    pattern[3, 5] = green_filter
    pattern[3, 7] = green_filter

    pattern[4, 0] = green_filter
    pattern[6, 0] = green_filter
    pattern[5, 1] = green_filter
    pattern[7, 1] = green_filter
    pattern[4, 2] = green_filter
    pattern[6, 2] = green_filter
    pattern[5, 3] = green_filter
    pattern[7, 3] = green_filter

    pattern[4, 4] = blue_filter
    pattern[5, 5] = blue_filter
    pattern[6, 6] = blue_filter
    pattern[7, 7] = blue_filter
    pattern[6, 4] = blue_filter
    pattern[7, 5] = blue_filter
    pattern[4, 6] = blue_filter
    pattern[5, 7] = blue_filter

    return pattern


def get_luo_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Luo CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Luo pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((4, 4, 1)), pan_filter)

    pattern[1, 0] = red_filter
    pattern[1, 2] = red_filter

    pattern[0, 1] = green_filter
    pattern[2, 1] = green_filter

    pattern[1, 1] = blue_filter

    return pattern


def get_wang_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Wang CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Wang pattern.
    """
    band_r, band_g, band_b, band_p = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    pattern = np.kron(np.ones((5, 5, 1)), pan_filter)

    pattern[2, 0] = red_filter
    pattern[0, 1] = red_filter
    pattern[1, 3] = red_filter
    pattern[3, 2] = red_filter
    pattern[4, 4] = red_filter

    pattern[3, 0] = green_filter
    pattern[1, 1] = green_filter
    pattern[4, 2] = green_filter
    pattern[2, 3] = green_filter
    pattern[0, 4] = green_filter

    pattern[4, 0] = blue_filter
    pattern[2, 1] = blue_filter
    pattern[0, 2] = blue_filter
    pattern[3, 3] = blue_filter
    pattern[1, 4] = blue_filter

    return pattern


def get_yamanaka_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Yamanaka CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Yamanaka pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((2, 4, 1)), green_filter)

    pattern[0, 1] = red_filter
    pattern[1, 3] = red_filter

    pattern[1, 1] = blue_filter
    pattern[0, 3] = blue_filter

    return pattern


def get_lukac_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Lukac CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Lukac pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((4, 2, 1)), green_filter)

    pattern[0, 1] = red_filter
    pattern[2, 0] = red_filter

    pattern[1, 1] = blue_filter
    pattern[3, 0] = blue_filter

    return pattern


def get_xtrans_pattern(spectral_stencil: np.ndarray, responses_file: str) -> np.ndarray:
    """Gives the Xtrans CFA pattern using the specified filters.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.

    Returns:
        np.ndarray: The Xtrans pattern.
    """
    band_r, band_g, band_b, _ = get_rgbp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    pattern = np.kron(np.ones((6, 6, 1)), green_filter)

    pattern[0, 4] = red_filter
    pattern[1, 0] = red_filter
    pattern[1, 2] = red_filter
    pattern[2, 4] = red_filter
    pattern[3, 1] = red_filter
    pattern[4, 3] = red_filter
    pattern[4, 5] = red_filter
    pattern[5, 1] = red_filter

    pattern[0, 1] = blue_filter
    pattern[1, 3] = blue_filter
    pattern[1, 5] = blue_filter
    pattern[2, 1] = blue_filter
    pattern[3, 4] = blue_filter
    pattern[4, 0] = blue_filter
    pattern[4, 2] = blue_filter
    pattern[5, 4] = blue_filter

    return pattern
