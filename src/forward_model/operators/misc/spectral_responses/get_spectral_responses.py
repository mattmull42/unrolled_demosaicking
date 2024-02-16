"""File containing the tools to retreive the filters from the files.
"""

import numpy as np
import scipy as sp
from os import path


DATA_DIR = path.join(path.dirname(__file__), 'data')


def get_filter_response(spectral_stencil: np.ndarray, responses_file: str, band: str) -> np.ndarray:
    """Retreive the filter of the correct color from the wanted file.
    If responses_file is 'dirac' an abstract dirac filter is created.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        responses_file (str): The name of the file in which the filters are. If 'dirac' then abstract dirac filters are used.
        band (str): Name of the band to filter. Must be either 'red', 'blue', 'green' or 'pan'.

    Returns:
        np.ndarray: The wanted filter.
    """
    if responses_file == 'dirac':
        return get_dirac_filter(spectral_stencil, band)

    else:
        array = np.load(path.join(DATA_DIR, responses_file))
        f = sp.interpolate.interp1d(array['spectral_stencil'], array['data'][band])

        return f(spectral_stencil)


def get_dirac_filter(spectral_stencil: np.ndarray, filter_type: str) -> np.ndarray:
    """Returns a dirac filter of the wanted color.

    Args:
        spectral_stencil (np.ndarray): Wavelength values in nanometers at which the input is sampled.
        filter_type (str): Name of the band to filter. Must be either 'red', 'blue', 'green' or 'pan'.

    Returns:
        np.ndarray: The wanted filter.
    """
    stencil = np.array(spectral_stencil)

    if filter_type == 'red':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 650).argmin()).reshape(-1)

    elif filter_type == 'green':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 525).argmin()).reshape(-1)

    elif filter_type == 'blue' or filter_type == 'cyan':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 480).argmin()).reshape(-1)

    elif filter_type == 'yellow':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 580).argmin()).reshape(-1)

    elif filter_type == 'orange':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 600).argmin()).reshape(-1)

    elif filter_type == 'pan':
        return np.full_like(spectral_stencil, 1 /len(spectral_stencil), dtype=np.float64)
