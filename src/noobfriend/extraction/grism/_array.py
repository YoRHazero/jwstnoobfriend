"""Array coercion helpers shared across grism extraction.

This module is internal to ``noobfriend.extraction.grism``.
"""

import numpy as np


def _native(a: np.ndarray) -> np.ndarray:
    """Coerce an array to a contiguous, native-endian float array.

    noobase's Rust kernels (``reproject_exact``, the ``convolve`` family)
    require native-endian ``float32``/``float64`` input, but JWST cal arrays
    are big-endian ``>f4``; this rebrands the byte order (preserving precision
    for float inputs) and contiguously copies.

    Parameters
    ----------
    a : numpy.ndarray
        Input array (typically a big-endian float JWST data/error plane).

    Returns
    -------
    numpy.ndarray
        Native-endian, C-contiguous float array.
    """
    arr = np.asarray(a)
    if arr.dtype.kind == "f" and arr.dtype.itemsize in (4, 8):
        dt = np.dtype(f"=f{arr.dtype.itemsize}")
    else:
        dt = np.dtype("=f8")
    return np.ascontiguousarray(arr, dtype=dt)
