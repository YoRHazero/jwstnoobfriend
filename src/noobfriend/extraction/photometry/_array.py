"""Array coercion helpers at the :mod:`noobase` kernel boundary.

``noobase``'s Rust-backed kernels are strict about dtype and byte order, and
JWST calibration arrays are commonly big-endian floats. Photometry normalizes
inputs here, at the module boundary, before handing them to those kernels.
"""

import numpy as np


def native_float64(data: np.ndarray) -> np.ndarray:
    """Return ``data`` as a C-contiguous, native-endian ``float64`` array.

    Parameters
    ----------
    data : numpy.ndarray
        Input numeric array.

    Returns
    -------
    numpy.ndarray
        Native-endian ``float64`` copy or view suitable for noobase kernels.
    """
    return np.ascontiguousarray(np.asarray(data), dtype=np.dtype("=f8"))


def valid_data_mask(data: np.ndarray) -> np.ndarray:
    """Mask pixels usable as a seed/region for aperture growth.

    On the growth side, strict ``0.0`` is treated the same as ``NaN``: in JWST
    calibration products an exact zero almost always marks an unexposed or
    flagged pixel rather than real background-subtracted signal, and growing an
    aperture into such pixels is never desired. Flux *measurement* does not use
    this rule (see :mod:`noobfriend.extraction.photometry._measure`); there only
    ``NaN`` marks missing data, so a genuine zero-valued science pixel still
    contributes.

    Parameters
    ----------
    data : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Boolean mask where pixels are finite and not exactly zero.
    """
    arr = np.asarray(data)
    return np.isfinite(arr) & (arr != 0.0)
