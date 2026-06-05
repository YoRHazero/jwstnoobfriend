"""Array coercion helpers for photometry.

``noobase``'s Rust-backed kernels are strict about dtype and byte order. JWST
calibration arrays are commonly big-endian floats, so photometry normalizes
inputs at the module boundary before calling those kernels.
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


def native_int32(data: np.ndarray) -> np.ndarray:
    """Return ``data`` as a C-contiguous, native-endian ``int32`` array.

    Parameters
    ----------
    data : numpy.ndarray
        Input integer-like array.

    Returns
    -------
    numpy.ndarray
        Native-endian ``int32`` array.
    """
    return np.ascontiguousarray(np.asarray(data), dtype=np.dtype("=i4"))


def valid_data_mask(data: np.ndarray) -> np.ndarray:
    """Mask pixels considered usable for aperture growth and photometry.

    In noobfriend photometry, strict ``0.0`` is treated the same as ``NaN``:
    it marks missing data rather than real background-subtracted signal.

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


def measurement_image(data: np.ndarray) -> np.ndarray:
    """Coerce an image for measurement, replacing invalid pixels with ``NaN``.

    Parameters
    ----------
    data : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Native ``float64`` image where ``NaN`` and strict ``0.0`` are masked.
    """
    arr = native_float64(data)
    return np.where(valid_data_mask(arr), arr, np.nan)
