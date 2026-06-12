"""The 2-D-to-1-D spectral collapse (flux-conserving sum over a window).

Pure geometry, mechanism-only: sum a 2-D spectrum over a cross-dispersion window
into a 1-D flux and propagate the error in quadrature. The correlated-noise
correction to that naive error lives in
:mod:`noobfriend.core.specutils._noise`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def collapse(
    data: ArrayLike,
    error: ArrayLike,
    *,
    window: tuple[int, int],
    dispersion_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum a 2-D spectrum over a cross-dispersion window into 1-D.

    Parameters
    ----------
    data, error : array_like
        2-D spectrum and its per-pixel 1-sigma error, equal shape.
    window : tuple of int
        ``(lo, hi)`` half-open index range on the cross-dispersion axis.
    dispersion_axis : int
        The dispersion (wavelength) axis (``1`` or ``0``).

    Returns
    -------
    flux : numpy.ndarray
        1-D flux: the ``NaN``-aware sum over the window rows.
    error : numpy.ndarray
        1-D *naive* 1-sigma error: the quadrature sum over the window rows
        (independent-pixel; inflate it with
        :mod:`noobfriend.core.specutils._noise` for correlated noise).
    """
    flux2d = np.asarray(data, dtype=float)
    error2d = np.asarray(error, dtype=float)
    cross_axis = 1 - dispersion_axis
    lo, hi = int(window[0]), int(window[1])
    sl: list[slice] = [slice(None), slice(None)]
    sl[cross_axis] = slice(lo, hi)
    flux = np.nansum(flux2d[tuple(sl)], axis=cross_axis)
    error = np.sqrt(np.nansum(error2d[tuple(sl)] ** 2, axis=cross_axis))
    return flux, error
