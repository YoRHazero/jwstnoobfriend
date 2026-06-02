"""Percentile-based intensity limits for image display.

Internal helper for :mod:`noobfriend.core.display.plot`. Turns a pair of
percentile cuts (``pmin``, ``pmax``) into concrete ``(low, high)`` data limits,
ignoring non-finite pixels so that NaNs and infs in real detector frames do not
poison the stretch.
"""

import numpy as np


def percentile_limits(
    data: np.ndarray, pmin: float, pmax: float
) -> tuple[float, float]:
    """Compute ``(low, high)`` display limits from percentile cuts.

    Parameters
    ----------
    data : numpy.ndarray
        Image array. Non-finite pixels (NaN, ``+/-inf``) are ignored.
    pmin, pmax : float
        Lower and upper percentiles, in ``[0, 100]`` with ``pmin < pmax``.

    Returns
    -------
    low, high : float
        Data values at the requested percentiles. If the finite pixels are all
        equal (a flat region) the degenerate ``low == high`` pair is widened by
        a small epsilon so downstream color mappers stay valid; if no pixel is
        finite, ``(0.0, 1.0)`` is returned.

    Raises
    ------
    ValueError
        If ``pmin``/``pmax`` are out of ``[0, 100]`` or ``pmin >= pmax``.
    """
    if not (0.0 <= pmin < pmax <= 100.0):
        raise ValueError(
            f"require 0 <= pmin < pmax <= 100, got pmin={pmin}, pmax={pmax}"
        )
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(v) for v in np.percentile(finite, (pmin, pmax)))
    if low == high:
        eps = abs(low) * 1e-6 or 1e-6
        return low - eps, high + eps
    return low, high
