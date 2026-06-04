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


def resolve_limits(
    data: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    pmin: float,
    pmax: float,
) -> tuple[float, float]:
    """Resolve display limits, preferring explicit ``vmin``/``vmax`` over percentiles.

    Each bound is taken from ``vmin``/``vmax`` when given; a bound left as
    ``None`` falls back to the corresponding percentile cut. The percentile
    computation runs only when at least one side is missing, so passing both
    explicit limits skips it entirely.

    Parameters
    ----------
    data : numpy.ndarray
        Image array, used only for the percentile fallback (non-finite pixels
        ignored, see :func:`percentile_limits`).
    vmin, vmax : float or None
        Explicit lower/upper limits. ``None`` means "use the percentile".
    pmin, pmax : float
        Percentile cuts used for whichever side is ``None``.

    Returns
    -------
    low, high : float
        Display limits with ``low < high``. A degenerate ``low == high`` pair
        (possible when both explicit limits are equal) is widened by a small
        epsilon so downstream color mappers stay valid.

    Raises
    ------
    ValueError
        If the resolved ``low > high`` (inverted explicit limits), or if the
        percentile fallback is needed and ``pmin``/``pmax`` are invalid.
    """
    if vmin is None or vmax is None:
        plo, phi = percentile_limits(data, pmin, pmax)
    low = float(vmin) if vmin is not None else plo
    high = float(vmax) if vmax is not None else phi
    if low > high:
        raise ValueError(f"require vmin < vmax, got vmin={low}, vmax={high}")
    if low == high:
        eps = abs(low) * 1e-6 or 1e-6
        return low - eps, high + eps
    return low, high
