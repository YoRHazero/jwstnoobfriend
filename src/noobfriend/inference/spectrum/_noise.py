"""Correlated-noise correction factors for the 1-D error.

Resampled / drizzled data has spatially correlated noise, so the naive
quadrature error of a collapsed (or extracted) spectrum underestimates the true
1-sigma. Two estimators of a single multiplicative ``boost`` recover it:

* :func:`continuum_boost` -- *empirical*, on the collapsed 1-D spectrum: the
  scatter of the line-free (continuum) pixels divided by their naive error. It
  measures the real inflation directly, with no model and no isotropy
  assumption, but needs enough line-free pixels.
* :func:`background_boost` -- *model*, on the 2-D cutout: estimates the noise
  autocovariance from the source-free **background** (reusing
  :mod:`noobfriend.core.imgutils`) and evaluates how much summing the
  cross-dispersion window inflates the variance. Works when the cutout has
  enough sky, and measures the cross-dispersion correlation directly (no
  isotropy assumption) provided the cutout is taller than the window.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _robust_std(values: np.ndarray) -> float:
    """Median-absolute-deviation standard deviation."""
    if values.size == 0:
        return 0.0
    mad = float(np.median(np.abs(values - np.median(values))))
    return 1.4826 * mad


def continuum_boost(
    wavelength: ArrayLike,
    flux: ArrayLike,
    error: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    degree: int = 3,
    nsigma: float = 3.0,
    niter: int = 5,
    min_pixels: int = 10,
) -> float:
    """Error-correction factor from the line-free scatter of a 1-D spectrum.

    Fits a sigma-clipped polynomial continuum (the clipping rejects lines), then
    returns ``robust_std(residual) / median(error)`` over the kept (line-free)
    pixels -- how much larger the real scatter is than the quoted error.

    Parameters
    ----------
    wavelength, flux, error : array_like
        The 1-D spectrum and its (naive) 1-sigma error.
    mask : array_like of bool, optional
        ``True`` on pixels eligible as continuum (line-free). Sigma-clipping is
        still applied within it. Defaults to all finite pixels.
    degree : int, default 3
        Continuum polynomial degree.
    nsigma : float, default 3.0
        Sigma-clip threshold (pixels above it are treated as lines).
    niter : int, default 5
        Maximum clipping iterations.
    min_pixels : int, default 10
        Minimum line-free pixels required; below this returns ``1.0`` (no
        correction).

    Returns
    -------
    float
        The multiplicative boost. ``1.0`` when it cannot be estimated.

    Notes
    -----
    Best when the quoted error is simply mis-scaled (white residual). When the
    noise is *also* strongly correlated along dispersion, the polynomial
    continuum absorbs its low-frequency component and this underestimates the
    boost -- prefer :func:`background_boost` in that regime.
    """
    wl = np.asarray(wavelength, dtype=float)
    fl = np.asarray(flux, dtype=float)
    er = np.asarray(error, dtype=float)
    cand = np.isfinite(wl) & np.isfinite(fl) & np.isfinite(er) & (er > 0)
    if mask is not None:
        cand &= np.asarray(mask, dtype=bool)

    keep = cand.copy()
    resid: np.ndarray | None = None
    for _ in range(niter):
        if int(keep.sum()) < max(degree + 2, min_pixels):
            break
        coef = np.polyfit(wl[keep], fl[keep], degree)
        resid = fl - np.polyval(coef, wl)
        scatter = _robust_std(resid[keep])
        if scatter <= 0:
            break
        new_keep = cand & (np.abs(resid) <= nsigma * scatter)
        if int(new_keep.sum()) == int(keep.sum()):
            keep = new_keep
            break
        keep = new_keep

    if resid is None or int(keep.sum()) < min_pixels:
        return 1.0
    median_error = float(np.median(er[keep]))
    if median_error <= 0:
        return 1.0
    return _robust_std(resid[keep]) / median_error


def background_boost(
    data: ArrayLike,
    error: ArrayLike,
    *,
    collapse_window: tuple[int, int],
    dispersion_axis: int,
    max_lag: int = 8,
    source_mask: ArrayLike | None = None,
) -> float:
    """Error-correction factor from the 2-D background noise autocovariance.

    Estimates the stationary noise autocovariance ``C(d)`` from the source-free
    background (via :func:`noobfriend.core.imgutils.noise_autocovariance`), then
    computes how much summing the ``collapse_window`` cross-dispersion pixels
    inflates the variance over the independent-pixel value -- the boost. The
    cross-dispersion correlation is taken directly from ``C`` (no isotropy
    assumption) as long as the cutout is taller than the window.

    Parameters
    ----------
    data, error : array_like
        The 2-D cutout and its per-pixel 1-sigma error.
    collapse_window : tuple of int
        ``(lo, hi)`` half-open range on the cross-dispersion axis.
    dispersion_axis : int
        The dispersion axis (``1`` or ``0``).
    max_lag : int, default 8
        Lag-window half-width for the autocovariance; cover the noise
        correlation length.
    source_mask : array_like of bool, optional
        ``True`` where the cutout holds source flux (excluded from the
        background). Defaults to :func:`noobfriend.core.imgutils.segment`'s
        detections; the collapse window is always excluded too.

    Returns
    -------
    float
        The multiplicative boost (``>= 1`` for positively correlated noise).

    Raises
    ------
    ValueError
        Propagated from :func:`noise_autocovariance` if the cutout is too small
        or has too few background pixels for ``max_lag``.
    """
    from noobfriend.core.imgutils import (
        correlated_sum_variance,
        noise_autocovariance,
        segment,
    )

    img = np.asarray(data, dtype=float)
    err = np.asarray(error, dtype=float)
    cross_axis = 1 - dispersion_axis
    lo, hi = int(collapse_window[0]), int(collapse_window[1])

    if source_mask is not None:
        source = np.asarray(source_mask, dtype=bool)
    else:
        source = segment(np.nan_to_num(img)) > 0
    sky = ~source & np.isfinite(img) & np.isfinite(err) & (err > 0)
    window = [slice(None), slice(None)]
    window[cross_axis] = slice(lo, hi)
    sky[tuple(window)] = False  # the trace itself is not background

    autocov = noise_autocovariance(img, mask=sky, max_lag=max_lag, error=err)

    weight = np.zeros_like(img)
    mid = [slice(None), slice(None)]
    mid[cross_axis] = slice(lo, hi)
    mid[dispersion_axis] = img.shape[dispersion_axis] // 2  # one representative column
    weight[tuple(mid)] = 1.0

    correlated = correlated_sum_variance(weight, autocov)
    independent = autocov.variance * (hi - lo)
    if independent <= 0:
        return 1.0
    return float(np.sqrt(max(correlated / independent, 0.0)))
