"""Correlated-noise correction factors for spectra.

The image sibling :mod:`noobfriend.core.imgutils` estimates a full 2-D
autocovariance for aperture sums (both axes spatial). A 2-D *spectrum* is
different: one axis is dispersion (wavelength), the other cross-dispersion
(spatial), and collapsing it sums only the cross-dispersion direction. So the
only correlation that matters is the **cross-dispersion** one, ``C(k)`` along the
spatial axis -- estimated here directly (averaged over wavelength columns), with
no need for the general 2-D machinery or an isotropy assumption.

Two estimators of the scalar error ``boost`` that recovers correlated noise:

* :func:`continuum_boost` -- *empirical*, on a collapsed 1-D spectrum: the
  line-free scatter divided by the quoted error.
* :func:`cross_dispersion_boost` -- *model*, on the 2-D spectrum: the
  cross-dispersion autocovariance of the source-free background, turned into the
  variance inflation of summing the collapse window.
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
    boost -- prefer :func:`cross_dispersion_boost` in that regime.
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


def cross_dispersion_boost(
    data: ArrayLike,
    error: ArrayLike,
    *,
    window: tuple[int, int],
    dispersion_axis: int,
    source_mask: ArrayLike | None = None,
    max_lag: int = 8,
    min_sky: int = 100,
) -> float:
    """Error-correction factor from the 2-D cross-dispersion noise correlation.

    Estimates the cross-dispersion (spatial) autocovariance ``C(k)`` from the
    source-free background -- averaging same-column pixel pairs separated by
    ``k`` rows over all wavelength columns -- then returns the inflation of
    summing the ``window`` rows over the independent-pixel value:

        boost^2 = 1 + (2 / N) * sum_{k=1}^{N-1} (N - k) * rho(k),

    with ``rho(k) = C(k) / C(0)`` and ``N`` the window height. Only the
    cross-dispersion correlation enters (the dispersion-axis correlation is
    irrelevant to the collapse), so no isotropy assumption is needed.

    Parameters
    ----------
    data, error : array_like
        The 2-D spectrum and its per-pixel 1-sigma error.
    window : tuple of int
        ``(lo, hi)`` half-open range on the cross-dispersion axis (the rows the
        collapse sums).
    dispersion_axis : int
        The dispersion (wavelength) axis (``1`` or ``0``).
    source_mask : array_like of bool, optional
        ``True`` where the cutout holds source flux (excluded from the
        background). Defaults to :func:`noobfriend.core.imgutils.segment`'s
        detections; the collapse window is always excluded too.
    max_lag : int, default 8
        Largest cross-dispersion lag estimated (cover the correlation length).
    min_sky : int, default 100
        Minimum background pixels required; below this returns ``1.0``.

    Returns
    -------
    float
        The multiplicative boost (``>= 1`` for positively correlated noise);
        ``1.0`` when it cannot be estimated.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D.
    """
    img = np.asarray(data, dtype=float)
    err = np.asarray(error, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {img.shape}.")
    cross_axis = 1 - dispersion_axis
    lo, hi = int(window[0]), int(window[1])
    n_window = hi - lo
    if n_window < 1:
        return 1.0

    if source_mask is None:
        from noobfriend.core.imgutils import segment

        source = segment(np.nan_to_num(img)) > 0
    else:
        source = np.asarray(source_mask, dtype=bool)

    # Orient the cross-dispersion axis first, so lags run along axis 0.
    data_c = np.moveaxis(img, cross_axis, 0)
    err_c = np.moveaxis(err, cross_axis, 0)
    source_c = np.moveaxis(source, cross_axis, 0)
    sky = (~source_c) & np.isfinite(data_c) & np.isfinite(err_c) & (err_c > 0)
    sky[lo:hi, :] = False  # the trace itself is not background
    if int(sky.sum()) < min_sky:
        return 1.0

    resid = np.where(sky, data_c - data_c[sky].mean(), 0.0)
    c0 = float(np.sum(resid * resid)) / float(np.sum(sky))
    if c0 <= 0:
        return 1.0

    correction = 0.0
    for k in range(1, min(int(max_lag), n_window - 1) + 1):
        pairs = float(np.sum(sky[:-k] & sky[k:]))
        if pairs <= 0:
            continue
        ck = float(np.sum(resid[:-k] * resid[k:])) / pairs
        correction += (n_window - k) * (ck / c0)
    return float(np.sqrt(max(1.0 + (2.0 / n_window) * correction, 0.0)))
