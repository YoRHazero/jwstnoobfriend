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


def continuum_noise(
    flux: ArrayLike,
    error: ArrayLike,
    *,
    wavelength: ArrayLike,
    exclude: ArrayLike | None = None,
    lags: int = 0,
    degree: int = 3,
    nsigma: float = 3.0,
    niter: int = 5,
    min_pixels: int = 10,
) -> tuple[float, np.ndarray]:
    """Estimate the error scale and dispersion-axis correlation of a 1-D spectrum.

    Fits a sigma-clipped polynomial continuum (the clipping rejects lines),
    normalizes the residual by the quoted error, ``u = (flux - continuum) /
    error``, and returns its robust standard deviation together with its
    autocorrelation at native-pixel lags ``1 ... lags``. Pixel pairs are only
    formed between kept pixels, so excluded pixels and gaps leave the true
    separations intact.

    Parameters
    ----------
    flux, error : array_like
        The 1-D spectrum and its 1-sigma error on the native pixel grid.
    wavelength : array_like
        Pixel wavelengths (used only for the continuum polynomial).
    exclude : array_like of bool, optional
        ``True`` on pixels not to use (lines, bad pixels). Sigma-clipping is
        applied to the rest.
    lags : int, default 0
        Largest lag of the autocorrelation; ``0`` estimates the scale only.
    degree : int, default 3
        Continuum polynomial degree.
    nsigma : float, default 3.0
        Sigma-clip threshold.
    niter : int, default 5
        Maximum clipping iterations.
    min_pixels : int, default 10
        Minimum kept pixels required.

    Returns
    -------
    scale : float
        Robust standard deviation of ``u`` (``1`` when the errors are right).
    correlations : numpy.ndarray
        ``rho(1), ..., rho(lags)`` of ``u``.

    Raises
    ------
    ValueError
        If too few line-free pixels remain or a requested lag has no pixel
        pairs.

    Notes
    -----
    The polynomial absorbs low-frequency noise, so both the scale and the
    correlations are biased low when the noise is strongly correlated; prefer
    :func:`background_noise` when a 2-D spectrum is available.
    """
    wl = np.asarray(wavelength, dtype=float)
    fl = np.asarray(flux, dtype=float)
    er = np.asarray(error, dtype=float)
    candidate = np.isfinite(wl) & np.isfinite(fl) & np.isfinite(er) & (er > 0)
    if exclude is not None:
        candidate &= ~np.asarray(exclude, dtype=bool)
    if int(candidate.sum()) < max(degree + 2, min_pixels):
        raise ValueError(
            f"only {int(candidate.sum())} usable pixels; continuum noise needs "
            f"at least {max(degree + 2, min_pixels)}."
        )

    keep = candidate.copy()
    resid = np.zeros_like(fl)
    for _ in range(niter):
        coef = np.polyfit(wl[keep], fl[keep], degree, w=1.0 / er[keep])
        resid = np.where(
            candidate, (fl - np.polyval(coef, wl)) / np.where(candidate, er, 1.0), 0.0
        )
        scatter = _robust_std(resid[keep])
        if scatter <= 0:
            raise ValueError("continuum residual has zero scatter.")
        new_keep = candidate & (np.abs(resid) <= nsigma * scatter)
        if int(new_keep.sum()) < max(degree + 2, min_pixels):
            raise ValueError("sigma-clipping left too few line-free pixels.")
        converged = bool(np.array_equal(new_keep, keep))
        keep = new_keep
        if converged:
            break

    scale = _robust_std(resid[keep])
    return scale, _lag_correlations(np.where(keep, resid, 0.0), keep, lags)


def background_noise(
    data: ArrayLike,
    error: ArrayLike,
    *,
    window: tuple[int, int],
    dispersion_axis: int,
    source_mask: ArrayLike | None = None,
    lags: int = 0,
    min_sky: int = 100,
) -> tuple[float, np.ndarray]:
    """Estimate the collapsed spectrum's error scale and correlation from sky pixels.

    Normalizes the source-free background by its error, ``n = data / error``,
    and estimates its 2-D autocovariance ``C(dr, dc)`` (``dr`` across, ``dc``
    along dispersion). Collapsing ``window`` rows then has, per column,

        Var  = sum_{r, r'} w_r w_r' C(r' - r, 0)
        Cov_k = sum_{r, r'} w_r w_r' C(r' - r, k),

    with ``w_r`` each window row's median error. The scale is
    ``sqrt(Var / sum w_r**2)`` (the inflation over the naive quadrature error,
    including any per-pixel error mis-calibration through ``C(0, 0)``) and
    ``rho(k) = Cov_k / Var``.

    Parameters
    ----------
    data, error : array_like
        The 2-D spectrum and its per-pixel 1-sigma error.
    window : tuple of int
        ``(lo, hi)`` half-open collapse range on the cross-dispersion axis.
    dispersion_axis : int
        The dispersion (wavelength) axis, ``0`` or ``1``.
    source_mask : array_like of bool, optional
        ``True`` where the cutout holds source flux. Defaults to
        :func:`noobfriend.core.imgutils.segment`'s detections; the collapse
        window is always excluded too.
    lags : int, default 0
        Largest dispersion lag of the correlation; ``0`` estimates the scale
        only.
    min_sky : int, default 100
        Minimum background pixels required.

    Returns
    -------
    scale : float
        Error scale of the collapsed spectrum.
    correlations : numpy.ndarray
        ``rho(1), ..., rho(lags)`` of the collapsed spectrum.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, the window is empty, the background has too
        few pixels, or the estimated variance is not positive.
    """
    img = np.asarray(data, dtype=float)
    err = np.asarray(error, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {img.shape}.")
    lo, hi = int(window[0]), int(window[1])
    if hi - lo < 1:
        raise ValueError(f"collapse window {window} is empty.")

    if source_mask is None:
        from noobfriend.core.imgutils import segment

        source = segment(np.nan_to_num(img)) > 0
    else:
        source = np.asarray(source_mask, dtype=bool)

    cross_axis = 1 - dispersion_axis
    data_c = np.moveaxis(img, cross_axis, 0)
    err_c = np.moveaxis(err, cross_axis, 0)
    source_c = np.moveaxis(source, cross_axis, 0)
    sky = (~source_c) & np.isfinite(data_c) & np.isfinite(err_c) & (err_c > 0)
    sky[lo:hi, :] = False
    if int(sky.sum()) < min_sky:
        raise ValueError(
            f"only {int(sky.sum())} background pixels; background noise needs {min_sky}."
        )

    normalized = np.where(sky, data_c / np.where(sky, err_c, 1.0), 0.0)
    normalized = np.where(sky, normalized - normalized[sky].mean(), 0.0)
    weights = np.nanmedian(
        np.where(np.isfinite(err_c[lo:hi]), err_c[lo:hi], np.nan), axis=1
    )
    if not np.all(np.isfinite(weights)):
        raise ValueError("a collapse-window row has no finite error.")

    n_rows = hi - lo
    covariances = np.zeros(lags + 1)
    for dc in range(lags + 1):
        for dr in range(-(n_rows - 1), n_rows):
            c = _autocovariance(normalized, sky, dr, dc)
            # Rows r, r' = r + dr inside the window weigh w_r * w_{r'}.
            pairs = sum(
                weights[r] * weights[r + dr]
                for r in range(n_rows)
                if 0 <= r + dr < n_rows
            )
            covariances[dc] += pairs * c
    variance = covariances[0]
    if variance <= 0:
        raise ValueError("estimated background variance is not positive.")
    scale = float(np.sqrt(variance / np.sum(weights**2)))
    return scale, covariances[1:] / variance


def _autocovariance(values: np.ndarray, valid: np.ndarray, dr: int, dc: int) -> float:
    """Mean product of valid pixel pairs separated by ``(dr, dc)``."""
    rows, cols = values.shape
    r0, r1 = max(0, -dr), rows - max(0, dr)
    if r1 <= r0 or dc >= cols:
        return 0.0
    a = values[r0:r1, : cols - dc]
    b = values[r0 + dr : r1 + dr, dc:]
    both = valid[r0:r1, : cols - dc] & valid[r0 + dr : r1 + dr, dc:]
    count = int(both.sum())
    if count == 0:
        raise ValueError(f"no background pixel pairs at separation ({dr}, {dc}).")
    return float(np.sum(np.where(both, a * b, 0.0))) / count


def _lag_correlations(values: np.ndarray, valid: np.ndarray, lags: int) -> np.ndarray:
    """Autocorrelation of ``values`` at lags ``1 ... lags`` over valid pairs."""
    if lags < 0:
        raise ValueError("lags must be non-negative.")
    centered = np.where(valid, values - values[valid].mean(), 0.0)
    c0 = float(np.sum(centered**2)) / int(valid.sum())
    correlations = np.empty(lags)
    for k in range(1, lags + 1):
        both = valid[:-k] & valid[k:]
        count = int(both.sum())
        if count == 0:
            raise ValueError(f"no pixel pairs at lag {k}.")
        correlations[k - 1] = float(np.sum(centered[:-k] * centered[k:])) / count / c0
    return correlations
