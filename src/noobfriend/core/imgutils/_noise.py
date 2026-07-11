"""Correlated-noise aperture variance from a stationary sky autocovariance.

Mechanism-only, pixel-space (like :mod:`noobfriend.core.imgutils._segment`): no
WCS, no photometry. The problem it solves is the variance of a *weighted pixel
sum* when the noise is spatially **correlated** -- the regime where the naive
``sqrt(sum(w**2 * sigma**2))`` (which assumes independent pixels) underestimates
the true error, e.g. on drizzled / resampled mosaics.

For a weighted sum ``S = sum_i w_i x_i`` with zero-mean noise of covariance
``K_ij``,

    Var(S) = sum_i sum_j w_i w_j K_ij.

Assuming the noise is **stationary** (``K_ij`` depends only on the separation
``d = r_i - r_j``), ``K_ij = C(d)`` and the double sum collapses to

    Var(S) = sum_d C(d) * A(d),     A(d) = sum_i w_i w_{i-d},

where ``C`` is the noise autocovariance (estimated from blank sky by
:func:`noise_autocovariance`) and ``A`` is the autocorrelation of the weight map
(pure geometry). :func:`correlated_sum_variance` evaluates that sum.

Both ingredients reduce to small FFTs, so a band costs ~1 ms. The estimator
assumes the supplied ``mask`` cleanly excludes real sources and that the sky
noise is stationary across the field; the returned :attr:`NoiseAutocovariance.autocorr`
lets a caller check the correlation has decayed within ``max_lag``.
"""

from dataclasses import dataclass

import numpy as np


def _lagged_autocorr(a: np.ndarray, max_lag: int) -> np.ndarray:
    """Linear autocorrelation ``sum_r a(r) a(r + d)`` over a centred lag window.

    Computed by FFT with zero-padding of ``max_lag`` on each axis so the
    circular wrap never connects two real (non-padding) pixels; the result is
    therefore the *linear* correlation. Pixels outside the support of ``a``
    (i.e. where ``a == 0``) contribute nothing, which is what makes the masked
    covariance and the weight-map autocorrelation both fall out of this one
    helper.

    Parameters
    ----------
    a : numpy.ndarray
        2-D array (sky residuals, a 0/1 mask, or a weight map).
    max_lag : int
        Half-width of the returned window in pixels.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2 * max_lag + 1, 2 * max_lag + 1)``, centred so the
        zero-lag value is at ``[max_lag, max_lag]``.
    """
    n_rows, n_cols = a.shape
    pad = (n_rows + max_lag, n_cols + max_lag)
    spec = np.fft.rfft2(a, s=pad)
    full = np.fft.irfft2(np.abs(spec) ** 2, s=pad)
    full = np.roll(full, (max_lag, max_lag), axis=(0, 1))
    return full[: 2 * max_lag + 1, : 2 * max_lag + 1]


@dataclass(frozen=True)
class NoiseAutocovariance:
    """Stationary noise autocovariance ``C(d)`` estimated from masked sky.

    Attributes
    ----------
    cov : numpy.ndarray
        ``C(d)`` on a centred lag window of shape ``(2 * max_lag + 1,
        2 * max_lag + 1)``; ``cov[max_lag, max_lag]`` is ``C(0)``, the per-pixel
        noise variance.
    max_lag : int
        Half-width of the lag window, in pixels.
    n_pairs : numpy.ndarray or None
        Number of valid sky pixel-pairs averaged at each lag (same shape as
        ``cov``); a reliability diagnostic -- small counts at the largest lags
        mean a noisy ``C`` there. ``None`` on a model reconstituted from a
        persisted kernel: the pair counts are a build-time diagnostic and are
        not written to disk.
    error_var : float or None
        Mean of ``error ** 2`` over the sky, ``<err^2>``. Set only when an
        ``error`` map was supplied to :func:`noise_autocovariance`; it is the
        amplitude the per-pixel error map is calibrated against in the hybrid
        variance (see :func:`correlated_sum_variance`). ``None`` otherwise.
    """

    cov: np.ndarray
    max_lag: int
    n_pairs: np.ndarray | None
    error_var: float | None

    @property
    def variance(self) -> float:
        """Per-pixel noise variance ``C(0)``."""
        return float(self.cov[self.max_lag, self.max_lag])

    @property
    def sigma(self) -> float:
        """Per-pixel noise RMS, ``sqrt(C(0))``."""
        return float(np.sqrt(max(self.variance, 0.0)))

    @property
    def autocorr(self) -> np.ndarray:
        """Normalised autocorrelation ``rho(d) = C(d) / C(0)`` (decay diagnostic)."""
        variance = self.variance
        if variance <= 0.0:
            return np.zeros_like(self.cov)
        return self.cov / variance


def noise_autocovariance(
    image: np.ndarray,
    *,
    mask: np.ndarray,
    max_lag: int = 8,
    error: np.ndarray | None = None,
) -> NoiseAutocovariance:
    """Estimate the stationary noise autocovariance ``C(d)`` from blank-sky pixels.

    The masked pixels are mean-subtracted, then per lag ``d`` in
    ``[-max_lag, max_lag]^2``,

        C(d) = ( sum over valid pairs of x0(r) x0(r + d) ) / ( pair count at d ),

    with both the numerator (from the residual image) and the pair counts (from
    the mask) computed by FFT (see :func:`_lagged_autocorr`). Only pairs whose
    *both* pixels are in the mask contribute.

    Parameters
    ----------
    image : numpy.ndarray
        2-D image to estimate the noise from.
    mask : numpy.ndarray
        Boolean array, ``True`` where ``image`` is a blank-sky noise sample.
        Combined with ``isfinite(image)``. **Required**: for real frames it must
        exclude detected sources (e.g. ``segment(image) == 0``), or sources bias
        ``C`` high.
    max_lag : int, default 8
        Half-width of the lag window in pixels. Choose it to cover the noise
        correlation length, not the aperture size; verify via
        :attr:`NoiseAutocovariance.autocorr`.
    error : numpy.ndarray, optional
        Per-pixel 1-sigma error (depth) map. When given, its sky mean-square is
        stored as :attr:`NoiseAutocovariance.error_var`, enabling the hybrid
        (depth-aware) path of :func:`correlated_sum_variance`.

    Returns
    -------
    NoiseAutocovariance

    Raises
    ------
    ValueError
        If ``image`` is not 2-D; if ``mask``/``error`` shapes mismatch; if
        ``max_lag`` is non-positive or its window does not fit the image; if the
        mask leaves fewer than ``(2 * max_lag + 1) ** 2`` sky pixels; or if no
        finite sky error pixel exists when ``error`` is given.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D; got shape {img.shape}.")
    sky_mask = np.asarray(mask, dtype=bool)
    if sky_mask.shape != img.shape:
        raise ValueError(
            f"mask shape {sky_mask.shape} does not match image shape {img.shape}."
        )
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1; got {max_lag}.")
    if 2 * max_lag + 1 > min(img.shape):
        raise ValueError(
            f"max_lag={max_lag} window ({2 * max_lag + 1}) does not fit image "
            f"shape {img.shape}."
        )

    sky = sky_mask & np.isfinite(img)
    n_sky = int(sky.sum())
    if n_sky < (2 * max_lag + 1) ** 2:
        raise ValueError(
            f"too few sky pixels ({n_sky}) to estimate C(d) up to max_lag="
            f"{max_lag}; need at least {(2 * max_lag + 1) ** 2}."
        )

    residual = np.where(sky, img - float(img[sky].mean()), 0.0)
    num = _lagged_autocorr(residual, max_lag)
    den = _lagged_autocorr(sky.astype(float), max_lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(den > 0.5, num / den, 0.0)

    error_var: float | None = None
    if error is not None:
        err = np.asarray(error, dtype=float)
        if err.shape != img.shape:
            raise ValueError(
                f"error shape {err.shape} does not match image shape {img.shape}."
            )
        err_sky = sky & np.isfinite(err)
        if not err_sky.any():
            raise ValueError("no finite error pixels inside the sky mask.")
        error_var = float(np.mean(err[err_sky] ** 2))

    return NoiseAutocovariance(
        cov=cov, max_lag=int(max_lag), n_pairs=den, error_var=error_var
    )


def correlated_sum_variance(
    weights: np.ndarray,
    autocov: NoiseAutocovariance,
    *,
    error: np.ndarray | None = None,
) -> float:
    """Variance of the weighted sum ``sum_i w_i x_i`` under correlated noise.

    Two modes:

    - **stationary** (``error`` omitted): one noise level across the aperture,

          Var = sum_d C(d) * A_w(d),     A_w = autocorr(weights).

    - **hybrid** (``error`` given): keep the per-pixel depth from the error map,
      taking only the correlation shape and the amplitude from the sky,

          Var = ( 1 / <err^2> ) * sum_d C(d) * A_{w.err}(d),

      where ``A_{w.err}`` is the autocorrelation of ``weights * error`` and
      ``<err^2>`` is :attr:`NoiseAutocovariance.error_var`. This reduces exactly
      to the stationary form when ``error`` is uniform, but tracks varying
      exposure depth across the aperture when it is not.

    With white noise (``C = sigma^2`` at lag 0, else 0) both modes collapse to
    the independent-pixel ``sigma^2 * sum_i w_i^2``.

    Parameters
    ----------
    weights : numpy.ndarray
        Aperture weight map (e.g. fractional coverage), on the **same pixel
        grid** as the image that produced ``autocov``. Non-finite entries are
        treated as zero.
    autocov : NoiseAutocovariance
        Sky autocovariance from :func:`noise_autocovariance`.
    error : numpy.ndarray, optional
        Per-pixel 1-sigma error map for the hybrid path. Must match ``weights``
        in shape; non-finite entries are treated as zero. Requires ``autocov``
        to have been built with the same ``error`` (so ``error_var`` is set).

    Returns
    -------
    float
        The variance of the weighted sum (caller takes ``sqrt`` for an error).

    Raises
    ------
    ValueError
        If ``weights`` is not 2-D; if ``error`` is given but ``autocov`` was
        built without one; or if ``error`` and ``weights`` shapes mismatch.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D; got shape {w.shape}.")

    if error is None:
        sum_map = np.nan_to_num(w, nan=0.0)
        scale = 1.0
    else:
        if autocov.error_var is None:
            raise ValueError(
                "hybrid variance needs an `error`-aware autocov; rebuild "
                "noise_autocovariance with error= before passing error here."
            )
        err = np.asarray(error, dtype=float)
        if err.shape != w.shape:
            raise ValueError(
                f"error shape {err.shape} does not match weights shape {w.shape}."
            )
        sum_map = np.nan_to_num(w * err, nan=0.0)
        scale = 1.0 / autocov.error_var

    weight_autocorr = _lagged_autocorr(sum_map, autocov.max_lag)
    variance = float(np.sum(autocov.cov * weight_autocorr) * scale)
    return max(variance, 0.0)
