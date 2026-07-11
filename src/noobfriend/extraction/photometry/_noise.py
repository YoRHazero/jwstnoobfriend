"""Annotate a band's photometry with a correlation-aware error and SNR.

The domain glue between :class:`~noobfriend.extraction.photometry._measure.BandPhotometry`
and the generic :mod:`noobfriend.core.imgutils` noise primitive. It builds the
blank-sky mask (excluding the measurement aperture, the target's own segment,
and -- dilated -- every *other* source), estimates the sky autocovariance there,
folds it with the aperture weight map to get the honest aperture error, and
records the median sky background and the background-subtracted detection SNR.

Only this enrichment knows the JWST-specific "exact ``0.0`` is unexposed/flagged"
rule (via :func:`~noobfriend.extraction.photometry._array.valid_data_mask`); the
core primitive stays grid- and convention-neutral. When the sky is too small for
a stable estimate the band is returned unchanged -- ``error`` keeps its
uncorrelated fallback and ``background_level``/``snr`` follow from that.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_dilation

from noobfriend.core.imgutils import (
    correlated_sum_variance,
    noise_autocovariance,
    segment,
)
from noobfriend.extraction.photometry._array import valid_data_mask
from noobfriend.extraction.photometry._band import Band
from noobfriend.extraction.photometry._measure import BandPhotometry, aperture_snr


def _sky_mask(
    data: np.ndarray, coverage: np.ndarray, other_source_dilation: int
) -> np.ndarray:
    """Boolean blank-sky mask for the noise estimate.

    Excludes invalid pixels (non-finite or exact ``0.0``), the measurement
    aperture (``coverage > 0``), the target source's own segment (undilated, so
    good sky stays available right next to it), and every *other* detected
    source dilated by ``other_source_dilation`` to bury its sub-threshold wings.
    """
    valid = valid_data_mask(data)
    seg = segment(data)
    in_aperture = coverage > 0.0
    target_labels = {int(v) for v in np.unique(seg[in_aperture]) if v != 0}
    is_target = (
        np.isin(seg, list(target_labels))
        if target_labels
        else np.zeros(seg.shape, dtype=bool)
    )
    other = (seg > 0) & ~is_target
    if other_source_dilation > 0:
        other = binary_dilation(other, iterations=int(other_source_dilation))
    return valid & ~in_aperture & ~is_target & ~other


def measure_aperture_noise(
    band: Band,
    coverage: np.ndarray,
    formal: BandPhotometry,
    *,
    max_lag: int = 8,
    other_source_dilation: int = 2,
) -> BandPhotometry:
    """Return ``formal`` enriched with the correlation-aware error, background, SNR.

    Parameters
    ----------
    band : Band
        The band, carrying native ``data`` / ``error`` / calibration.
    coverage : numpy.ndarray
        The union aperture's fractional coverage on this band's grid (the
        measurement weight map).
    formal : BandPhotometry
        The result from :func:`~noobfriend.extraction.photometry._measure.measure_band`
        (its ``error`` is the uncorrelated fallback).
    max_lag : int, default 8
        Lag-window half-width for the sky autocovariance.
    other_source_dilation : int, default 2
        Dilation (pixels) applied to non-target sources when masking the sky.

    Returns
    -------
    BandPhotometry
        ``formal`` with ``error`` (and ``error_mjy``) replaced by the
        correlation-aware estimate, plus ``background_level`` and ``snr``. When
        the local sky is too small to estimate ``C(d)``, the error falls back to
        ``band.noise_kernel`` (the product's persisted field kernel, applied in
        stationary mode) if the band carries one, else ``formal`` is returned
        unchanged (the uncorrelated fallback).
    """
    data = np.asarray(band.data, dtype=float)
    cov = np.asarray(coverage, dtype=float)
    sky = _sky_mask(data, cov, other_source_dilation)
    try:
        autocov = noise_autocovariance(
            data, mask=sky, max_lag=max_lag, error=band.error
        )
    except ValueError:
        # Too little local sky to estimate C(d) here. Fall back to the product's
        # persisted field kernel (stationary; measured on the whole mosaic's sky)
        # when the band carries one -- still correlation-aware, unlike the plain
        # uncorrelated formula -- otherwise keep the uncorrelated `formal`.
        if band.noise_kernel is None:
            return formal
        error = float(np.sqrt(correlated_sum_variance(cov, band.noise_kernel)))
    else:
        error = float(np.sqrt(correlated_sum_variance(cov, autocov, error=band.error)))

    background_level = float(np.median(data[sky])) if sky.any() else 0.0
    scale = band.flux_scale_mjy
    error_mjy = None if scale is None else float(error * abs(scale))
    return replace(
        formal,
        error=error,
        error_mjy=error_mjy,
        background_level=background_level,
        snr=aperture_snr(formal.flux, error, background_level, formal.covered_area),
    )
