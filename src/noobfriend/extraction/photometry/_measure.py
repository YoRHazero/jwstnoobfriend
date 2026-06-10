"""Native-grid, coverage-weighted measurement of one band.

Given a band and the union aperture expressed on that band's own native grid as
a fractional coverage map ``w in [0, 1]``, the flux is the coverage-weighted sum
``sum(w * data)`` over finite pixels. Because the science data is never
resampled, the per-native-pixel ``flux_scale_mjy`` calibration applies by
construction (no pixel-area factor), and the native error pixels stay
independent so the quadrature error sum is valid. Partially covered boundary
pixels enter proportionally, exactly as in exact aperture photometry — there is
no hard in/out threshold on the measurement grid.
"""

from dataclasses import dataclass

import numpy as np

from noobfriend.extraction.photometry._array import native_float64
from noobfriend.extraction.photometry._band import Band


@dataclass(frozen=True)
class BandPhotometry:
    """Coverage-weighted photometry for one band through the union aperture.

    Attributes
    ----------
    band : str
        Band name.
    wavelength : float or None
        Effective wavelength in microns, if supplied.
    wavelength_error : tuple[float, float] or None
        Left/right wavelength uncertainty, if supplied.
    flux : float
        Coverage-weighted sum ``sum(w * data)`` over finite aperture pixels, in
        the band's raw per-pixel units.
    error : float or None
        Primary 1-sigma aperture error (raw units): the *correlation-aware*
        estimate when it can be computed (see
        :mod:`noobfriend.extraction.photometry._noise`), otherwise the
        :attr:`error_uncorrelated` fallback; ``None`` only when neither is
        available.
    flux_mjy : float or None
        :attr:`flux` converted to mJy when ``flux_scale_mjy`` is known.
    error_mjy : float or None
        :attr:`error` converted to mJy when both are known.
    error_uncorrelated : float or None
        Independent-pixel quadrature error ``sqrt(sum(w**2 * sigma**2))``, or
        ``None`` if the band has no error image. Underestimates correlated
        (e.g. drizzled) noise; kept beside :attr:`error` for reference.
    flux_scale_mjy : float or None
        Multiplicative scale from one raw per-pixel value to mJy.
    flux_unit : str or None
        Original image unit or calibration source behind the scale.
    covered_area : float
        Effective aperture area on this band's grid, ``sum(w)`` (in native
        pixels). Fractional because boundary pixels are partially covered.
    valid_area : float
        Effective aperture area with finite data, ``sum(w)`` over finite pixels.
    bad_fraction : float
        ``1 - valid_area / covered_area``: the fraction of the aperture area
        that is missing (non-finite) in this band.
    background_level : float or None
        Median per-pixel sky background (raw units), or ``None`` when the sky
        could not be sampled (e.g. too few blank pixels).
    snr : float or None
        Detection signal-to-noise ``(flux - background_level * covered_area) /
        error``, with the background taken as ``0`` when
        :attr:`background_level` is ``None``; ``None`` when no error is known.
    flagged : bool
        ``True`` when :attr:`bad_fraction` meets the configured threshold.
    """

    band: str
    wavelength: float | None
    wavelength_error: tuple[float, float] | None
    flux: float
    error: float | None
    flux_mjy: float | None
    error_mjy: float | None
    error_uncorrelated: float | None
    flux_scale_mjy: float | None
    flux_unit: str | None
    covered_area: float
    valid_area: float
    bad_fraction: float
    background_level: float | None
    snr: float | None
    flagged: bool


def measure_band(
    band: Band,
    coverage: np.ndarray,
    *,
    flag_bad_fraction: float,
) -> BandPhotometry:
    """Measure one band through a native-grid coverage map.

    Parameters
    ----------
    band : Band
        The band, carrying native ``data`` / ``error`` and calibration.
    coverage : numpy.ndarray
        Fractional coverage ``w in [0, 1]`` of the union aperture on this band's
        native grid (see :func:`noobfriend.extraction.photometry._coverage`).
    flag_bad_fraction : float
        Bands whose :attr:`BandPhotometry.bad_fraction` is at least this value
        are flagged.

    Returns
    -------
    BandPhotometry
    """
    weight = native_float64(coverage)
    data = native_float64(band.data)
    in_aperture = weight > 0.0
    finite = in_aperture & np.isfinite(data)

    covered_area = float(weight[in_aperture].sum())
    valid_area = float(weight[finite].sum())
    bad_fraction = 1.0 - valid_area / covered_area if covered_area > 0.0 else 1.0
    flux = (
        float(np.sum(weight[finite] * data[finite]))
        if valid_area > 0.0
        else float("nan")
    )

    error_uncorrelated = _measure_error(band, weight, finite, valid_area)

    scale = band.flux_scale_mjy
    flux_mjy = None if scale is None else float(flux * scale)
    # The primary error defaults to the uncorrelated value; the correlation-aware
    # estimate (when computable) replaces it in measure_aperture_noise.
    error = error_uncorrelated
    error_mjy = None if (scale is None or error is None) else float(error * abs(scale))

    flagged = bad_fraction >= flag_bad_fraction
    return BandPhotometry(
        band=band.name,
        wavelength=band.wavelength,
        wavelength_error=band.wavelength_error,
        flux=flux,
        error=error,
        flux_mjy=flux_mjy,
        error_mjy=error_mjy,
        error_uncorrelated=error_uncorrelated,
        flux_scale_mjy=scale,
        flux_unit=band.flux_unit,
        covered_area=covered_area,
        valid_area=valid_area,
        bad_fraction=bad_fraction,
        background_level=None,
        snr=aperture_snr(flux, error, None, covered_area),
        flagged=flagged,
    )


def aperture_snr(
    flux: float,
    error: float | None,
    background_level: float | None,
    covered_area: float,
) -> float | None:
    """Background-subtracted detection SNR, or ``None`` when not computable.

    ``snr = (flux - background_level * covered_area) / error``; the background is
    taken as ``0`` when ``background_level`` is ``None``. Returns ``None`` when
    ``error`` is missing or non-positive, or when the result is non-finite.
    """
    if error is None or not error > 0.0:
        return None
    background = 0.0 if background_level is None else background_level * covered_area
    value = (flux - background) / error
    return float(value) if np.isfinite(value) else None


def _measure_error(
    band: Band,
    weight: np.ndarray,
    finite: np.ndarray,
    valid_area: float,
) -> float | None:
    """Coverage-weighted quadrature error over the finite aperture pixels.

    Returns ``None`` when the band has no error image, and ``NaN`` when the
    aperture is empty or any finite-data pixel lacks a usable (finite,
    non-negative) error — dropping its unknown variance would bias the result
    low, matching the noobase reprojection convention.
    """
    if band.error is None:
        return None
    if valid_area <= 0.0:
        return float("nan")
    error = native_float64(band.error)
    usable = finite & np.isfinite(error) & (error >= 0.0)
    if not np.array_equal(usable, finite):
        return float("nan")
    return float(np.sqrt(np.sum((weight[finite] ** 2) * (error[finite] ** 2))))
