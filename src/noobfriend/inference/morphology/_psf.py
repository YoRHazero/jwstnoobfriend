"""Resolve a per-band PSF kernel for rendering.

A band's own empirical ePSF (from :mod:`noobfriend.extraction.psf`) is preferred
and used as given. When a band carries no PSF, a theoretical one is built lazily
with STPSF -- a convenience, explicitly the second-best option: STPSF gives a
detector-frame PSF that is broader-winged reality only approximately, and on a
drizzled mosaic it does not capture the resampled, orientation-dependent effective
PSF. This module is internal.
"""

from __future__ import annotations

import re

import numpy as np

#: A JWST filter name: ``F`` then digits encoding wavelength in centi-microns.
_FILTER_RE = re.compile(r"[fF](\d+)")


#: Cache of theoretical PSFs keyed by (filter, oversample, pixscale) -- the PSF is
#: independent of the target, so it is computed once and reused across sources.
_PSF_CACHE: dict[tuple[str, int, float | None], np.ndarray] = {}


def resolve_psf(
    psf: np.ndarray | None,
    *,
    band: str,
    oversample: int,
    pixscale: float | None = None,
) -> np.ndarray:
    """Return a normalised (sum-1) oversampled PSF kernel for one band.

    Parameters
    ----------
    psf : numpy.ndarray or None
        The band's PSF kernel; assumed already sampled at the model ``oversample``.
        ``None`` triggers the theoretical STPSF fallback.
    band : str
        Filter name, used to pick the instrument for the fallback.
    oversample : int
        Oversampling factor for the theoretical fallback.
    pixscale : float, optional
        Native pixel scale of the render grid, in arcsec. When given, the
        theoretical PSF is sampled at this scale (so it matches a drizzled mosaic
        grid rather than the detector's native scale); STPSF data files are
        auto-downloaded on first use.

    Returns
    -------
    numpy.ndarray
        A 2-D kernel summing to 1.

    Raises
    ------
    ValueError
        If a supplied PSF is not a positive 2-D array, or no PSF is given and STPSF
        is unavailable / the filter's instrument is unknown.
    """
    if psf is not None:
        arr = np.asarray(psf, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"band {band!r} psf must be 2-D; got shape {arr.shape}.")
        total = float(arr.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"band {band!r} psf must have a positive finite sum.")
        return arr / total
    return _theoretical_psf(band, oversample, pixscale)


def _theoretical_psf(band: str, oversample: int, pixscale: float | None) -> np.ndarray:
    """Build (and cache) a theoretical PSF with STPSF (lazy import).

    Notes
    -----
    STPSF returns a *detector-frame* PSF; sampling it at the mosaic ``pixscale``
    matches the render grid but does not capture the drizzle convolution -- an
    approximation that is nonetheless far better than a Gaussian for the core.
    """
    key = (
        band.upper(),
        int(oversample),
        None if pixscale is None else round(float(pixscale), 5),
    )
    if key in _PSF_CACHE:
        return _PSF_CACHE[key]
    try:
        import stpsf
    except ImportError as exc:  # pragma: no cover - exercised only without STPSF
        raise ValueError(
            f"band {band!r} has no PSF and STPSF is not installed. Provide a PSF "
            "array, or install the optional 'psf' extra (uv sync --extra psf)."
        ) from exc

    instrument_name = _instrument_for(band)
    instrument = getattr(stpsf, instrument_name)()
    instrument.filter = band.upper()
    if pixscale is not None:
        instrument.pixelscale = float(pixscale)
    hdul = instrument.calc_psf(oversample=oversample, fov_pixels=17)
    data = hdul[0].data  # oversampled extension
    total = float(data.sum())
    if total <= 0:
        raise ValueError(f"band {band!r} STPSF PSF has non-positive sum.")
    out = np.asarray(data, dtype=float) / total
    _PSF_CACHE[key] = out
    return out


def _instrument_for(band: str) -> str:
    """Map a JWST filter to its STPSF instrument by its encoded wavelength.

    The digits after ``F`` give the wavelength in centi-microns (``F115W`` ->
    1.15 um, ``F1000W`` -> 10 um), so NIRCam (< 5 um) and MIRI (>= 5 um) separate
    cleanly -- unlike a string prefix, which confuses ``F115W`` with ``F1130W``.
    """
    match = _FILTER_RE.match(band)
    if match is None:
        raise ValueError(
            f"cannot infer the instrument for filter {band!r}; pass an explicit PSF."
        )
    wavelength_um = int(match.group(1)) * 0.01
    if wavelength_um < 5.0:
        return "NIRCam"
    if wavelength_um <= 28.0:
        return "MIRI"
    raise ValueError(
        f"cannot infer the instrument for filter {band!r}; pass an explicit PSF."
    )
