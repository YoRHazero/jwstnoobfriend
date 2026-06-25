"""Fitting-window selection.

The fit runs on a slice of the spectrum around the modelled lines, not the whole
array: :func:`select_window` returns a boolean mask of the finite pixels within
``pad_kms`` of the bracketing line centres (or an explicit ``window``). Keeping
the window local makes the continuum a low-order polynomial and stops far-away
features from biasing the fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.spectrum._setup import ResolvedComponent
    from noobfriend.inference.spectrum._spectrum import NoobSpectrum

#: Speed of light in km/s.
C_KMS: float = 299792.458


def component_centres(components: tuple[ResolvedComponent, ...]) -> list[float]:
    """Return each component's systemic observed centre in the spectrum frame."""
    return [c.centre_wavelength for c in components]


def select_window(
    spectrum: NoobSpectrum,
    components: tuple[ResolvedComponent, ...],
    *,
    window: tuple[float, float] | None = None,
    pad_kms: float = 5000.0,
    min_points: int = 8,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Select the finite in-window pixels to fit.

    Parameters
    ----------
    spectrum : NoobSpectrum
        The spectrum. Its ``mask_excluded`` pixels are always dropped.
    components : tuple of ResolvedComponent
        The compiled components, whose systemic centres bracket the window.
    window : tuple of float, optional
        Explicit ``(lo, hi)`` wavelength bounds; overrides the automatic span.
    pad_kms : float, default 5000.0
        Velocity padding added on each side of the bracketing centres when
        ``window`` is not given.
    min_points : int, default 8
        Minimum finite pixels required.
    exclude : numpy.ndarray, optional
        An extra boolean mask (``True`` = drop) layered on top of the spectrum's
        ``mask_excluded`` -- the fit-time exclusions from
        :meth:`~noobfriend.inference.spectrum.LineFitSetup.with_mask`.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask over the spectrum arrays selecting finite in-window pixels.
    bounds : tuple of float
        The ``(lo, hi)`` extent of the *selected* pixels, i.e. clamped to the
        first and last valid wavelength inside the requested window -- never
        beyond real data. The nominal velocity-padded request can reach past the
        spectrum's coverage; returning the realised extent keeps callers (the
        pre-fit preview, any band overlay) on pixels that actually exist.

    Raises
    ------
    ValueError
        If fewer than ``min_points`` finite pixels fall in the window.
    """
    centres = component_centres(components)
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
    else:
        lo = min(centres) * (1.0 - pad_kms / C_KMS)
        hi = max(centres) * (1.0 + pad_kms / C_KMS)
    wl = spectrum.wavelength
    mask = (
        (wl >= lo)
        & (wl <= hi)
        & np.isfinite(wl)
        & np.isfinite(spectrum.flux)
        & np.isfinite(spectrum.error)
        & (spectrum.error > 0)
    )
    # ``mask`` is True = selected; the exclusion masks are True = drop, so AND
    # in their negation.
    if spectrum.mask_excluded is not None:
        mask &= ~spectrum.mask_excluded
    if exclude is not None:
        mask &= ~exclude
    if int(mask.sum()) < min_points:
        raise ValueError(
            f"window [{lo:g}, {hi:g}] holds only {int(mask.sum())} finite pixels "
            f"(need >= {min_points}); widen pad_kms or pass an explicit window."
        )
    wl_in = wl[mask]
    return mask, (float(wl_in.min()), float(wl_in.max()))
