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


def component_centres(
    spectrum: NoobSpectrum, components: tuple[ResolvedComponent, ...]
) -> list[float]:
    """Return each component's systemic observed centre ``(1 + z) * rest``."""
    return [(1.0 + spectrum.z) * c.line.rest_wavelength for c in components]


def select_window(
    spectrum: NoobSpectrum,
    components: tuple[ResolvedComponent, ...],
    *,
    window: tuple[float, float] | None = None,
    pad_kms: float = 5000.0,
    min_points: int = 8,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Select the finite in-window pixels to fit.

    Parameters
    ----------
    spectrum : NoobSpectrum
        The spectrum.
    components : tuple of ResolvedComponent
        The compiled components, whose systemic centres bracket the window.
    window : tuple of float, optional
        Explicit ``(lo, hi)`` wavelength bounds; overrides the automatic span.
    pad_kms : float, default 5000.0
        Velocity padding added on each side of the bracketing centres when
        ``window`` is not given.
    min_points : int, default 8
        Minimum finite pixels required.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask over the spectrum arrays selecting finite in-window pixels.
    bounds : tuple of float
        The ``(lo, hi)`` wavelength bounds used.

    Raises
    ------
    ValueError
        If fewer than ``min_points`` finite pixels fall in the window.
    """
    centres = component_centres(spectrum, components)
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
    if int(mask.sum()) < min_points:
        raise ValueError(
            f"window [{lo:g}, {hi:g}] holds only {int(mask.sum())} finite pixels "
            f"(need >= {min_points}); widen pad_kms or pass an explicit window."
        )
    return mask, (lo, hi)
