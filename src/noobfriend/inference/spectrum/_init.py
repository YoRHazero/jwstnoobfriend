"""Data-driven initial estimates feeding the priors.

The priors are weakly informative but *centred on the data*: the continuum on a
robust median (line cores masked out), the flux scale on the window's peak
excess. These are deliberately rough -- NUTS only needs a sensible basin, not a
precise start -- so the estimates here set prior centres/scales rather than a
separate ``initvals`` map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum._window import C_KMS

if TYPE_CHECKING:
    from noobfriend.inference.spectrum._setup import ResolvedComponent
    from noobfriend.inference.spectrum._spectrum import NoobSpectrum

_FWHM_TO_SIGMA: float = 1.0 / 2.3548200450309493
_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class InitialEstimates:
    """Prior-centring estimates derived from the window.

    Attributes
    ----------
    c0 : float
        Continuum level (flux units) at ``lambda0``.
    c1 : float
        Continuum slope (flux per wavelength); ``0`` for degree 0.
    c0_scale, c1_scale : float
        Weakly-informative prior scales for ``c0`` / ``c1``.
    lambda0 : float
        Continuum reference wavelength (window mean).
    flux_scale : float
        Positive scale for emission-flux priors (integrated line flux units).
    """

    c0: float
    c1: float
    c0_scale: float
    c1_scale: float
    lambda0: float
    flux_scale: float


def _robust_std(values: np.ndarray) -> float:
    """Median-absolute-deviation standard deviation (robust to the line)."""
    if values.size == 0:
        return 1.0
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(1.4826 * mad, 1e-30)


def initial_estimates(
    spectrum: NoobSpectrum,
    components: tuple[ResolvedComponent, ...],
    wl: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    degree: int,
) -> InitialEstimates:
    """Estimate continuum and flux scales over the window.

    Parameters
    ----------
    spectrum : NoobSpectrum
        The spectrum (for the redshift).
    components : tuple of ResolvedComponent
        Compiled components, whose centres define the masked line cores.
    wl, flux, error : numpy.ndarray
        The in-window wavelength, flux, and error.
    degree : int
        Continuum polynomial degree (0 or 1).

    Returns
    -------
    InitialEstimates
    """
    lambda0 = float(np.mean(wl))
    z = spectrum.z

    # Mask the line cores (a couple of nominal widths) before fitting continuum.
    core = np.zeros_like(wl, dtype=bool)
    for c in components:
        centre = (1.0 + z) * c.line.rest_wavelength
        fwhm_kms = c.template.fwhm_kms_init
        half = centre * (2.0 * fwhm_kms) / C_KMS
        core |= np.abs(wl - centre) <= half
    cont_mask = ~core
    if int(cont_mask.sum()) < degree + 2:
        cont_mask = np.ones_like(wl, dtype=bool)

    wl_c, flux_c = wl[cont_mask], flux[cont_mask]
    if degree >= 1 and wl_c.size >= 2:
        c1, c0_at0 = np.polyfit(wl_c - lambda0, flux_c, 1)
        c0 = float(c0_at0)
        c1 = float(c1)
    else:
        c0 = float(np.median(flux_c))
        c1 = 0.0

    noise = _robust_std(flux_c - (c0 + c1 * (wl_c - lambda0)))
    c0_scale = max(5.0 * noise, abs(c0) + 1e-30)
    span = float(wl.max() - wl.min()) or 1.0
    c1_scale = max(5.0 * noise / span, 1e-30)

    # Flux scale: window peak excess turned into an integrated-flux magnitude.
    cont = c0 + c1 * (wl - lambda0)
    peak = float(np.max(np.abs(flux - cont))) if flux.size else noise
    fwhm_typ = float(np.median([c.template.fwhm_kms_init for c in components]))
    sigma_w_typ = lambda0 * (fwhm_typ / C_KMS) * _FWHM_TO_SIGMA
    flux_scale = max(3.0 * peak * sigma_w_typ * _SQRT_2PI, 1e-30)

    return InitialEstimates(
        c0=c0,
        c1=c1,
        c0_scale=c0_scale,
        c1_scale=c1_scale,
        lambda0=lambda0,
        flux_scale=flux_scale,
    )
