"""Shared synthetic-spectrum helpers for spectrum inference tests."""

from __future__ import annotations

from math import log, pi, sqrt

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import C_KMS


def gaussian(
    wavelength: np.ndarray,
    *,
    center: float,
    flux: float,
    fwhm_kms: float,
    delta_v_kms: float = 0.0,
    resolving_power: float | None = None,
) -> np.ndarray:
    """Return an integrated-flux Gaussian line on a wavelength grid."""
    shifted_center = center * (1.0 + delta_v_kms / C_KMS)
    effective_fwhm = fwhm_kms
    if resolving_power is not None:
        effective_fwhm = sqrt(fwhm_kms**2 + (C_KMS / resolving_power) ** 2)
    fwhm_wavelength = shifted_center * effective_fwhm / C_KMS
    sigma = fwhm_wavelength / (2.0 * sqrt(2.0 * log(2.0)))
    template = np.exp(-0.5 * ((wavelength - shifted_center) / sigma) ** 2) / (
        sigma * sqrt(2.0 * pi)
    )
    return flux * template
