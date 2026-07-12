"""Numerical profile templates for compiled workspace models."""

from __future__ import annotations

from math import log, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.handles import LineHandle


C_KMS = 299792.458
_GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))


def profile_template(
    wavelength: np.ndarray,
    *,
    handle: LineHandle,
    center: float,
    fwhm_kms: float,
    resolving_power: float | None,
) -> np.ndarray:
    """Return a unit-integral line profile on ``wavelength``."""
    if resolving_power is not None:
        if handle.profile != "gaussian":
            raise NotImplementedError("resolving_power is only implemented for gaussian line profiles.")
        fwhm_kms = sqrt(fwhm_kms**2 + (C_KMS / resolving_power) ** 2)

    fwhm_wavelength = center * fwhm_kms / C_KMS
    if handle.profile == "gaussian":
        sigma = fwhm_wavelength / _GAUSSIAN_FWHM_TO_SIGMA
        profile = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2) / (sigma * sqrt(2.0 * pi))
    elif handle.profile == "lorentzian":
        gamma = 0.5 * fwhm_wavelength
        profile = (gamma / pi) / ((wavelength - center) ** 2 + gamma**2)
    elif handle.profile == "exponential":
        scale = fwhm_wavelength / (2.0 * log(2.0))
        profile = np.exp(-np.abs(wavelength - center) / scale) / (2.0 * scale)
    else:
        raise ValueError(f"Unsupported profile: {handle.profile!r}.")
    profile.setflags(write=False)
    return profile
