"""Synthetic spectra with known truth for the spectrum tests.

Resolution defaults to R = 1600 (FRESCO grism), so line widths and LSF
scales match the real use case.
"""

from __future__ import annotations

import numpy as np

from noobfriend.inference.spectrum._wavelength import C_KMS
from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.data import NoobSpectrum
from noobfriend.inference.spectrum.profiles import line_profile

Z = 2.0
HALPHA = 6564.6
RESOLVING_POWER = 1600.0


def gaussian_line(
    obs: np.ndarray,
    *,
    center: float,
    fwhm_kms: float,
    flux: float,
    resolving_power: float | None = RESOLVING_POWER,
) -> np.ndarray:
    """Signed gaussian line observed through a gaussian LSF."""
    lsf = None if resolving_power is None else C_KMS / resolving_power
    profile = line_profile(
        numpy_namespace(),
        obs,
        profile="gaussian",
        center=center,
        fwhm_kms=fwhm_kms,
        lsf_fwhm_kms=lsf,
    )
    return flux * profile


def halpha_spectrum(
    *,
    seed: int = 0,
    flux: float = 40.0,
    fwhm_kms: float = 300.0,
    delta_v_kms: float = 0.0,
    continuum: float = 1.0,
    noise: float = 0.05,
    n_pixels: int = 300,
    resolving_power: float | None = RESOLVING_POWER,
) -> NoobSpectrum:
    """One frame with a single narrow Halpha line on a flat continuum."""
    rng = np.random.default_rng(seed)
    center_rest = HALPHA * (1.0 + Z)
    obs = np.linspace(center_rest - 150.0, center_rest + 150.0, n_pixels)
    center = center_rest * (1.0 + delta_v_kms / C_KMS)
    model = continuum + gaussian_line(
        obs,
        center=center,
        fwhm_kms=fwhm_kms,
        flux=flux,
        resolving_power=resolving_power,
    )
    error = np.full(obs.size, noise)
    return NoobSpectrum(
        obs,
        model + noise * rng.normal(size=obs.size),
        error,
        unit="angstrom",
        resolving_power=resolving_power,
        z=Z,
    )
