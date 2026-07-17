"""Numerical profile templates for compiled workspace models."""

from __future__ import annotations

from collections.abc import Sequence
from math import log, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfcx

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.handles import LineHandle


C_KMS = 299792.458
_GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))
_LAPLACE_FWHM_TO_SCALE = 2.0 * log(2.0)


def profile_template(
    wavelength: np.ndarray,
    *,
    handle: LineHandle,
    center: float,
    fwhm_kms: float,
    resolving_power: float | None,
    kernels: Sequence[tuple[str, float, float]] = (),
) -> np.ndarray:
    """Return a unit-integral line profile on ``wavelength``.

    Parameters
    ----------
    wavelength
        Observed-frame wavelengths to evaluate on.
    handle
        The line handle providing the base profile family.
    center
        Line centre in the wavelength units of ``wavelength``.
    fwhm_kms
        Base-profile FWHM in km/s (intrinsic; the instrumental width is
        folded in here when ``resolving_power`` is given).
    resolving_power
        Spectral resolving power; folds a Gaussian LSF into a gaussian base
        by quadrature. Only implemented for gaussian base profiles.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels. Each kernel
        convolves ``fraction`` of the profile and leaves the rest untouched
        (branch mixture). Gaussian kernels fold into a gaussian base by
        quadrature; one Laplace kernel yields the exact two-sided
        exponentially-modified-Gaussian closed form per branch.

    Returns
    -------
    numpy.ndarray
        Read-only profile values integrating to one over wavelength.
    """
    if resolving_power is not None:
        if handle.profile != "gaussian":
            raise NotImplementedError(
                "resolving_power is only implemented for gaussian line profiles."
            )
        fwhm_kms = sqrt(fwhm_kms**2 + (C_KMS / resolving_power) ** 2)

    if kernels:
        if handle.profile != "gaussian":
            raise NotImplementedError(
                "convolution kernels currently require a gaussian base profile."
            )
        if sum(1 for kind, _, _ in kernels if kind == "laplace") > 1:
            raise NotImplementedError(
                "at most one laplace convolution kernel is supported."
            )

    delta = wavelength - center
    if handle.profile == "gaussian":
        branches: list[tuple[float, float, float | None]] = [(1.0, fwhm_kms, None)]
        for kind, width, fraction in kernels:
            updated: list[tuple[float, float, float | None]] = []
            for weight, gauss_fwhm, laplace_fwhm in branches:
                if fraction < 1.0:
                    updated.append(
                        (weight * (1.0 - fraction), gauss_fwhm, laplace_fwhm)
                    )
                if kind == "gaussian":
                    updated.append(
                        (
                            weight * fraction,
                            sqrt(gauss_fwhm**2 + width**2),
                            laplace_fwhm,
                        )
                    )
                elif kind == "laplace":
                    updated.append((weight * fraction, gauss_fwhm, width))
                else:
                    raise ValueError(f"Unsupported kernel kind: {kind!r}.")
            branches = updated
        profile = np.zeros_like(wavelength, dtype=float)
        for weight, gauss_fwhm, laplace_fwhm in branches:
            sigma = center * gauss_fwhm / C_KMS / _GAUSSIAN_FWHM_TO_SIGMA
            if laplace_fwhm is None:
                profile = profile + weight * np.exp(-0.5 * (delta / sigma) ** 2) / (
                    sigma * sqrt(2.0 * pi)
                )
            else:
                scale = center * laplace_fwhm / C_KMS / _LAPLACE_FWHM_TO_SCALE
                u_plus = (sigma / scale + delta / sigma) / sqrt(2.0)
                u_minus = (sigma / scale - delta / sigma) / sqrt(2.0)
                profile = profile + weight * (
                    np.exp(-0.5 * (delta / sigma) ** 2)
                    / (4.0 * scale)
                    * (erfcx(u_plus) + erfcx(u_minus))
                )
    elif handle.profile == "lorentzian":
        fwhm_wavelength = center * fwhm_kms / C_KMS
        gamma = 0.5 * fwhm_wavelength
        profile = (gamma / pi) / (delta**2 + gamma**2)
    elif handle.profile == "exponential":
        fwhm_wavelength = center * fwhm_kms / C_KMS
        scale = fwhm_wavelength / _LAPLACE_FWHM_TO_SCALE
        profile = np.exp(-np.abs(delta) / scale) / (2.0 * scale)
    else:
        raise ValueError(f"Unsupported profile: {handle.profile!r}.")
    profile.setflags(write=False)
    return profile
