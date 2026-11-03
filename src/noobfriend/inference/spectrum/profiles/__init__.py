"""Closed-form line profiles and convolution kernels.

One implementation per formula, evaluated with NumPy or JAX through the
``xp`` array namespace: base profiles (gaussian, lorentzian, exponential),
gaussian / laplace convolution kernels with fractions, and the exact fold of a
gaussian line-spread function into each base family.
"""

from noobfriend.inference.spectrum.profiles._profiles import (
    GAUSSIAN_FWHM_TO_SIGMA,
    KERNELS,
    LAPLACE_FWHM_TO_SCALE,
    PROFILES,
    Kernel,
    check_kernels,
    effective_fwhm,
    line_profile,
    voigt,
)

__all__ = [
    "GAUSSIAN_FWHM_TO_SIGMA",
    "KERNELS",
    "LAPLACE_FWHM_TO_SCALE",
    "PROFILES",
    "Kernel",
    "check_kernels",
    "effective_fwhm",
    "line_profile",
    "voigt",
]
