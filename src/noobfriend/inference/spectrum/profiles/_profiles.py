"""Closed-form unit-area line profiles, written once against ``xp``.

A line contributes ``sign * flux * profile(wavelength)``, where ``profile`` is
the unit-integral shape built here. Widths are FWHMs in km/s; a centre in
wavelength units turns them into wavelength widths. Every convolution has an
exact closed form:

- a gaussian kernel or a gaussian LSF on a gaussian base adds in quadrature;
- one laplace kernel on a gaussian base gives the two-sided
  exponentially-modified gaussian (Normal--Laplace) via ``erfcx``;
- an exponential base under a gaussian LSF is the same Normal--Laplace form;
- a lorentzian base under a gaussian LSF is a Voigt profile via ``wofz``.

A kernel with fraction ``f`` splits every branch into ``(1 - f) * branch +
f * (branch * kernel)``, so ``n`` kernels expand to ``2**n`` branches.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log, pi, sqrt
from typing import Any

from noobfriend.inference.spectrum._wavelength import C_KMS

#: Base profile families.
PROFILES: tuple[str, ...] = ("gaussian", "lorentzian", "exponential")
#: Convolution kernel kinds.
KERNELS: tuple[str, ...] = ("gaussian", "laplace")

GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))
LAPLACE_FWHM_TO_SCALE = 2.0 * log(2.0)
_SQRT_TWO_PI = sqrt(2.0 * pi)
_SQRT_TWO = sqrt(2.0)

#: ``(kind, fwhm_kms, fraction)`` for one kernel.
type Kernel = tuple[str, Any, Any]
#: ``(weight, gaussian_fwhm, laplace_fwhm or None)`` for one mixture branch.
type _Branch = tuple[Any, Any, Any | None]


def check_kernels(profile: str, kinds: Sequence[str]) -> None:
    """Reject kernel combinations without a closed form.

    Parameters
    ----------
    profile
        Base profile family.
    kinds
        Kernel kinds attached to the line.

    Raises
    ------
    ValueError
        If a name is unknown or a kind repeats.
    NotImplementedError
        If kernels sit on a non-gaussian base or more than one kernel is
        non-gaussian (these need numerical convolution, not implemented).
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}; got {profile!r}.")
    for kind in kinds:
        if kind not in KERNELS:
            raise ValueError(f"kernel kind must be one of {KERNELS}; got {kind!r}.")
    if len(set(kinds)) != len(kinds):
        raise ValueError("each kernel kind may appear once per line.")
    if kinds and profile != "gaussian":
        raise NotImplementedError(
            "convolution kernels currently need a gaussian base profile."
        )
    if sum(kind != "gaussian" for kind in kinds) > 1:
        raise NotImplementedError(
            "at most one non-gaussian kernel is supported per line."
        )


def line_profile(
    xp: Any,
    wavelength: Any,
    *,
    profile: str,
    center: Any,
    fwhm_kms: Any,
    kernels: Sequence[Kernel] = (),
    lsf_fwhm_kms: float | None = None,
) -> Any:
    """Return the unit-area profile of one line on ``wavelength``.

    Parameters
    ----------
    xp
        Array namespace.
    wavelength
        Observed-frame wavelengths.
    profile
        ``"gaussian"``, ``"lorentzian"`` or ``"exponential"``.
    center
        Line centre, same unit as ``wavelength``.
    fwhm_kms
        Intrinsic base-profile FWHM in km/s.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels (gaussian base only).
    lsf_fwhm_kms
        FWHM of the gaussian line-spread function in km/s, or ``None``.

    Returns
    -------
    array
        Profile values; arrays broadcast, so a leading draw axis on the
        parameters yields one profile per draw.
    """
    if kernels and profile != "gaussian":
        raise NotImplementedError(
            "convolution kernels currently need a gaussian base profile."
        )
    if lsf_fwhm_kms is not None:
        if profile == "gaussian":
            fwhm_kms = xp.sqrt(fwhm_kms**2 + lsf_fwhm_kms**2)
        elif profile == "exponential":
            return _gaussian_mixture(
                xp, wavelength, center, lsf_fwhm_kms, (("laplace", fwhm_kms, 1.0),)
            )
        elif profile == "lorentzian":
            sigma = center * lsf_fwhm_kms / C_KMS / GAUSSIAN_FWHM_TO_SIGMA
            gamma = 0.5 * center * fwhm_kms / C_KMS
            return voigt(xp, wavelength - center, sigma, gamma)
    if profile == "gaussian":
        return _gaussian_mixture(xp, wavelength, center, fwhm_kms, kernels)
    width = center * fwhm_kms / C_KMS
    if profile == "lorentzian":
        gamma = 0.5 * width
        return (gamma / pi) / ((wavelength - center) ** 2 + gamma**2)
    if profile == "exponential":
        scale = width / LAPLACE_FWHM_TO_SCALE
        return xp.exp(-xp.abs(wavelength - center) / scale) / (2.0 * scale)
    raise ValueError(f"profile must be one of {PROFILES}; got {profile!r}.")


def voigt(xp: Any, delta: Any, sigma: Any, gamma: Any) -> Any:
    """Unit-area Voigt profile via the Faddeeva function.

    Parameters
    ----------
    xp
        Array namespace.
    delta
        Offsets from the line centre.
    sigma
        Gaussian standard deviation.
    gamma
        Lorentzian half width at half maximum.

    Returns
    -------
    array
        ``Re[w((delta + i gamma) / (sigma sqrt 2))] / (sigma sqrt(2 pi))``.
    """
    z = (delta + 1j * gamma) / (sigma * _SQRT_TWO)
    return xp.real(xp.special.wofz(z)) / (sigma * _SQRT_TWO_PI)


def effective_fwhm(fwhm_kms: Any, lsf_fwhm_kms: float) -> Any:
    """Quadrature sum of an intrinsic width and the LSF width (km/s)."""
    return (fwhm_kms**2 + lsf_fwhm_kms**2) ** 0.5


def _gaussian_mixture(
    xp: Any, wavelength: Any, center: Any, fwhm_kms: Any, kernels: Sequence[Kernel]
) -> Any:
    """Sum the gaussian / Normal--Laplace branches of a convolved gaussian."""
    delta = wavelength - center
    total: Any = None
    for weight, gauss_fwhm, laplace_fwhm in _branches(xp, fwhm_kms, kernels):
        sigma = center * gauss_fwhm / C_KMS / GAUSSIAN_FWHM_TO_SIGMA
        z = delta / sigma
        gaussian = xp.exp(-0.5 * z**2)
        if laplace_fwhm is None:
            term = weight * gaussian / (sigma * _SQRT_TWO_PI)
        else:
            scale = center * laplace_fwhm / C_KMS / LAPLACE_FWHM_TO_SCALE
            a = sigma / scale
            emg = _gaussian_erfcx(xp, gaussian, z, (a + z) / _SQRT_TWO)
            emg = emg + _gaussian_erfcx(xp, gaussian, z, (a - z) / _SQRT_TWO)
            term = weight * emg / (4.0 * scale)
        total = term if total is None else total + term
    return total


def _gaussian_erfcx(xp: Any, gaussian: Any, z: Any, u: Any) -> Any:
    """Return ``exp(-z**2 / 2) * erfcx(u)`` without overflow on either wing.

    For ``u >= 0`` ``erfcx`` is bounded and the product is direct; for
    ``u < 0`` the gaussian folds into ``erfc`` through
    ``erfcx(u) = exp(u**2) erfc(u)``, whose combined exponent is non-positive
    there. Each side clamps the argument it does not select so both stay
    finite (and differentiable) before ``where`` picks one.
    """
    u_positive = xp.maximum(u, 0.0)
    u_negative = u - u_positive
    return xp.where(
        u >= 0.0,
        gaussian * xp.special.erfcx(u_positive),
        xp.exp(u_negative**2 - 0.5 * z**2) * xp.special.erfc(u_negative),
    )


def _branches(xp: Any, fwhm_kms: Any, kernels: Sequence[Kernel]) -> list[_Branch]:
    """Expand kernels into weighted mixture branches.

    A branch with weight known to be exactly zero (a plain-float fraction of
    ``1``) is dropped; array fractions keep every branch.
    """
    branches: list[_Branch] = [(1.0, fwhm_kms, None)]
    for kind, width, fraction in kernels:
        fully_convolved = isinstance(fraction, (int, float)) and fraction >= 1.0
        updated: list[_Branch] = []
        for weight, gauss_fwhm, laplace_fwhm in branches:
            if not fully_convolved:
                updated.append((weight * (1.0 - fraction), gauss_fwhm, laplace_fwhm))
            if kind == "gaussian":
                widened = xp.sqrt(gauss_fwhm**2 + width**2)
                updated.append((weight * fraction, widened, laplace_fwhm))
            elif kind == "laplace":
                updated.append((weight * fraction, gauss_fwhm, width))
            else:
                raise ValueError(f"kernel kind must be one of {KERNELS}; got {kind!r}.")
        branches = updated
    return branches
