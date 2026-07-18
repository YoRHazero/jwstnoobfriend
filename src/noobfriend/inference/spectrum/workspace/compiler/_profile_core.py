"""Backend-agnostic line-profile math shared by the numeric and symbolic paths.

A single emission or absorption line contributes ``sign * flux *
profile(wavelength)`` to a spectrum, where ``profile`` is the unit-integral
*shape* built here. The very same formulas must be evaluated in two settings:
as concrete NumPy arrays when a model is reconstructed from known parameter
values (prior scaling, MLE, posterior plots, cross-validation) and as a
differentiable PyTensor graph inside the PyMC likelihood. To keep those two
paths provably identical, the math lives here once and is parameterized by a
small bundle of elementary-function primitives (:class:`ProfileOps`); the NumPy
and PyTensor callers differ only in which primitives they inject.

This module imports neither ``numpy`` nor ``pytensor``: the primitives are
passed in, so it stays a dependency-free leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Any

#: Speed of light in km/s, the velocity unit every line width speaks.
C_KMS = 299792.458
_GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))
_LAPLACE_FWHM_TO_SCALE = 2.0 * log(2.0)
_SQRT_TWO_PI = sqrt(2.0 * pi)
_SQRT_TWO = sqrt(2.0)

#: One branch of the gaussian mixture: ``(weight, gaussian_fwhm, laplace_fwhm)``
#: with a ``None`` laplace width for a pure-gaussian branch.
type ProfileBranch = tuple[Any, Any, Any | None]


@dataclass(frozen=True, slots=True)
class ProfileOps:
    """Elementary-function backend for evaluating a line profile.

    Bundling the primitives lets one profile implementation serve both the
    NumPy reconstruction path (``numpy`` ufuncs and ``scipy.special``) and the
    PyTensor sampling graph (``pytensor.tensor`` ops).

    Attributes
    ----------
    exp, absolute, sqrt
        Elementwise exponential, absolute value, and square root.
    erfcx, erfc
        Scaled and plain complementary error functions; both are needed to
        keep the exponentially-modified-gaussian wings finite.
    maximum, where
        Elementwise ``maximum(x, y)`` and ternary ``where(cond, a, b)`` used to
        clamp transcendental arguments in the far wings.
    prune_zero_weight
        Whether to drop a kernel branch whose weight is identically zero. Set
        only for the numeric backend, where kernel fractions are concrete
        floats and a fully-convolved branch (``fraction == 1``) can drop its
        zero-weight complement; the symbolic backend keeps every branch because
        a fraction may be a random variable that cannot be compared at build
        time.
    """

    exp: Callable[[Any], Any]
    absolute: Callable[[Any], Any]
    erfcx: Callable[[Any], Any]
    erfc: Callable[[Any], Any]
    sqrt: Callable[[Any], Any]
    maximum: Callable[[Any, Any], Any]
    where: Callable[[Any, Any, Any], Any]
    prune_zero_weight: bool


def evaluate_profile(
    ops: ProfileOps,
    *,
    profile: str,
    wavelength: Any,
    center: Any,
    fwhm_kms: Any,
    kernels: Sequence[tuple[str, Any, Any]] = (),
) -> Any:
    """Return the unit-integral line profile on ``wavelength``.

    Parameters
    ----------
    ops
        The elementary-function backend (NumPy or PyTensor).
    profile
        Base profile family: ``"gaussian"``, ``"lorentzian"``, or
        ``"exponential"``.
    wavelength
        Observed-frame wavelengths to evaluate on.
    center
        Line centre in the wavelength units of ``wavelength``.
    fwhm_kms
        Base-profile FWHM in km/s. Any instrumental broadening is folded into
        this value by the caller before evaluation.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels applied on top of a
        gaussian base. Each kernel convolves ``fraction`` of the profile and
        leaves the rest untouched (a branch mixture); a gaussian kernel folds
        into the base by quadrature, while a single laplace kernel yields the
        exact two-sided exponentially-modified-gaussian closed form per branch.

    Returns
    -------
    Any
        Profile values integrating to one over wavelength, in the array or
        tensor type produced by ``ops``.

    Raises
    ------
    NotImplementedError
        If kernels accompany a non-gaussian base, or more than one laplace
        kernel is supplied.
    ValueError
        If ``profile`` is not a supported family.
    """
    if kernels:
        if profile != "gaussian":
            raise NotImplementedError(
                "convolution kernels currently require a gaussian base profile."
            )
        if sum(1 for kind, _, _ in kernels if kind == "laplace") > 1:
            raise NotImplementedError(
                "at most one laplace convolution kernel is supported."
            )

    if profile == "gaussian":
        return _gaussian_mixture(ops, wavelength, center, fwhm_kms, kernels)
    fwhm_wavelength = center * fwhm_kms / C_KMS
    if profile == "lorentzian":
        gamma = 0.5 * fwhm_wavelength
        return (gamma / pi) / ((wavelength - center) ** 2 + gamma**2)
    if profile == "exponential":
        scale = fwhm_wavelength / _LAPLACE_FWHM_TO_SCALE
        return ops.exp(-ops.absolute(wavelength - center) / scale) / (2.0 * scale)
    raise ValueError(f"Unsupported profile: {profile!r}.")


def _gaussian_mixture(
    ops: ProfileOps,
    wavelength: Any,
    center: Any,
    fwhm_kms: Any,
    kernels: Sequence[tuple[str, Any, Any]],
) -> Any:
    """Sum the gaussian/EMG branches of a (possibly convolved) gaussian line."""
    delta = wavelength - center
    total: Any = None
    for weight, gauss_fwhm, laplace_fwhm in _compose_branches(ops, fwhm_kms, kernels):
        sigma = center * gauss_fwhm / C_KMS / _GAUSSIAN_FWHM_TO_SIGMA
        z = delta / sigma
        gaussian = ops.exp(-0.5 * z**2)
        if laplace_fwhm is None:
            term = weight * gaussian / (sigma * _SQRT_TWO_PI)
        else:
            scale = center * laplace_fwhm / C_KMS / _LAPLACE_FWHM_TO_SCALE
            a = sigma / scale
            emg = _gaussian_erfcx(ops, gaussian, z, (a + z) / _SQRT_TWO)
            emg = emg + _gaussian_erfcx(ops, gaussian, z, (a - z) / _SQRT_TWO)
            term = weight * emg / (4.0 * scale)
        total = term if total is None else total + term
    return total


def _gaussian_erfcx(ops: ProfileOps, gaussian: Any, z: Any, u: Any) -> Any:
    r"""Return ``exp(-z**2 / 2) * erfcx(u)`` without overflow on either wing.

    The two-sided exponentially-modified-gaussian multiplies a vanishing
    gaussian by a diverging ``erfcx`` in the far wings, so evaluating the two
    factors separately hits ``0 * inf = NaN`` there. For ``u >= 0`` ``erfcx``
    is bounded and the product is taken directly; for ``u < 0`` the gaussian is
    folded into the plain error function via ``erfcx(u) = exp(u**2) erfc(u)``,
    whose combined exponent ``u**2 - z**2 / 2`` is non-positive on that wing.
    Each branch clamps the argument of the transcendental it does not select,
    so both evaluate finite before :func:`where` picks one.
    """
    u_positive = ops.maximum(u, 0.0)
    u_negative = u - u_positive  # elementwise min(u, 0)
    return ops.where(
        u >= 0.0,
        gaussian * ops.erfcx(u_positive),
        ops.exp(u_negative**2 - 0.5 * z**2) * ops.erfc(u_negative),
    )


def _compose_branches(
    ops: ProfileOps, fwhm_kms: Any, kernels: Sequence[tuple[str, Any, Any]]
) -> list[ProfileBranch]:
    """Expand convolution kernels into a weighted mixture of profile branches.

    Starting from the bare gaussian, each kernel splits every branch into an
    untouched ``(1 - fraction)`` part and a convolved ``fraction`` part: a
    gaussian kernel adds its width in quadrature, a laplace kernel tags the
    branch for the exponentially-modified-gaussian closed form.
    """
    branches: list[ProfileBranch] = [(1.0, fwhm_kms, None)]
    for kind, width, fraction in kernels:
        updated: list[ProfileBranch] = []
        for weight, gauss_fwhm, laplace_fwhm in branches:
            if not (ops.prune_zero_weight and fraction >= 1.0):
                updated.append((weight * (1.0 - fraction), gauss_fwhm, laplace_fwhm))
            if kind == "gaussian":
                updated.append(
                    (
                        weight * fraction,
                        ops.sqrt(gauss_fwhm**2 + width**2),
                        laplace_fwhm,
                    )
                )
            elif kind == "laplace":
                updated.append((weight * fraction, gauss_fwhm, width))
            else:
                raise ValueError(f"Unsupported kernel kind: {kind!r}.")
        branches = updated
    return branches
