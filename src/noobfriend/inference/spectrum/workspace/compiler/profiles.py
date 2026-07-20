"""Numerical line-profile templates for compiled workspace models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfc, erfcx, voigt_profile

from noobfriend.inference.spectrum.workspace.compiler._profile_core import (
    C_KMS,
    ProfileOps,
    evaluate_profile,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.handles import LineHandle

__all__ = ["C_KMS", "profile_template", "profile_template_stack"]

_NUMPY_OPS = ProfileOps(
    exp=np.exp,
    absolute=np.abs,
    erfcx=erfcx,
    erfc=erfc,
    sqrt=sqrt,
    maximum=np.maximum,
    where=np.where,
    voigt=voigt_profile,
    prune_zero_weight=True,
)
# The batched path carries a draw axis, so kernel widths/fractions are arrays:
# ``sqrt`` must be the array ufunc (a gaussian kernel quadrature-adds widths),
# and fractions cannot be compared against 1.0 to prune a branch (keeping every
# branch is numerically identical, since a pruned branch has weight zero).
_NUMPY_STACK_OPS = replace(_NUMPY_OPS, sqrt=np.sqrt, prune_zero_weight=False)


def profile_template(
    wavelength: np.ndarray,
    *,
    handle: LineHandle,
    center: float,
    fwhm_kms: float,
    resolving_power: float | None,
    kernels: Sequence[tuple[str, float, float]] = (),
) -> np.ndarray:
    """Evaluate one line's unit-area *shape* on a wavelength grid (NumPy).

    This is the numeric forward-model building block. Given where a line sits
    and how wide it is, it returns the line's shape as an array that integrates
    to one over ``wavelength``. Callers multiply that shape by the line's
    integrated flux and its emission/absorption sign to obtain the line's
    contribution to the spectrum::

        line_flux_density = sign * flux * profile_template(...)

    so flux and sign are deliberately **not** applied here — flux is a separate
    fitted axis with its own priors and ties. Every consumer that reconstructs
    a model from known parameter values calls this function (flux-prior
    scaling, MLE evaluation, posterior plots, per-draw reconstruction, frame
    cross-validation). The PyMC sampling graph builds the identical math
    symbolically through ``symbolic_profile``; both share
    :func:`~noobfriend.inference.spectrum.workspace.compiler._profile_core.evaluate_profile`
    so the two paths cannot drift.

    Parameters
    ----------
    wavelength
        Observed-frame wavelengths to evaluate on.
    handle
        The line handle providing the base profile family.
    center
        Line centre in the wavelength units of ``wavelength``.
    fwhm_kms
        Intrinsic base-profile FWHM in km/s.
    resolving_power
        Spectral resolving power; folds a Gaussian LSF exactly into the base
        profile — by quadrature for a gaussian base, via the closed-form
        Normal–Laplace (Laplace ⊛ Gaussian) for an exponential base, and via
        the Voigt profile (Faddeeva function) for a lorentzian base.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels; ``kind`` is
        ``"gaussian"`` or ``"laplace"`` and each width is in km/s. Gaussian
        kernels fold into the base by quadrature; one Laplace kernel yields the
        exact two-sided exponentially-modified-Gaussian closed form per branch.

    Returns
    -------
    numpy.ndarray
        Read-only profile values integrating to one over wavelength.

    Raises
    ------
    NotImplementedError
        If kernels accompany a non-gaussian base, or more than one laplace
        kernel is supplied.
    """
    profile = np.asarray(
        evaluate_profile(
            _NUMPY_OPS,
            profile=handle.profile,
            wavelength=np.asarray(wavelength, dtype=float),
            center=center,
            fwhm_kms=fwhm_kms,
            kernels=kernels,
            instrumental_fwhm_kms=(
                None if resolving_power is None else C_KMS / resolving_power
            ),
        ),
        dtype=float,
    )
    profile.setflags(write=False)
    return profile


def profile_template_stack(
    wavelength: np.ndarray,
    *,
    handle: LineHandle,
    centers: np.ndarray,
    fwhms_kms: np.ndarray,
    resolving_power: float | None,
    kernels: Sequence[tuple[str, np.ndarray, np.ndarray]] = (),
) -> np.ndarray:
    """Evaluate one line's profile for a whole stack of parameter draws.

    The batched counterpart of :func:`profile_template`. ``centers``,
    ``fwhms_kms``, and each kernel's width/fraction carry a leading draw axis of
    length ``S``; the result is an ``(S, wavelength.size)`` array of unit-area
    profiles, one per draw. Reconstructing a posterior band or a frame
    cross-validation model this way replaces a Python loop over draws with one
    vectorized evaluation.

    Parameters
    ----------
    wavelength
        Observed-frame wavelengths to evaluate on, shape ``(W,)``.
    handle
        The line handle providing the base profile family.
    centers, fwhms_kms
        Per-draw line centres and intrinsic FWHMs (km/s), each shape ``(S,)``.
    resolving_power
        Spectral resolving power folded exactly into every draw's base
        profile, as in :func:`profile_template`.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels whose widths and
        fractions are per-draw arrays of shape ``(S,)``.

    Returns
    -------
    numpy.ndarray
        Profiles of shape ``(S, wavelength.size)``.

    Raises
    ------
    NotImplementedError
        If kernels accompany a non-gaussian base, or more than one laplace
        kernel is supplied.
    """
    centers = np.asarray(centers, dtype=float)
    fwhms = np.asarray(fwhms_kms, dtype=float)
    profile = evaluate_profile(
        _NUMPY_STACK_OPS,
        profile=handle.profile,
        wavelength=np.asarray(wavelength, dtype=float),
        center=centers[:, None],
        fwhm_kms=fwhms[:, None],
        kernels=tuple(
            (
                kind,
                np.asarray(width, dtype=float)[:, None],
                np.asarray(fraction, dtype=float)[:, None],
            )
            for kind, width, fraction in kernels
        ),
        instrumental_fwhm_kms=(
            None if resolving_power is None else C_KMS / resolving_power
        ),
    )
    return np.asarray(profile, dtype=float)
