"""Symbolic (PyTensor) profile and expression evaluation for PyMC models.

The NumPy reconstruction path in :mod:`.profiles` and this module are two faces
of the same forward model: identical formulas, evaluated as concrete arrays
there and as a differentiable PyTensor graph here so NUTS can sample the
likelihood. Both delegate the profile math to :mod:`._profile_core`; only the
injected primitives differ. Neither function imports ``pytensor`` — the tensor
namespace is passed in by the model builder, which keeps the package importable
without the optional ``mcmc`` dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from noobfriend.inference.spectrum.workspace.compiler._faddeeva import (
    symbolic_voigt,
)
from noobfriend.inference.spectrum.workspace.compiler._profile_core import (
    ProfileOps,
    evaluate_profile,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.compiler.expressions import (
        FluxExpression,
        ParameterExpression,
    )


def symbolic_profile(
    pt: Any,
    *,
    profile: str,
    wavelength: Any,
    center: Any,
    fwhm_kms: Any,
    kernels: Sequence[tuple[str, Any, Any]] = (),
    instrumental_fwhm_kms: Any | None = None,
) -> Any:
    """Build the PyTensor graph of a line's unit-integral profile.

    The symbolic counterpart of ``profile_template``: the same math evaluated
    as a differentiable tensor.

    Parameters
    ----------
    pt
        The ``pytensor.tensor`` namespace supplying ``exp``, ``abs``,
        ``erfcx``, and ``sqrt``.
    profile
        Base profile family.
    wavelength
        Observed-frame wavelengths (a concrete array) to evaluate on.
    center, fwhm_kms
        Line centre and intrinsic FWHM as symbolic tensors.
    kernels
        ``(kind, fwhm_kms, fraction)`` convolution kernels as symbolic tensors.
    instrumental_fwhm_kms
        Gaussian instrumental line-spread FWHM in km/s (``C_KMS /
        resolving_power``), or ``None``. Folded exactly into the base profile
        by the shared core, as in ``profile_template``.

    Returns
    -------
    Any
        A PyTensor expression for the unit-integral profile.
    """
    ops = ProfileOps(
        exp=pt.exp,
        absolute=pt.abs,
        erfcx=pt.erfcx,
        erfc=pt.erfc,
        sqrt=pt.sqrt,
        maximum=pt.maximum,
        where=pt.where,
        voigt=partial(symbolic_voigt, pt),
        prune_zero_weight=False,
    )
    return evaluate_profile(
        ops,
        profile=profile,
        wavelength=wavelength,
        center=center,
        fwhm_kms=fwhm_kms,
        kernels=kernels,
        instrumental_fwhm_kms=instrumental_fwhm_kms,
    )


def symbolic_expression(
    pt: Any,
    expression: FluxExpression | ParameterExpression,
    values: Mapping[int, Any],
) -> Any:
    """Evaluate a compiled linear expression as a PyTensor graph.

    The symbolic counterpart of ``evaluate_expression``: it sums the fixed term
    and the source contributions ``coefficient * values[source_id]`` into one
    tensor.

    Parameters
    ----------
    pt
        The ``pytensor.tensor`` namespace.
    expression
        A compiled flux or parameter expression.
    values
        Symbolic source variables keyed by line identity.

    Returns
    -------
    Any
        A PyTensor expression for the compiled linear combination.
    """
    output = pt.as_tensor_variable(float(expression.fixed))
    for source_id, coefficient in expression.terms.items():
        output = output + float(coefficient) * values[source_id]
    return output
