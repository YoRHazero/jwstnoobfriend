"""Compile line rules into reusable parameter expressions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from math import inf, isfinite, sqrt
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.line import (
    NOMINAL_FWHM_KMS,
    LogUniform,
    Normal,
    Prior,
    TruncatedNormal,
    Uniform,
    prior_support,
)
from noobfriend.inference.spectrum.workspace.compiler.profiles import C_KMS

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.line._rules import _ParameterRule
    from noobfriend.inference.spectrum.workspace.handles import LineHandle


#: Half-width, in standard deviations, of the finite range an unbounded
#: ``Normal`` prior collapses to. The optimizer needs a box and the ridge
#: search needs a start, but neither can represent an unbounded prior; three
#: sigmas keep 99.7% of the declared mass inside the box, so a fit pinned at
#: its edge is a genuine tension with the prior rather than an artifact of a
#: tight conversion.
NORMAL_RANGE_SIGMAS = 3.0

#: Name of the base-profile width every line carries. Additional shape
#: parameters (e.g. convolution-kernel widths) are keyed by their own names.
BASE_SHAPE = "fwhm"


@dataclass(frozen=True, slots=True)
class FluxExpression:
    """Linear expression for one line flux from source flux parameters."""

    fixed: float
    terms: dict[int, float]


@dataclass(frozen=True, slots=True)
class ParameterExpression:
    """Linear expression for one line scalar parameter from source parameters."""

    fixed: float
    terms: dict[int, float]
    specs: dict[int, VariableSpec]


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Bounded scalar variable specification used by fit backends.

    ``lower``/``upper``/``initial`` are all a backend needs to optimize the
    variable, and the MLE backend uses nothing else. ``prior`` carries a
    declared distribution when the user supplied one; ``None`` leaves the
    choice of prior to the sampling backend, which is the original behavior
    (uniform in the variable for a velocity offset, uniform in its logarithm
    for a width).
    """

    lower: float
    upper: float
    initial: float
    prior: Prior | None = None


@dataclass(frozen=True, slots=True)
class ParameterOffsets:
    """Offsets of parameter groups in a flat optimizer vector.

    ``shapes`` marks the start of the shape-parameter block, which holds one
    contiguous group per shape name (``BASE_SHAPE`` first).
    """

    continuum: int
    flux: int
    center: int
    shapes: int


def flux_expression(
    line: NoobLine, handle_by_line: dict[int, LineHandle]
) -> FluxExpression:
    """Compile a line's flux rule."""
    rule = line.flux_rule
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed flux rule is missing a value.")
        return FluxExpression(fixed=rule.value, terms={})
    if rule.is_ratio:
        if line.base is None:
            raise RuntimeError("ratio flux rule is missing a base line.")
        ratio = _ratio_value(line)
        base_expression = flux_expression(line.base, handle_by_line)
        return FluxExpression(
            fixed=base_expression.fixed * ratio,
            terms={
                source_id: scale * ratio
                for source_id, scale in base_expression.terms.items()
            },
        )
    identity = id(line)
    if identity not in handle_by_line:
        raise RuntimeError("flux source line is missing from the workspace.")
    return FluxExpression(fixed=0.0, terms={identity: 1.0})


def flux_bounds(line: NoobLine) -> tuple[float, float]:
    """Return the source flux bounds implied by a line rule."""
    rule = line.flux_rule
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded flux rule is missing bounds.")
        return rule.bounds
    if rule.is_prior:
        return prior_range(_required_prior(rule))
    return 0.0, inf


def prior_range(prior: Prior) -> tuple[float, float]:
    """Return the finite range a declared prior is optimized and sampled on.

    A prior with finite support reports it unchanged. An unbounded ``Normal``
    collapses to ``mu +/- NORMAL_RANGE_SIGMAS * sigma``, which is what lets
    the maximum-likelihood backend keep treating every parameter as a plain
    box without knowing which prior produced it.
    """
    lower, upper = prior_support(prior)
    if isfinite(lower) and isfinite(upper):
        return lower, upper
    if isinstance(prior, Normal):
        half_width = NORMAL_RANGE_SIGMAS * prior.sigma
        return prior.mu - half_width, prior.mu + half_width
    raise RuntimeError(f"prior {type(prior).__name__} has no finite range.")


def prior_initial(prior: Prior) -> float:
    """Return one representative value of a declared prior.

    Used as the optimizer start and wherever a nominal parameter value is
    needed before fitting, such as the data-derived flux scale.
    """
    if isinstance(prior, (Normal, TruncatedNormal)):
        lower, upper = prior_range(prior)
        return min(max(prior.mu, lower), upper)
    if isinstance(prior, LogUniform):
        return sqrt(prior.low * prior.high)
    if isinstance(prior, Uniform):
        return 0.5 * (prior.low + prior.high)
    raise RuntimeError(f"prior {type(prior).__name__} has no representative value.")


def scale_prior(prior: Prior, factor: float) -> Prior:
    """Return a copy of ``prior`` with every value scaled by ``factor``.

    Converts a prior declared in one unit into another, which a center rule
    needs when it is stated as a wavelength offset but compiled in km/s. The
    factor is always positive, so bounds keep their order.
    """
    if factor <= 0.0:
        raise ValueError("prior scale factor must be positive.")
    if isinstance(prior, Normal):
        return Normal(prior.mu * factor, prior.sigma * factor)
    if isinstance(prior, TruncatedNormal):
        return TruncatedNormal(
            prior.mu * factor,
            prior.sigma * factor,
            low=prior.low * factor,
            high=prior.high * factor,
        )
    if isinstance(prior, Uniform):
        return Uniform(prior.low * factor, prior.high * factor)
    if isinstance(prior, LogUniform):
        return LogUniform(prior.low * factor, prior.high * factor)
    raise RuntimeError(f"prior {type(prior).__name__} cannot be rescaled.")


def _required_prior(rule: _ParameterRule) -> Prior:
    """Return a prior rule's declared distribution."""
    if rule.prior is None:
        raise RuntimeError("prior rule is missing its distribution.")
    return rule.prior


def _velocity_prior(rule: _ParameterRule, observed_wavelength: float) -> Prior:
    """Return a center rule's prior expressed in km/s."""
    prior = _required_prior(rule)
    if rule.offset_unit == "wavelength":
        return scale_prior(prior, C_KMS / observed_wavelength)
    return prior


def center_expression(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
) -> ParameterExpression:
    """Compile a line's center rule as a velocity-offset expression."""
    return _center_expression_inner(
        handle,
        handle_by_line,
        seen=set(),
    )


def fwhm_expression(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
) -> ParameterExpression:
    """Compile a line's FWHM rule."""
    return _fwhm_expression_inner(handle, handle_by_line, seen=set())


def line_center(handle: LineHandle, handle_by_line: dict[int, LineHandle]) -> float:
    """Return the fixed-template center implied by the current center rule."""
    return handle.observed_wavelength * (
        1.0 + center_velocity_kms(handle, handle_by_line) / C_KMS
    )


def center_velocity_kms(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
    seen: set[int] | None = None,
) -> float:
    """Return the fixed-template velocity offset implied by the center rule."""
    seen = set() if seen is None else seen
    identity = id(handle.line)
    if identity in seen:
        raise ValueError("center lock graph contains a cycle.")
    seen.add(identity)

    rule = handle.line.center_rule
    if rule.is_locked:
        if rule.target is None:
            raise RuntimeError("locked center rule is missing a target.")
        return center_velocity_kms(
            handle_by_line[id(rule.target)], handle_by_line, seen
        )
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed center rule is missing a value.")
        return center_offset_to_velocity(
            rule.value, rule.offset_unit, handle.line.observed_wavelength
        )
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded center rule is missing bounds.")
        value = initial_from_bounds(rule.bounds, preferred=0.0)
        return center_offset_to_velocity(
            value, rule.offset_unit, handle.line.observed_wavelength
        )
    if rule.is_prior:
        return prior_initial(_velocity_prior(rule, handle.line.observed_wavelength))
    raise RuntimeError(
        "center rule must be fixed, bounded, or locked before compilation."
    )


def line_fwhm_kms(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
    seen: set[int] | None = None,
) -> float:
    """Return the fixed-template FWHM implied by the current FWHM rule."""
    seen = set() if seen is None else seen
    identity = id(handle.line)
    if identity in seen:
        raise ValueError("FWHM lock graph contains a cycle.")
    seen.add(identity)

    rule = handle.line.fwhm_rule
    if rule.is_locked:
        if rule.target is None:
            raise RuntimeError("locked FWHM rule is missing a target.")
        return line_fwhm_kms(handle_by_line[id(rule.target)], handle_by_line, seen)
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed FWHM rule is missing a value.")
        return rule.value
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded FWHM rule is missing bounds.")
        return initial_from_bounds(
            rule.bounds, preferred=NOMINAL_FWHM_KMS[handle.component]
        )
    if rule.is_prior:
        return prior_initial(_required_prior(rule))
    raise RuntimeError(
        "FWHM rule must be fixed, bounded, or locked before compilation."
    )


def ordered_specs(
    expressions: tuple[ParameterExpression, ...],
) -> OrderedDict[int, VariableSpec]:
    """Collect expression source specs in first-seen order."""
    specs: OrderedDict[int, VariableSpec] = OrderedDict()
    for expression in expressions:
        for source_id, spec in expression.specs.items():
            specs.setdefault(source_id, spec)
    return specs


def shape_expressions(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
) -> OrderedDict[str, ParameterExpression]:
    """Compile every named shape parameter of one line.

    Shape parameters are the width-like degrees of freedom of a line's
    profile. Every line carries the base-profile width under ``BASE_SHAPE``;
    composite profiles (convolution kernels) will contribute further named
    entries in declaration order.

    Parameters
    ----------
    handle
        The prepared line handle to compile.
    handle_by_line
        Lookup from ``id(line)`` to handle for resolving locked rules.

    Returns
    -------
    OrderedDict of {str: ParameterExpression}
        Named shape expressions, ``BASE_SHAPE`` first, then kernel shape
        parameters in declaration order.
    """
    shapes = OrderedDict({BASE_SHAPE: fwhm_expression(handle, handle_by_line)})
    for kernel in handle.line.kernels:
        for parameter, rule in kernel.rules:
            name = f"{kernel.name}__{parameter}"
            shapes[name] = _kernel_shape_expression_inner(
                name, rule, handle, handle_by_line, seen=set()
            )
    return shapes


def _kernel_shape_expression_inner(
    name: str,
    rule: _ParameterRule,
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
    *,
    seen: set[int],
) -> ParameterExpression:
    """Compile one kernel shape rule, resolving locks to the target's source."""
    identity = id(handle.line)
    if identity in seen:
        raise ValueError("kernel shape lock graph contains a cycle.")
    seen.add(identity)
    if rule.is_locked:
        if rule.target is None:
            raise RuntimeError("locked kernel shape rule is missing a target.")
        target = handle_by_line[id(rule.target)]
        return _kernel_shape_expression_inner(
            name, _kernel_rule(target.line, name), target, handle_by_line, seen=seen
        )
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed kernel rule is missing a value.")
        return ParameterExpression(fixed=rule.value, terms={}, specs={})
    if rule.is_prior:
        prior = _required_prior(rule)
        lower, upper = prior_range(prior)
        spec = VariableSpec(
            lower=lower, upper=upper, initial=prior_initial(prior), prior=prior
        )
        return ParameterExpression(
            fixed=0.0, terms={identity: 1.0}, specs={identity: spec}
        )
    if not rule.is_bounded or rule.bounds is None:
        raise RuntimeError("kernel shape rules must be fixed, bounded, or locked.")
    lower, upper = rule.bounds
    spec = VariableSpec(lower=lower, upper=upper, initial=sqrt(lower * upper))
    return ParameterExpression(fixed=0.0, terms={identity: 1.0}, specs={identity: spec})


def _kernel_rule(line: NoobLine, name: str) -> _ParameterRule:
    """Return a line's kernel shape rule by fully-qualified name."""
    for kernel in line.kernels:
        for parameter, rule in kernel.rules:
            if f"{kernel.name}__{parameter}" == name:
                return rule
    raise RuntimeError(
        f"locked kernel shape {name!r} has no matching kernel on its target line."
    )


def line_kernels_kms(
    handle: LineHandle,
    shapes: Mapping[str, ParameterExpression],
) -> tuple[tuple[str, float, float], ...]:
    """Return fixed-template kernel widths for one line.

    Evaluates each kernel shape expression at its initial (or fixed) value,
    pairing the kernel dispatch kind with a nominal width in km/s. Used
    wherever a representative template is needed before fitting (prior
    scaling, optimizer start heuristics).

    Parameters
    ----------
    handle
        The line handle whose kernels to resolve.
    shapes
        The line's compiled shape expressions from :func:`shape_expressions`.

    Returns
    -------
    tuple of (str, float, float)
        ``(kind, fwhm_kms, fraction)`` per kernel, declaration order.
    """

    def initial(name: str) -> float:
        expression = shapes[name]
        values = {
            source_id: spec.initial for source_id, spec in expression.specs.items()
        }
        return evaluate_expression(expression, values)

    return tuple(
        (
            kernel.kind,
            initial(kernel.shape_name("fwhm")),
            initial(kernel.shape_name("fraction")),
        )
        for kernel in handle.line.kernels
    )


def collect_shape_sources(
    per_handle: tuple[OrderedDict[str, ParameterExpression], ...],
) -> OrderedDict[str, OrderedDict[int, VariableSpec]]:
    """Group independent shape sources by shape name, in first-seen order.

    Parameters
    ----------
    per_handle
        Per-handle named shape expressions from :func:`shape_expressions`.

    Returns
    -------
    OrderedDict of {str: OrderedDict[int, VariableSpec]}
        For each shape name, the independent sources keyed by line identity.
    """
    names: OrderedDict[str, None] = OrderedDict()
    for shapes in per_handle:
        for name in shapes:
            names.setdefault(name, None)
    return OrderedDict(
        (
            name,
            ordered_specs(
                tuple(shapes[name] for shapes in per_handle if name in shapes)
            ),
        )
        for name in names
    )


def pack_parameter_specs(
    *groups: tuple[VariableSpec, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable specs into optimizer bounds, initial values, and scales."""
    specs = tuple(spec for group in groups for spec in group)
    lower = np.asarray([spec.lower for spec in specs], dtype=float)
    upper = np.asarray([spec.upper for spec in specs], dtype=float)
    initial = np.asarray(
        [clip_initial(spec.initial, spec.lower, spec.upper) for spec in specs],
        dtype=float,
    )
    scale = np.asarray(
        [
            parameter_scale(spec, init)
            for spec, init in zip(specs, initial, strict=True)
        ],
        dtype=float,
    )
    return lower, upper, initial, scale


def clip_initial(value: float, lower: float, upper: float) -> float:
    """Clip an initial value inside optimizer bounds."""
    if lower == upper:
        return lower
    if upper < inf:
        value = min(value, upper - max(abs(upper) * 1e-8, 1e-30))
    if lower > -inf:
        value = max(value, lower + max(abs(lower) * 1e-8, 1e-30))
    return value


def evaluate_expression(
    expression: FluxExpression | ParameterExpression, values: Mapping[int, float]
) -> float:
    """Evaluate a compiled linear expression."""
    total = expression.fixed
    for source_id, scale in expression.terms.items():
        total += scale * values[source_id]
    return float(total)


def initial_from_bounds(bounds: tuple[float, float], *, preferred: float) -> float:
    """Return a preferred initial value clipped to bounds."""
    lower, upper = bounds
    if lower <= preferred <= upper:
        return preferred
    return min(max(preferred, lower), upper)


def contribution_sign(handle: LineHandle) -> float:
    """Return the signed contribution multiplier for a line."""
    return -1.0 if handle.contribution == "absorption" else 1.0


def center_offset_to_velocity(value: float, unit: str | None, center: float) -> float:
    """Convert a center offset to km/s.

    For a ``"wavelength"`` offset, ``value`` and ``center`` must be in the
    same wavelength unit — the velocity depends only on their ratio, so
    passing the line's own observed wavelength (in the line's declared unit)
    keeps a ``delta_wavelength`` rule correct regardless of the spectrum's
    unit. For ``"km/s"`` (or ``None``) offsets, ``center`` is unused.
    """
    if unit == "wavelength":
        return value / center * C_KMS
    return value


def parameter_scale(spec: VariableSpec, initial: float) -> float:
    """Return an optimizer scale for one variable."""
    if spec.upper < inf and spec.lower > -inf:
        return max((spec.upper - spec.lower) / 4.0, abs(initial), 1e-30)
    return max(abs(initial), 1e-30)


def _ratio_value(line: NoobLine) -> float:
    rule = line.flux_rule
    if rule.value is not None:
        return rule.value
    raise RuntimeError("ratio flux rule is missing a value.")


def _center_expression_inner(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
    *,
    seen: set[int],
) -> ParameterExpression:
    identity = id(handle.line)
    if identity in seen:
        raise ValueError("center lock graph contains a cycle.")
    seen.add(identity)
    rule = handle.line.center_rule
    if rule.is_locked:
        if rule.target is None:
            raise RuntimeError("locked center rule is missing a target.")
        return _center_expression_inner(
            handle_by_line[id(rule.target)],
            handle_by_line,
            seen=seen,
        )
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed center rule is missing a value.")
        return ParameterExpression(
            fixed=center_offset_to_velocity(
                rule.value, rule.offset_unit, handle.line.observed_wavelength
            ),
            terms={},
            specs={},
        )
    prior: Prior | None = None
    if rule.is_prior:
        prior = _velocity_prior(rule, handle.line.observed_wavelength)
        lower, upper = prior_range(prior)
        initial = prior_initial(prior)
    elif rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded center rule is missing bounds.")
        lower = center_offset_to_velocity(
            rule.bounds[0], rule.offset_unit, handle.line.observed_wavelength
        )
        upper = center_offset_to_velocity(
            rule.bounds[1], rule.offset_unit, handle.line.observed_wavelength
        )
        initial = initial_from_bounds((lower, upper), preferred=0.0)
    else:
        raise RuntimeError(
            "center rule must be fixed, bounded, or locked before compilation."
        )
    spec = VariableSpec(lower=lower, upper=upper, initial=initial, prior=prior)
    return ParameterExpression(fixed=0.0, terms={identity: 1.0}, specs={identity: spec})


def _fwhm_expression_inner(
    handle: LineHandle,
    handle_by_line: dict[int, LineHandle],
    *,
    seen: set[int],
) -> ParameterExpression:
    identity = id(handle.line)
    if identity in seen:
        raise ValueError("FWHM lock graph contains a cycle.")
    seen.add(identity)
    rule = handle.line.fwhm_rule
    if rule.is_locked:
        if rule.target is None:
            raise RuntimeError("locked FWHM rule is missing a target.")
        return _fwhm_expression_inner(
            handle_by_line[id(rule.target)],
            handle_by_line,
            seen=seen,
        )
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed FWHM rule is missing a value.")
        return ParameterExpression(fixed=rule.value, terms={}, specs={})
    prior: Prior | None = None
    if rule.is_prior:
        prior = _required_prior(rule)
        lower, upper = prior_range(prior)
        initial = prior_initial(prior)
    elif rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded FWHM rule is missing bounds.")
        lower, upper = rule.bounds
        initial = initial_from_bounds(
            (lower, upper), preferred=NOMINAL_FWHM_KMS[handle.component]
        )
    else:
        raise RuntimeError(
            "FWHM rule must be fixed, bounded, or locked before compilation."
        )
    spec = VariableSpec(lower=lower, upper=upper, initial=initial, prior=prior)
    return ParameterExpression(fixed=0.0, terms={identity: 1.0}, specs={identity: spec})
