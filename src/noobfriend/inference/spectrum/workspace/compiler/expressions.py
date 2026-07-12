"""Compile line rules into reusable parameter expressions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler.profiles import C_KMS

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.workspace.handles import LineHandle


_INIT_FWHM_KMS = {
    "narrow": 200.0,
    "broad": 1200.0,
}


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
    """Bounded scalar variable specification used by fit backends."""

    lower: float
    upper: float
    initial: float


@dataclass(frozen=True, slots=True)
class ParameterOffsets:
    """Offsets of parameter groups in a flat optimizer vector."""

    continuum: int
    flux: int
    center: int
    fwhm: int


def flux_expression(line: NoobLine, handle_by_line: dict[int, LineHandle]) -> FluxExpression:
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
            terms={source_id: scale * ratio for source_id, scale in base_expression.terms.items()},
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
    return 0.0, inf


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
    return handle.observed_wavelength * (1.0 + center_velocity_kms(handle, handle_by_line) / C_KMS)


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
        return center_velocity_kms(handle_by_line[id(rule.target)], handle_by_line, seen)
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed center rule is missing a value.")
        return center_offset_to_velocity(rule.value, rule.offset_unit, handle.observed_wavelength)
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded center rule is missing bounds.")
        value = initial_from_bounds(rule.bounds, preferred=0.0)
        return center_offset_to_velocity(value, rule.offset_unit, handle.observed_wavelength)
    raise RuntimeError("center rule must be fixed, bounded, or locked before compilation.")


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
        return initial_from_bounds(rule.bounds, preferred=_INIT_FWHM_KMS[handle.component])
    raise RuntimeError("FWHM rule must be fixed, bounded, or locked before compilation.")


def ordered_specs(expressions: tuple[ParameterExpression, ...]) -> OrderedDict[int, VariableSpec]:
    """Collect expression source specs in first-seen order."""
    specs: OrderedDict[int, VariableSpec] = OrderedDict()
    for expression in expressions:
        for source_id, spec in expression.specs.items():
            specs.setdefault(source_id, spec)
    return specs


def pack_parameter_specs(
    *groups: tuple[VariableSpec, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable specs into optimizer bounds, initial values, and scales."""
    specs = tuple(spec for group in groups for spec in group)
    lower = np.asarray([spec.lower for spec in specs], dtype=float)
    upper = np.asarray([spec.upper for spec in specs], dtype=float)
    initial = np.asarray([clip_initial(spec.initial, spec.lower, spec.upper) for spec in specs], dtype=float)
    scale = np.asarray([parameter_scale(spec, init) for spec, init in zip(specs, initial, strict=True)], dtype=float)
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


def evaluate_expression(expression: FluxExpression | ParameterExpression, values: Mapping[int, float]) -> float:
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
    """Convert a center offset to km/s."""
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
            fixed=center_offset_to_velocity(rule.value, rule.offset_unit, handle.observed_wavelength),
            terms={},
            specs={},
        )
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded center rule is missing bounds.")
        lower = center_offset_to_velocity(rule.bounds[0], rule.offset_unit, handle.observed_wavelength)
        upper = center_offset_to_velocity(rule.bounds[1], rule.offset_unit, handle.observed_wavelength)
    else:
        raise RuntimeError("center rule must be fixed, bounded, or locked before compilation.")
    spec = VariableSpec(lower=lower, upper=upper, initial=initial_from_bounds((lower, upper), preferred=0.0))
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
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded FWHM rule is missing bounds.")
        lower, upper = rule.bounds
    else:
        raise RuntimeError("FWHM rule must be fixed, bounded, or locked before compilation.")
    spec = VariableSpec(
        lower=lower,
        upper=upper,
        initial=initial_from_bounds((lower, upper), preferred=_INIT_FWHM_KMS[handle.component]),
    )
    return ParameterExpression(fixed=0.0, terms={identity: 1.0}, specs={identity: spec})
