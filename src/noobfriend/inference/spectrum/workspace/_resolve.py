"""Resolve declarations into the parameter table, handles and fit pixels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

from noobfriend.inference.spectrum._wavelength import (
    C_KMS,
    Interval,
    convert_wavelength,
    in_intervals,
)
from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.noise import AR1
from noobfriend.inference.spectrum.priors import (
    CORRELATION,
    FRACTION,
    NONNEGATIVE,
    POSITIVE,
    REAL,
    Domain,
    JointMember,
    Prior,
)
from noobfriend.inference.spectrum.profiles import line_profile
from noobfriend.inference.spectrum.workspace import _defaults as defaults
from noobfriend.inference.spectrum.workspace._expressions import (
    Const,
    Expression,
    Product,
    Ref,
    Scale,
    Sum,
    simplify,
)
from noobfriend.inference.spectrum.workspace._handles import ContinuumHandle, LineHandle
from noobfriend.inference.spectrum.workspace._protocols import (
    ContinuumLike,
    LineLike,
    SpectrumLike,
)
from noobfriend.inference.spectrum.workspace._table import (
    JointBlock,
    Parameter,
    ParameterTable,
    Quantity,
)


@dataclass(frozen=True, slots=True)
class DataScales:
    """Scales of the fitted pixels used for data-derived defaults.

    Attributes
    ----------
    amplitude
        ``max(ptp(flux), median(error))``: the height a line could have.
    flux
        ``S = max(amplitude, median|flux|, median(error))``.
    wavelength
        ``W``: half the fitted wavelength span.
    """

    amplitude: float
    flux: float
    wavelength: float


@dataclass(frozen=True, slots=True)
class Resolution:
    """Everything ``prepare`` derives from the declarations."""

    fitted: Mapping[str, np.ndarray]
    lines: tuple[LineHandle, ...]
    continuum: ContinuumHandle
    table: ParameterTable
    joint_blocks: tuple[JointBlock, ...]
    scales: DataScales


def fitted_pixels(
    spectrum: SpectrumLike, window: tuple[Interval, ...] | None
) -> dict[str, np.ndarray]:
    """Return each frame's fitted mask: usable and inside the window.

    Raises
    ------
    ValueError
        If a frame has no fitted pixel.
    """
    fitted = {}
    for frame_id, frame in spectrum.frames.items():
        mask = np.array(frame.usable, dtype=bool)
        if window is not None:
            mask &= in_intervals(frame.obs, window)
        if not mask.any():
            raise ValueError(
                f"frame {frame_id!r} has no usable pixel inside the window."
            )
        mask.setflags(write=False)
        fitted[frame_id] = mask
    return fitted


def resolve(
    spectrum: SpectrumLike,
    lines: Sequence[LineLike],
    continuum: ContinuumLike,
    window: tuple[Interval, ...] | None,
) -> Resolution:
    """Resolve declarations against the data.

    Parameters
    ----------
    spectrum
        The data.
    lines
        Line declarations in prepare order.
    continuum
        The continuum declaration.
    window
        Fit window intervals (observed frame, data unit) or ``None``.

    Returns
    -------
    Resolution
        Table, handles, fitted pixels and data scales.
    """
    fitted = fitted_pixels(spectrum, window)
    scales = _data_scales(spectrum, fitted)
    builder = _TableBuilder()
    by_name = _check_lines(lines)
    spans = window if window is not None else _data_spans(spectrum, fitted)

    handles: dict[str, LineHandle] = {}
    for line in _base_first(lines, by_name):
        handles[line.name] = _resolve_line(line, spectrum, spans, scales, builder)
    ordered = tuple(handles[line.name] for line in lines)
    continuum_handle = _resolve_continuum(continuum, spectrum, scales, builder)
    _resolve_noise(spectrum, builder)
    table = ParameterTable(builder.rows, builder.quantities)
    return Resolution(
        fitted=MappingProxyType(fitted),
        lines=ordered,
        continuum=continuum_handle,
        table=table,
        joint_blocks=_joint_blocks(table),
        scales=scales,
    )


class _TableBuilder:
    """Accumulates rows and quantities plus the expressions lines refer to."""

    def __init__(self) -> None:
        self.rows: list[Parameter] = []
        self.quantities: list[Quantity] = []
        #: Physical expressions by flat quantity name, for derived lines.
        self.expressions: dict[str, Expression] = {}

    def declare(
        self,
        name: str,
        owner: str,
        role: str,
        unit: str,
        domain: Domain,
        value: float | Prior,
    ) -> Expression:
        """Add a fixed or free row and return the expression reading it."""
        if isinstance(value, Prior):
            self.rows.append(
                Parameter(
                    name, owner, role, unit, domain, "free", prior=value.resolve(domain)
                )
            )
            return Ref(name)
        self.rows.append(
            Parameter(name, owner, role, unit, domain, "fixed", value=float(value))
        )
        return Const(float(value), name)

    def quantity(
        self, name: str, owner: str, role: str, unit: str, expression: Expression
    ) -> None:
        """Record a physical line quantity."""
        self.expressions[name] = expression
        self.quantities.append(
            Quantity(name, owner, role, unit, expression, simplify(expression))
        )


def _check_lines(lines: Sequence[LineLike]) -> dict[str, LineLike]:
    by_name: dict[str, LineLike] = {}
    for line in lines:
        if not isinstance(line, LineLike):
            raise TypeError(
                f"prepare expects NoobLine declarations; got {type(line).__name__}."
            )
        if line.name in by_name:
            raise ValueError(
                f"line name {line.name!r} is used twice; names must be unique."
            )
        by_name[line.name] = line
    for line in lines:
        if line.base is not None and line.base.name not in by_name:
            raise ValueError(
                f"line {line.name!r} is derived from {line.base.name!r}, which is not "
                "among the prepared lines."
            )
    return by_name


def _base_first(
    lines: Sequence[LineLike], by_name: Mapping[str, LineLike]
) -> list[LineLike]:
    """Order lines so every base is resolved before the lines derived from it."""
    ordered: list[LineLike] = []
    seen: set[str] = set()

    def visit(line: LineLike, chain: tuple[str, ...]) -> None:
        if line.name in seen:
            return
        if line.name in chain:
            raise ValueError(f"derive cycle through {line.name!r}.")
        if line.base is not None:
            visit(by_name[line.base.name], (*chain, line.name))
        seen.add(line.name)
        ordered.append(line)

    for line in lines:
        visit(line, ())
    return ordered


def _data_scales(
    spectrum: SpectrumLike, fitted: Mapping[str, np.ndarray]
) -> DataScales:
    flux = np.concatenate(
        [np.asarray(f.flux)[fitted[i]] for i, f in spectrum.frames.items()]
    )
    error = np.concatenate(
        [np.asarray(f.error)[fitted[i]] for i, f in spectrum.frames.items()]
    )
    obs = np.concatenate(
        [np.asarray(f.obs)[fitted[i]] for i, f in spectrum.frames.items()]
    )
    amplitude = max(float(np.ptp(flux)), float(np.median(error)))
    flux_scale = max(amplitude, float(np.median(np.abs(flux))), float(np.median(error)))
    span = float(np.ptp(obs)) / 2.0
    return DataScales(amplitude, flux_scale, span if span > 0 else 1.0)


def _data_spans(
    spectrum: SpectrumLike, fitted: Mapping[str, np.ndarray]
) -> tuple[Interval, ...]:
    spans = []
    for frame_id, frame in spectrum.frames.items():
        obs = np.asarray(frame.obs)[fitted[frame_id]]
        spans.append((float(obs[0]), float(obs[-1])))
    return tuple(spans)


def _resolve_line(
    line: LineLike,
    spectrum: SpectrumLike,
    spans: tuple[Interval, ...],
    scales: DataScales,
    builder: _TableBuilder,
) -> LineHandle:
    name = line.name
    unit = spectrum.unit
    if line.rest is not None:
        if spectrum.z is None:
            raise ValueError(
                f"line {name!r} has a rest wavelength but the spectrum has no z."
            )
        nominal = convert_wavelength(line.rest, from_unit=line.unit, to_unit=unit)
        nominal *= 1.0 + spectrum.z
    else:
        nominal = convert_wavelength(line.obs, from_unit=line.unit, to_unit=unit)
    if not any(lo <= nominal <= hi for lo, hi in spans):
        raise ValueError(
            f"line {name!r} at observed {nominal:.6g} {unit} lies outside every fit window."
        )
    lsf = {
        frame_id: _lsf(frame.resolving_power_at(nominal))
        for frame_id, frame in spectrum.frames.items()
    }
    base = line.base.name if line.base is not None else None
    start = len(builder.rows)
    expressions = builder.expressions

    # Centre: velocity offset relative to the line's own nominal wavelength.
    rule = line.center_rule
    if rule.kind == "default":
        delta_v = builder.declare(
            f"{name}.delta_v", name, "delta_v", "km/s", REAL, defaults.CENTER_PRIOR
        )
    elif rule.kind == "delta_v_kms":
        delta_v = builder.declare(
            f"{name}.delta_v", name, "delta_v", "km/s", REAL, rule.value
        )
    elif rule.kind == "delta_wavelength":
        factor = convert_wavelength(1.0, from_unit=line.unit, to_unit=unit)
        value = (
            rule.value.scaled(factor)
            if isinstance(rule.value, Prior)
            else rule.value * factor
        )
        shift = builder.declare(
            f"{name}.delta_wavelength", name, "delta_wavelength", unit, REAL, value
        )
        delta_v = Scale(C_KMS / nominal, shift)
    else:
        difference = builder.declare(
            f"{name}.delta_v_from_base",
            name,
            "delta_v_from_base",
            "km/s",
            REAL,
            rule.value,
        )
        delta_v = Sum(expressions[f"{base}.delta_v"], difference)
    builder.quantity(f"{name}.delta_v", name, "delta_v", "km/s", delta_v)
    center = Sum(Const(nominal), Scale(nominal / C_KMS, delta_v))
    builder.quantity(f"{name}.center", name, "center", unit, center)

    fwhm = _scale_quantity(
        builder, line, "fwhm", "km/s", POSITIVE, defaults.FWHM_PRIORS[line.component]
    )
    builder.quantity(f"{name}.fwhm", name, "fwhm", "km/s", fwhm)

    for kernel in line.kernels:
        for role, value, domain in (
            ("fwhm", kernel.fwhm, POSITIVE),
            ("fraction", kernel.fraction, FRACTION),
        ):
            flat = f"{name}.{kernel.kind}.{role}"
            unit_text = "km/s" if role == "fwhm" else ""
            if value is None:
                locked = expressions.get(f"{base}.{kernel.kind}.{role}")
                if locked is None:
                    raise ValueError(
                        f"line {name!r} locks its {kernel.kind} kernel to base {base!r}, "
                        "which has no such kernel."
                    )
                expression = locked
            else:
                expression = builder.declare(
                    flat, name, f"{kernel.kind}.{role}", unit_text, domain, value
                )
            builder.quantity(flat, name, f"{kernel.kind}.{role}", unit_text, expression)

    flux_unit = f"flux*{unit}"
    default_flux = None
    if line.flux_rule.kind == "default":
        nominal_fwhm = simplify(fwhm)
        width = (
            nominal_fwhm.value
            if isinstance(nominal_fwhm, Const)
            else defaults.NOMINAL_FWHM_KMS[line.component]
        )
        default_flux = defaults.flux_prior(
            _reference_flux(line.profile, nominal, width, lsf, scales.amplitude)
        )
    flux = _scale_quantity(builder, line, "flux", flux_unit, NONNEGATIVE, default_flux)
    builder.quantity(f"{name}.flux", name, "flux", flux_unit, flux)

    return LineHandle(
        name=name,
        component=line.component,
        contribution=line.contribution,
        profile=line.profile,
        base=base,
        nominal=nominal,
        sign=1.0 if line.contribution == "emission" else -1.0,
        kernels=tuple(kernel.kind for kernel in line.kernels),
        lsf_fwhm_kms=MappingProxyType(lsf),
        parameters=tuple(row.name for row in builder.rows[start:]),
        declaration=line,
    )


def _scale_quantity(
    builder: _TableBuilder,
    line: LineLike,
    role: str,
    unit: str,
    domain: Domain,
    default: Prior | None,
) -> Expression:
    """Resolve a width or flux rule (independent value or ratio to the base)."""
    name = line.name
    rule = line.fwhm_rule if role == "fwhm" else line.flux_rule
    if rule.kind == "ratio_to_base":
        ratio_domain = POSITIVE if role == "fwhm" else NONNEGATIVE
        ratio = builder.declare(
            f"{name}.{role}_ratio_to_base",
            name,
            f"{role}_ratio_to_base",
            "",
            ratio_domain,
            rule.value,
        )
        return Product(builder.expressions[f"{line.base.name}.{role}"], ratio)
    value = default if rule.kind == "default" else rule.value
    return builder.declare(f"{name}.{role}", name, role, unit, domain, value)


def _reference_flux(
    profile: str,
    nominal: float,
    fwhm_kms: float,
    lsf: Mapping[str, float | None],
    amplitude: float,
) -> float:
    """Flux whose unit-area profile peaks at the data amplitude.

    Uses the frame with the broadest LSF (lowest peak), so the derived flux
    prior is the wider one.
    """
    xp = numpy_namespace()
    half_width = (
        5.0 * nominal * max([fwhm_kms, *(v or 0.0 for v in lsf.values())]) / C_KMS
    )
    grid = np.linspace(nominal - half_width, nominal + half_width, 2001)
    peaks = [
        float(
            np.max(
                line_profile(
                    xp,
                    grid,
                    profile=profile,
                    center=nominal,
                    fwhm_kms=fwhm_kms,
                    lsf_fwhm_kms=width,
                )
            )
        )
        for width in (set(lsf.values()) or {None})
    ]
    return amplitude / min(peaks)


def _resolve_continuum(
    continuum: ContinuumLike,
    spectrum: SpectrumLike,
    scales: DataScales,
    builder: _TableBuilder,
) -> ContinuumHandle:
    if not isinstance(continuum, ContinuumLike):
        raise TypeError(
            f"continuum must be a NoobContinuum; got {type(continuum).__name__}."
        )
    frame_ids = tuple(spectrum.frames)
    sharing = continuum.sharing if len(frame_ids) > 1 else "shared"
    unit = spectrum.unit
    start = len(builder.rows)
    frame_parameters: dict[str, list[str]] = {frame_id: [] for frame_id in frame_ids}
    for parameter in continuum.parameter_names:
        domain = continuum.domain(parameter)
        declared = continuum.declarations.get(parameter)
        value = (
            declared if declared is not None else _continuum_default(parameter, scales)
        )
        unit_text = _continuum_unit(parameter, unit)
        mean_name = f"continuum.{parameter}"
        if not isinstance(value, Prior) or sharing == "shared":
            builder.declare(mean_name, "continuum", parameter, unit_text, domain, value)
            for frame_id in frame_ids:
                frame_parameters[frame_id].append(mean_name)
            continue
        if sharing == "independent":
            for frame_id in frame_ids:
                name = f"frame.{frame_id}.continuum.{parameter}"
                builder.declare(
                    name,
                    f"frame.{frame_id}",
                    f"continuum.{parameter}",
                    unit_text,
                    domain,
                    value,
                )
                frame_parameters[frame_id].append(name)
            continue
        builder.declare(mean_name, "continuum", parameter, unit_text, domain, value)
        log_pooled = parameter == "level"
        spread = defaults.spread_prior(parameter, scales.flux, scales.wavelength)
        builder.declare(
            f"{mean_name}.spread",
            "continuum",
            f"{parameter}.spread",
            "dex-e" if log_pooled else unit_text,
            POSITIVE,
            spread,
        )
        for frame_id in frame_ids:
            name = f"frame.{frame_id}.continuum.{parameter}"
            builder.rows.append(
                Parameter(
                    name,
                    f"frame.{frame_id}",
                    f"continuum.{parameter}",
                    unit_text,
                    domain,
                    "pooled",
                    pooled_from=mean_name,
                    log_pooled=log_pooled,
                )
            )
            frame_parameters[frame_id].append(name)
    return ContinuumHandle(
        kind=continuum.kind,
        order=continuum.order,
        lambda_0=continuum.lambda_0,
        sharing=sharing,
        parameter_names=tuple(continuum.parameter_names),
        frame_parameters=MappingProxyType(
            {frame_id: tuple(names) for frame_id, names in frame_parameters.items()}
        ),
        parameters=tuple(row.name for row in builder.rows[start:]),
        declaration=continuum,
    )


def _continuum_default(parameter: str, scales: DataScales) -> Prior:
    if parameter == "level":
        return defaults.level_prior(scales.flux)
    if parameter == "index":
        return defaults.INDEX_PRIOR
    return defaults.coefficient_prior(
        scales.flux, scales.wavelength, int(parameter[1:])
    )


def _continuum_unit(parameter: str, unit: str) -> str:
    if parameter == "index":
        return ""
    if parameter == "level" or parameter == "c0":
        return "flux"
    degree = int(parameter[1:])
    return f"flux/{unit}" if degree == 1 else f"flux/{unit}^{degree}"


def _resolve_noise(spectrum: SpectrumLike, builder: _TableBuilder) -> None:
    for frame_id, frame in spectrum.frames.items():
        owner = f"frame.{frame_id}"
        noise = frame.noise
        builder.declare(
            f"{owner}.noise.scale", owner, "noise.scale", "", POSITIVE, noise.scale
        )
        correlation = noise.correlation
        if isinstance(correlation, AR1):
            builder.declare(
                f"{owner}.noise.rho",
                owner,
                "noise.rho",
                "",
                CORRELATION,
                correlation.rho,
            )


def _joint_blocks(table: ParameterTable) -> tuple[JointBlock, ...]:
    """Group joint-prior members and marginalize each joint prior onto them."""
    groups: dict[int, list[Parameter]] = {}
    joints = {}
    for row in table.free():
        if isinstance(row.prior, JointMember):
            key = id(row.prior.joint)
            joints[key] = row.prior.joint
            groups.setdefault(key, []).append(row)
    blocks = []
    for key, rows in groups.items():
        joint = joints[key]
        members = tuple(row.prior.name for row in rows)  # type: ignore[union-attr]
        if len(set(members)) != len(members):
            raise ValueError("one joint prior member is used by two parameters.")
        mean, covariance = joint.marginal(members)
        blocks.append(
            JointBlock(
                joint=joint,
                parameters=tuple(row.name for row in rows),
                members=members,
                mean=mean,
                covariance=covariance,
                marginalized=tuple(name for name in joint.names if name not in members),
            )
        )
    return tuple(blocks)


def _lsf(resolving_power: float | None) -> float | None:
    return None if resolving_power is None else C_KMS / resolving_power
