"""Numerical maximum-likelihood problem compiled from one workspace."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from noobfriend.inference.spectrum.workspace.compiler import (
    C_KMS,
    CompiledLineGraph,
    ParameterExpression,
    ParameterOffsets,
    compile_line_graph,
    contribution_sign,
    evaluate_expression,
    flux_bounds,
    profile_template,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import LineHandle, NoobFitWorkspace


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """One physical parameter vector evaluated as a spectrum model."""

    continuum: np.ndarray
    line_models: tuple[np.ndarray, ...]
    model: np.ndarray
    line_fluxes: tuple[float, ...]
    velocity_offsets_kms: tuple[float, ...]
    centers: tuple[float, ...]
    fwhm_kms: tuple[float, ...]
    cancellation: float
    cancellation_fraction: float


@dataclass(frozen=True, slots=True)
class MLEProblem:
    """Scaled optimizer problem for one internal absorption-bound variant."""

    workspace: NoobFitWorkspace
    graph: CompiledLineGraph
    wavelength: np.ndarray
    data: np.ndarray
    error: np.ndarray
    continuum_design: np.ndarray
    flux_source_ids: tuple[int, ...]
    center_source_ids: tuple[int, ...]
    fwhm_source_ids: tuple[int, ...]
    offsets: ParameterOffsets
    lower: np.ndarray
    upper: np.ndarray
    initial: np.ndarray
    scale: np.ndarray
    absorption_bound_multiplier: float

    @property
    def size(self) -> int:
        """Number of optimizer variables."""
        return int(self.initial.size)

    @property
    def continuum_slice(self) -> slice:
        """Continuum coefficient slice."""
        return slice(self.offsets.continuum, self.offsets.flux)

    @property
    def flux_slice(self) -> slice:
        """Independent flux source slice."""
        return slice(self.offsets.flux, self.offsets.center)

    @property
    def center_slice(self) -> slice:
        """Independent velocity-offset source slice."""
        return slice(self.offsets.center, self.offsets.fwhm)

    @property
    def fwhm_slice(self) -> slice:
        """Independent FWHM source slice."""
        return slice(self.offsets.fwhm, self.size)

    @property
    def normalized_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds in scaled optimizer coordinates."""
        return self.lower / self.scale, self.upper / self.scale

    @property
    def has_free_absorption(self) -> bool:
        """Whether this problem contains an internally bounded free absorption flux."""
        return any(self._is_free_absorption(source_id) for source_id in self.flux_source_ids)

    @property
    def has_free_broad_flux(self) -> bool:
        """Whether this problem contains an independent broad-line flux."""
        return any(self.source_handle(source_id).component == "broad" for source_id in self.flux_source_ids)

    @property
    def data_range(self) -> float:
        """Peak-to-peak valid data range used for internal start scales."""
        return float(np.ptp(self.data))

    def source_handle(self, source_id: int) -> LineHandle:
        """Return the workspace handle that owns an independent parameter."""
        return self.graph.handle_by_line[source_id]

    def to_normalized(self, physical: np.ndarray) -> np.ndarray:
        """Convert physical parameters to scaled optimizer coordinates."""
        return np.asarray(physical, dtype=float) / self.scale

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        """Convert scaled optimizer parameters to physical coordinates."""
        return np.asarray(normalized, dtype=float) * self.scale

    def clip(self, physical: np.ndarray) -> np.ndarray:
        """Clip a physical vector strictly inside finite optimizer bounds."""
        value = np.asarray(physical, dtype=float).copy()
        for index, (lower, upper, scale) in enumerate(zip(self.lower, self.upper, self.scale, strict=True)):
            if np.isfinite(lower) and np.isfinite(upper):
                margin = min(0.1 * (upper - lower), max(abs(lower), abs(upper), scale) * 1e-10)
                value[index] = np.clip(value[index], lower + margin, upper - margin)
            elif np.isfinite(lower):
                value[index] = max(value[index], lower + max(abs(lower), scale) * 1e-10)
            elif np.isfinite(upper):
                value[index] = min(value[index], upper - max(abs(upper), scale) * 1e-10)
        return value

    def residual(self, physical: np.ndarray) -> np.ndarray:
        """Return error-normalized model residuals on valid pixels."""
        return (self.evaluate(physical).model - self.data) / self.error

    def residual_normalized(self, normalized: np.ndarray) -> np.ndarray:
        """Return residuals from scaled optimizer coordinates."""
        return self.residual(self.to_physical(normalized))

    def chi2(self, physical: np.ndarray) -> float:
        """Return raw chi-square for a physical vector."""
        residual = self.residual(physical)
        return float(np.dot(residual, residual))

    def evaluate(self, physical: np.ndarray, *, wavelength: np.ndarray | None = None) -> ModelEvaluation:
        """Evaluate continuum and all signed line models."""
        parameters = np.asarray(physical, dtype=float)
        if parameters.shape != (self.size,):
            raise ValueError(f"physical parameter vector must have shape ({self.size},).")

        wl = self.wavelength if wavelength is None else np.asarray(wavelength, dtype=float)
        design = self.continuum_design if wavelength is None else self.workspace.continuum.design(wl)
        continuum = design @ parameters[self.continuum_slice]
        flux_values = dict(zip(self.flux_source_ids, parameters[self.flux_slice], strict=True))
        center_values = dict(zip(self.center_source_ids, parameters[self.center_slice], strict=True))
        fwhm_values = dict(zip(self.fwhm_source_ids, parameters[self.fwhm_slice], strict=True))

        line_models: list[np.ndarray] = []
        line_fluxes: list[float] = []
        velocity_offsets: list[float] = []
        centers: list[float] = []
        fwhm_values_kms: list[float] = []
        emission = np.zeros_like(wl, dtype=float)
        absorption = np.zeros_like(wl, dtype=float)

        for handle, flux_expr, center_expr, fwhm_expr in zip(
            self.workspace.handles,
            self.graph.flux_expressions,
            self.graph.center_expressions,
            self.graph.fwhm_expressions,
            strict=True,
        ):
            flux = evaluate_expression(flux_expr, flux_values)
            velocity = evaluate_expression(center_expr, center_values)
            fwhm = evaluate_expression(fwhm_expr, fwhm_values)
            center = handle.observed_wavelength * (1.0 + velocity / C_KMS)
            template = profile_template(
                wl,
                handle=handle,
                center=center,
                fwhm_kms=fwhm,
                resolving_power=self.workspace.spectrum.resolving_power,
            )
            unsigned = flux * template
            signed = contribution_sign(handle) * unsigned
            line_models.append(signed)
            line_fluxes.append(flux)
            velocity_offsets.append(velocity)
            centers.append(center)
            fwhm_values_kms.append(fwhm)
            if handle.contribution == "absorption":
                absorption += unsigned
            else:
                emission += unsigned

        line_total = np.sum(line_models, axis=0) if line_models else np.zeros_like(wl)
        model = continuum + line_total
        cancellation = 2.0 * float(np.trapezoid(np.minimum(emission, absorption), wl))
        total_flux = float(sum(abs(value) for value in line_fluxes))
        cancellation_fraction = cancellation / total_flux if total_flux > 0.0 else 0.0
        return ModelEvaluation(
            continuum=continuum,
            line_models=tuple(line_models),
            model=model,
            line_fluxes=tuple(line_fluxes),
            velocity_offsets_kms=tuple(velocity_offsets),
            centers=tuple(centers),
            fwhm_kms=tuple(fwhm_values_kms),
            cancellation=cancellation,
            cancellation_fraction=cancellation_fraction,
        )

    def linearized_start(self, physical: np.ndarray) -> np.ndarray:
        """Solve continuum and source fluxes at fixed center/FWHM geometry."""
        start = self.clip(physical)
        center_values = dict(zip(self.center_source_ids, start[self.center_slice], strict=True))
        fwhm_values = dict(zip(self.fwhm_source_ids, start[self.fwhm_slice], strict=True))
        source_columns = {source_id: np.zeros_like(self.wavelength) for source_id in self.flux_source_ids}
        fixed_model = np.zeros_like(self.wavelength)

        for handle, flux_expr, center_expr, fwhm_expr in zip(
            self.workspace.handles,
            self.graph.flux_expressions,
            self.graph.center_expressions,
            self.graph.fwhm_expressions,
            strict=True,
        ):
            velocity = evaluate_expression(center_expr, center_values)
            fwhm = evaluate_expression(fwhm_expr, fwhm_values)
            center = handle.observed_wavelength * (1.0 + velocity / C_KMS)
            signed_template = contribution_sign(handle) * profile_template(
                self.wavelength,
                handle=handle,
                center=center,
                fwhm_kms=fwhm,
                resolving_power=self.workspace.spectrum.resolving_power,
            )
            fixed_model += flux_expr.fixed * signed_template
            for source_id, coefficient in flux_expr.terms.items():
                source_columns[source_id] += coefficient * signed_template

        matrix = np.column_stack(
            [self.continuum_design, *(source_columns[source_id] for source_id in self.flux_source_ids)]
        )
        weighted_matrix = matrix / self.error[:, None]
        weighted_data = (self.data - fixed_model) / self.error
        linear_lower = np.concatenate((self.lower[self.continuum_slice], self.lower[self.flux_slice]))
        linear_upper = np.concatenate((self.upper[self.continuum_slice], self.upper[self.flux_slice]))
        try:
            solution = lsq_linear(
                weighted_matrix,
                weighted_data,
                bounds=(linear_lower, linear_upper),
                lsmr_tol="auto",
                max_iter=200,
            )
        except (ValueError, np.linalg.LinAlgError):
            return start
        if not solution.success or not np.all(np.isfinite(solution.x)):
            return start
        stop = self.offsets.center
        start[:stop] = solution.x
        return self.clip(start)

    def flux_for_peak(self, source_id: int, *, fwhm_kms: float, peak_fraction: float) -> float:
        """Return a source flux whose nominal template reaches a data-range fraction."""
        handle = self.source_handle(source_id)
        template = profile_template(
            self.wavelength,
            handle=handle,
            center=handle.observed_wavelength,
            fwhm_kms=fwhm_kms,
            resolving_power=self.workspace.spectrum.resolving_power,
        )
        peak = float(np.max(template))
        amplitude = max(self.data_range, float(np.median(self.error)))
        return peak_fraction * amplitude / peak

    def _is_free_absorption(self, source_id: int) -> bool:
        handle = self.source_handle(source_id)
        return handle.contribution == "absorption" and handle.line.flux_rule.is_free


def build_mle_problem(workspace: NoobFitWorkspace, *, absorption_bound_multiplier: float) -> MLEProblem:
    """Build one optimizer problem with a backend-internal absorption upper bound."""
    graph = compile_line_graph(workspace)
    wavelength = np.asarray(workspace.spectrum.valid_wavelength, dtype=float)
    data = np.asarray(workspace.spectrum.valid_flux, dtype=float)
    error = np.asarray(workspace.spectrum.valid_error, dtype=float)
    continuum_design = workspace.continuum.design(wavelength)
    flux_source_ids = _ordered_flux_sources(graph)
    center_source_ids = tuple(graph.center_sources)
    fwhm_source_ids = tuple(graph.fwhm_sources)
    n_continuum = continuum_design.shape[1]
    offsets = ParameterOffsets(
        continuum=0,
        flux=n_continuum,
        center=n_continuum + len(flux_source_ids),
        fwhm=n_continuum + len(flux_source_ids) + len(center_source_ids),
    )

    continuum_initial = _continuum_initial(continuum_design, data, error)
    continuum_lower = np.full(n_continuum, -inf)
    continuum_upper = np.full(n_continuum, inf)
    continuum_scale = _continuum_scale(continuum_design, data, error)

    flux_lower: list[float] = []
    flux_upper: list[float] = []
    flux_initial: list[float] = []
    flux_scale: list[float] = []
    for source_id in flux_source_ids:
        handle = graph.handle_by_line[source_id]
        lower, upper = flux_bounds(handle.line)
        if handle.contribution == "absorption" and handle.line.flux_rule.is_free:
            upper = _absorption_flux_upper(
                wavelength,
                data,
                error,
                handle=handle,
                fwhm_upper_kms=_expression_upper(graph.fwhm_expressions[handle.index]),
                resolving_power=workspace.spectrum.resolving_power,
                multiplier=absorption_bound_multiplier,
            )
        scale = _source_flux_scale(
            wavelength,
            data,
            error,
            handle=handle,
            resolving_power=workspace.spectrum.resolving_power,
        )
        flux_lower.append(lower)
        flux_upper.append(upper)
        flux_initial.append(_bounded_initial(0.25 * scale, lower, upper))
        flux_scale.append(max(scale, np.finfo(float).tiny))

    center_specs = tuple(graph.center_sources.values())
    fwhm_specs = tuple(graph.fwhm_sources.values())
    center_lower = [spec.lower for spec in center_specs]
    center_upper = [spec.upper for spec in center_specs]
    center_initial = [spec.initial for spec in center_specs]
    center_scale = [max((spec.upper - spec.lower) / 4.0, abs(spec.initial), 1.0) for spec in center_specs]
    fwhm_lower = [spec.lower for spec in fwhm_specs]
    fwhm_upper = [spec.upper for spec in fwhm_specs]
    fwhm_initial = [spec.initial for spec in fwhm_specs]
    fwhm_scale = [max((spec.upper - spec.lower) / 4.0, abs(spec.initial), 1.0) for spec in fwhm_specs]

    lower = np.concatenate((continuum_lower, flux_lower, center_lower, fwhm_lower)).astype(float)
    upper = np.concatenate((continuum_upper, flux_upper, center_upper, fwhm_upper)).astype(float)
    initial = np.concatenate((continuum_initial, flux_initial, center_initial, fwhm_initial)).astype(float)
    scale = np.concatenate((continuum_scale, flux_scale, center_scale, fwhm_scale)).astype(float)
    return MLEProblem(
        workspace=workspace,
        graph=graph,
        wavelength=wavelength,
        data=data,
        error=error,
        continuum_design=continuum_design,
        flux_source_ids=flux_source_ids,
        center_source_ids=center_source_ids,
        fwhm_source_ids=fwhm_source_ids,
        offsets=offsets,
        lower=lower,
        upper=upper,
        initial=initial,
        scale=scale,
        absorption_bound_multiplier=absorption_bound_multiplier,
    )


def _ordered_flux_sources(graph: CompiledLineGraph) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for expression in graph.flux_expressions:
        for source_id in expression.terms:
            seen.setdefault(source_id, None)
    return tuple(seen)


def _continuum_initial(design: np.ndarray, data: np.ndarray, error: np.ndarray) -> np.ndarray:
    weighted_design = design / error[:, None]
    weighted_data = data / error
    try:
        value, *_ = np.linalg.lstsq(weighted_design, weighted_data, rcond=None)
    except np.linalg.LinAlgError:
        value = np.zeros(design.shape[1], dtype=float)
        value[0] = float(np.median(data))
    return np.asarray(value, dtype=float)


def _continuum_scale(design: np.ndarray, data: np.ndarray, error: np.ndarray) -> np.ndarray:
    amplitude = max(float(np.ptp(data)), float(np.median(error)), float(np.max(np.abs(data))) * 0.1)
    column_size = np.max(np.abs(design), axis=0)
    return amplitude / np.maximum(column_size, np.finfo(float).tiny)


def _source_flux_scale(
    wavelength: np.ndarray,
    data: np.ndarray,
    error: np.ndarray,
    *,
    handle: LineHandle,
    resolving_power: float | None,
) -> float:
    preferred_fwhm = 200.0 if handle.component == "narrow" else 1200.0
    template = profile_template(
        wavelength,
        handle=handle,
        center=handle.observed_wavelength,
        fwhm_kms=preferred_fwhm,
        resolving_power=resolving_power,
    )
    amplitude = max(float(np.ptp(data)), float(np.median(error)))
    return amplitude / float(np.max(template))


def _absorption_flux_upper(
    wavelength: np.ndarray,
    data: np.ndarray,
    error: np.ndarray,
    *,
    handle: LineHandle,
    fwhm_upper_kms: float,
    resolving_power: float | None,
    multiplier: float,
) -> float:
    if multiplier <= 0:
        raise ValueError("absorption_bound_multiplier must be positive.")
    template = profile_template(
        wavelength,
        handle=handle,
        center=handle.observed_wavelength,
        fwhm_kms=fwhm_upper_kms,
        resolving_power=resolving_power,
    )
    amplitude = float(np.ptp(data))
    if amplitude <= 0.0:
        amplitude = float(np.median(error))
    return multiplier * amplitude / float(np.max(template))


def _expression_upper(expression: ParameterExpression) -> float:
    value = expression.fixed
    for source_id, coefficient in expression.terms.items():
        spec = expression.specs[source_id]
        value += coefficient * (spec.upper if coefficient >= 0.0 else spec.lower)
    return value


def _bounded_initial(value: float, lower: float, upper: float) -> float:
    if np.isfinite(upper):
        value = min(value, lower + 0.25 * (upper - lower))
    return max(value, lower + max(abs(lower) * 1e-8, np.finfo(float).tiny))
