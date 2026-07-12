"""Build PyMC models from prepared spectrum workspaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose, log, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    C_KMS,
    compile_line_graph,
    contribution_sign,
)
from noobfriend.inference.spectrum.workspace.mcmc.priors import (
    CONTINUUM_PRIOR_SIGMA,
    MCMCPriors,
    build_flux_source_priors,
    build_translated_priors,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.compiler import (
        CompiledLineGraph,
        FluxExpression,
        ParameterExpression,
    )


_GAUSSIAN_FWHM_TO_SIGMA = 2.0 * sqrt(2.0 * log(2.0))
_SQRT2 = sqrt(2.0)


@dataclass(frozen=True, slots=True)
class LineVariableNames:
    """Posterior variable names for one line's physical parameters."""

    flux: str
    fwhm: str
    center: str
    delta_v_kms: str


@dataclass(frozen=True, slots=True)
class MCMCModelMetadata:
    """Metadata needed to materialize physical results after sampling."""

    model_scale: float
    flux_amplitude: float
    wavelength_scale: float
    line_variables: dict[int, LineVariableNames]
    continuum_variables: tuple[str, ...]
    diagnostic_variables: tuple[str, ...]
    priors: MCMCPriors
    reparameterized_flux_pairs: tuple[tuple[str, str], ...]
    effective_fwhm_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuiltMCMCModel:
    """One built PyMC model plus result-materialization metadata."""

    model: Any
    metadata: MCMCModelMetadata


def build_workspace_model(workspace: NoobFitWorkspace) -> BuiltMCMCModel:
    """Return a sampling-ready PyMC model without using an MLE starting point."""
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Spectrum MCMC requires the 'mcmc' optional dependency."
        ) from error

    graph = compile_line_graph(workspace)
    _validate_profiles(workspace)
    wavelength = np.asarray(workspace.spectrum.valid_wavelength, dtype=float)
    data = np.asarray(workspace.spectrum.valid_flux, dtype=float)
    error = np.asarray(workspace.spectrum.valid_error, dtype=float)
    flux_amplitude = max(float(np.ptp(data)), float(np.median(error)))
    model_scale = max(
        flux_amplitude, float(np.median(np.abs(data))), float(np.median(error))
    )
    wavelength_scale = max(float(np.ptp(wavelength)) / 2.0, 1.0)

    flux_source_ids = tuple(
        dict.fromkeys(
            source_id
            for expression in graph.flux_expressions
            for source_id in expression.terms
        )
    )
    flux_sources = build_flux_source_priors(
        workspace, graph, flux_source_ids, flux_amplitude
    )
    prior_by_source = {prior.source_id: prior for prior in flux_sources}
    flux_pairs = _find_flux_pairs(graph, flux_source_ids)
    paired_source_ids = {source_id for pair in flux_pairs for source_id in pair}
    line_variables: dict[int, LineVariableNames] = {}
    continuum_variables: list[str] = []
    effective_sources: list[str] = []

    with pm.Model() as model:
        continuum = pt.zeros_like(pt.as_tensor_variable(wavelength))
        for degree, parameter_name in enumerate(workspace.continuum.parameter_names):
            raw = pm.Normal(
                f"continuum__{parameter_name}__normalized",
                mu=0.0,
                sigma=CONTINUUM_PRIOR_SIGMA,
            )
            physical_name = f"continuum__{parameter_name}"
            physical = pm.Deterministic(
                physical_name,
                raw * model_scale / wavelength_scale**degree,
            )
            continuum_variables.append(physical_name)
            continuum = (
                continuum
                + physical
                * (wavelength - workspace.continuum.lambda_0) ** degree
                / model_scale
            )

        flux_values: dict[int, Any] = {}
        for pair_index, (emission_id, absorption_id) in enumerate(flux_pairs):
            net = pm.Normal(
                f"flux_pair__{pair_index}__net_raw", mu=0.0, sigma=1.0 / _SQRT2
            )
            cancellation = pm.Weibull(
                f"flux_pair__{pair_index}__cancellation_raw",
                alpha=2.0,
                beta=_SQRT2,
            )
            radius = pt.sqrt(net**2 + cancellation**2)
            pm.Potential(f"flux_pair__{pair_index}__jacobian", -pt.log(radius))
            emission_raw = (radius + net) / _SQRT2
            absorption_raw = (radius - net) / _SQRT2
            emission_prior = prior_by_source[emission_id]
            absorption_prior = prior_by_source[absorption_id]
            if emission_prior.scale is None or absorption_prior.scale is None:
                raise RuntimeError("paired free flux prior is missing its scale.")
            flux_values[emission_id] = emission_raw * emission_prior.scale
            flux_values[absorption_id] = absorption_raw * absorption_prior.scale

        for source_id in flux_source_ids:
            if source_id in paired_source_ids:
                continue
            handle = graph.handle_by_line[source_id]
            prior = prior_by_source[source_id]
            variable_name = f"source__{handle.index}__{_safe_name(handle.id)}__flux_raw"
            if prior.family == "uniform":
                if prior.bounds is None:
                    raise RuntimeError("bounded flux prior is missing bounds.")
                lower, upper = prior.bounds
                flux_values[source_id] = (
                    pm.Uniform(
                        variable_name,
                        lower=lower / model_scale,
                        upper=upper / model_scale,
                    )
                    * model_scale
                )
            else:
                if prior.scale is None:
                    raise RuntimeError("free flux prior is missing its scale.")
                flux_values[source_id] = (
                    pm.HalfNormal(variable_name, sigma=1.0) * prior.scale
                )

        center_values: dict[int, Any] = {}
        for source_id, spec in graph.center_sources.items():
            handle = graph.handle_by_line[source_id]
            center_values[source_id] = pm.Uniform(
                f"source__{handle.index}__{_safe_name(handle.id)}__delta_v_kms",
                lower=spec.lower,
                upper=spec.upper,
            )

        fwhm_values: dict[int, Any] = {}
        instrumental_fwhm = (
            None
            if workspace.spectrum.resolving_power is None
            else C_KMS / workspace.spectrum.resolving_power
        )
        for source_id, spec in graph.fwhm_sources.items():
            handle = graph.handle_by_line[source_id]
            prefix = f"source__{handle.index}__{_safe_name(handle.id)}"
            if instrumental_fwhm is None:
                log_fwhm = pm.Uniform(
                    f"{prefix}__log_fwhm",
                    lower=log(spec.lower),
                    upper=log(spec.upper),
                )
                fwhm_values[source_id] = pt.exp(log_fwhm)
                continue
            effective_lower = sqrt(spec.lower**2 + instrumental_fwhm**2)
            effective_upper = sqrt(spec.upper**2 + instrumental_fwhm**2)
            log_effective = pm.Uniform(
                f"{prefix}__log_effective_fwhm",
                lower=log(effective_lower),
                upper=log(effective_upper),
            )
            effective = pt.exp(log_effective)
            intrinsic = pt.sqrt(effective**2 - instrumental_fwhm**2)
            pm.Potential(
                f"{prefix}__effective_fwhm_jacobian",
                2.0 * log_effective - 2.0 * pt.log(intrinsic),
            )
            fwhm_values[source_id] = intrinsic
            effective_sources.append(handle.id)

        line_total = pt.zeros_like(pt.as_tensor_variable(wavelength))
        for handle, flux_expression, center_expression, fwhm_expression in zip(
            workspace.handles,
            graph.flux_expressions,
            graph.center_expressions,
            graph.fwhm_expressions,
            strict=True,
        ):
            flux = _symbolic_expression(flux_expression, flux_values, pt)
            velocity = _symbolic_expression(center_expression, center_values, pt)
            fwhm = _symbolic_expression(fwhm_expression, fwhm_values, pt)
            center = handle.observed_wavelength * (1.0 + velocity / C_KMS)
            effective_fwhm = fwhm
            if instrumental_fwhm is not None:
                effective_fwhm = pt.sqrt(fwhm**2 + instrumental_fwhm**2)
            template = _symbolic_profile(
                wavelength,
                profile=handle.profile,
                center=center,
                fwhm_kms=effective_fwhm,
                pt=pt,
            )
            line_total = (
                line_total + contribution_sign(handle) * flux * template / model_scale
            )

            prefix = f"line__{handle.index}__{_safe_name(handle.id)}"
            names = LineVariableNames(
                flux=f"{prefix}__flux",
                fwhm=f"{prefix}__fwhm",
                center=f"{prefix}__center",
                delta_v_kms=f"{prefix}__delta_v_kms",
            )
            line_variables[id(handle.line)] = names
            pm.Deterministic(names.flux, flux)
            pm.Deterministic(names.fwhm, fwhm)
            pm.Deterministic(names.center, center)
            pm.Deterministic(names.delta_v_kms, velocity)

        pm.Normal(
            "observed_flux_normalized",
            mu=continuum + line_total,
            sigma=error / model_scale,
            observed=data / model_scale,
        )

    translated_priors = build_translated_priors(
        workspace,
        graph,
        flux_sources,
        model_scale=model_scale,
        wavelength_scale=wavelength_scale,
    )
    metadata = MCMCModelMetadata(
        model_scale=model_scale,
        flux_amplitude=flux_amplitude,
        wavelength_scale=wavelength_scale,
        line_variables=line_variables,
        continuum_variables=tuple(continuum_variables),
        diagnostic_variables=tuple(variable.name for variable in model.free_RVs),
        priors=translated_priors,
        reparameterized_flux_pairs=tuple(
            (graph.handle_by_line[emission].id, graph.handle_by_line[absorption].id)
            for emission, absorption in flux_pairs
        ),
        effective_fwhm_sources=tuple(effective_sources),
    )
    return BuiltMCMCModel(model=model, metadata=metadata)


def _validate_profiles(workspace: NoobFitWorkspace) -> None:
    if workspace.spectrum.resolving_power is None:
        return
    if any(handle.profile != "gaussian" for handle in workspace.handles):
        raise NotImplementedError(
            "resolving_power is only implemented for gaussian line profiles."
        )


def _find_flux_pairs(
    graph: CompiledLineGraph,
    source_ids: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    free_sources = [
        source_id
        for source_id in source_ids
        if graph.handle_by_line[source_id].line.flux_rule.is_free
    ]
    emissions = [
        source_id
        for source_id in free_sources
        if graph.handle_by_line[source_id].contribution == "emission"
    ]
    absorptions = [
        source_id
        for source_id in free_sources
        if graph.handle_by_line[source_id].contribution == "absorption"
    ]
    used_absorptions: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for emission_id in emissions:
        emission = graph.handle_by_line[emission_id]
        for absorption_id in absorptions:
            if absorption_id in used_absorptions:
                continue
            absorption = graph.handle_by_line[absorption_id]
            if (
                emission.component != absorption.component
                or emission.profile != absorption.profile
            ):
                continue
            if not isclose(
                emission.observed_wavelength,
                absorption.observed_wavelength,
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                continue
            pairs.append((emission_id, absorption_id))
            used_absorptions.add(absorption_id)
            break
    return tuple(pairs)


def _symbolic_expression(
    expression: FluxExpression | ParameterExpression, values: dict[int, Any], pt: Any
) -> Any:
    output = pt.as_tensor_variable(float(expression.fixed))
    for source_id, coefficient in expression.terms.items():
        output = output + float(coefficient) * values[source_id]
    return output


def _symbolic_profile(
    wavelength: np.ndarray, *, profile: str, center: Any, fwhm_kms: Any, pt: Any
) -> Any:
    fwhm_wavelength = center * fwhm_kms / C_KMS
    if profile == "gaussian":
        sigma = fwhm_wavelength / _GAUSSIAN_FWHM_TO_SIGMA
        return pt.exp(-0.5 * ((wavelength - center) / sigma) ** 2) / (
            sigma * sqrt(2.0 * pi)
        )
    if profile == "lorentzian":
        gamma = 0.5 * fwhm_wavelength
        return (gamma / pi) / ((wavelength - center) ** 2 + gamma**2)
    if profile == "exponential":
        scale = fwhm_wavelength / (2.0 * log(2.0))
        return pt.exp(-pt.abs(wavelength - center) / scale) / (2.0 * scale)
    raise ValueError(f"Unsupported profile: {profile!r}.")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "line"
