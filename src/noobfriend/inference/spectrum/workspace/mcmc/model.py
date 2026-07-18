"""Build PyMC models from prepared spectrum workspaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose, log, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    BASE_SHAPE,
    C_KMS,
    compile_line_graph,
    contribution_sign,
)
from noobfriend.inference.spectrum.workspace.mcmc.priors import (
    CONTINUUM_PRIOR_SIGMA,
    CONTINUUM_TAU_SIGMA,
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
    """Posterior variable names for one line's physical parameters.

    ``shapes`` lists extra named shape parameters (convolution-kernel
    widths) as ``(label, variable_name)`` pairs in declaration order.
    """

    flux: str
    fwhm: str
    center: str
    delta_v_kms: str
    shapes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ContinuumVariableSpec:
    """How one continuum coefficient is exposed in the posterior.

    ``value_variable`` names the physical Deterministic (a scalar for a
    single-frame or ``"shared"`` continuum, a frame-indexed vector otherwise).
    ``mu_variable``/``tau_variable`` name the pooling hyperparameters and are
    set only for ``"pooled"`` continua.
    """

    parameter_name: str
    value_variable: str
    per_frame: bool
    mu_variable: str | None = None
    tau_variable: str | None = None


@dataclass(frozen=True, slots=True)
class MCMCModelMetadata:
    """Metadata needed to materialize physical results after sampling."""

    model_scale: float
    flux_amplitude: float
    wavelength_scale: float
    line_variables: dict[int, LineVariableNames]
    continuum_layout: tuple[ContinuumVariableSpec, ...]
    frame_ids: tuple[str, ...]
    frame_index: np.ndarray
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
    frames = workspace.spectra
    n_frames = len(frames)

    wavelength_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    frame_index_parts: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        frame_wavelength = np.asarray(frame.valid_wavelength, dtype=float)
        wavelength_parts.append(frame_wavelength)
        data_parts.append(np.asarray(frame.valid_flux, dtype=float))
        error_parts.append(np.asarray(frame.valid_error, dtype=float))
        frame_index_parts.append(np.full(frame_wavelength.size, index, dtype=int))
    wavelength = np.concatenate(wavelength_parts)
    data = np.concatenate(data_parts)
    error = np.concatenate(error_parts)
    frame_index = np.concatenate(frame_index_parts)

    flux_amplitude = max(float(np.ptp(data)), float(np.median(error)))
    model_scale = max(
        flux_amplitude, float(np.median(np.abs(data))), float(np.median(error))
    )
    wavelength_scale = max(float(np.ptp(wavelength)) / 2.0, 1.0)
    # Every frame shares one resolving power (enforced by NoobSpectrumSet), so a
    # single instrumental width and its effective-FWHM reparameterization apply.
    resolving_power = frames[0].resolving_power
    instrumental_fwhm = None if resolving_power is None else C_KMS / resolving_power

    flux_source_ids = tuple(
        dict.fromkeys(
            source_id
            for expression in graph.flux_expressions
            for source_id in expression.terms
        )
    )
    flux_sources = build_flux_source_priors(
        workspace,
        graph,
        flux_source_ids,
        flux_amplitude,
        wavelength=wavelength,
        resolving_power=resolving_power,
    )
    prior_by_source = {prior.source_id: prior for prior in flux_sources}
    flux_pairs = _find_flux_pairs(graph, flux_source_ids)
    paired_source_ids = {source_id for pair in flux_pairs for source_id in pair}
    line_variables: dict[int, LineVariableNames] = {}
    effective_sources: list[str] = []

    coords = None if n_frames == 1 else {"frame": list(workspace.frame_ids)}
    with pm.Model(coords=coords) as model:
        continuum, continuum_layout = _build_continuum(
            pm,
            pt,
            workspace,
            wavelength=wavelength,
            frame_index=frame_index,
            model_scale=model_scale,
            wavelength_scale=wavelength_scale,
            n_frames=n_frames,
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

        shape_values: dict[str, dict[int, Any]] = {
            name: {} for name in graph.shape_sources
        }
        for shape_name, sources in graph.shape_sources.items():
            for source_id, spec in sources.items():
                handle = graph.handle_by_line[source_id]
                prefix = f"source__{handle.index}__{_safe_name(handle.id)}"
                if shape_name != BASE_SHAPE or instrumental_fwhm is None:
                    log_value = pm.Uniform(
                        f"{prefix}__log_{shape_name}",
                        lower=log(spec.lower),
                        upper=log(spec.upper),
                    )
                    shape_values[shape_name][source_id] = pt.exp(log_value)
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
                shape_values[shape_name][source_id] = intrinsic
                effective_sources.append(handle.id)

        line_total = pt.zeros_like(pt.as_tensor_variable(wavelength))
        for handle, flux_expression, center_expression, shapes in zip(
            workspace.handles,
            graph.flux_expressions,
            graph.center_expressions,
            graph.shape_expressions,
            strict=True,
        ):
            flux = _symbolic_expression(flux_expression, flux_values, pt)
            velocity = _symbolic_expression(center_expression, center_values, pt)
            fwhm = _symbolic_expression(
                shapes[BASE_SHAPE], shape_values[BASE_SHAPE], pt
            )
            center = handle.observed_wavelength * (1.0 + velocity / C_KMS)
            effective_fwhm = fwhm
            if instrumental_fwhm is not None:
                effective_fwhm = pt.sqrt(fwhm**2 + instrumental_fwhm**2)
            shape_tensors = tuple(
                (name, _symbolic_expression(shapes[name], shape_values[name], pt))
                for kernel in handle.line.kernels
                for name in kernel.shape_names()
            )
            tensor_by_name = dict(shape_tensors)
            kernel_triples = tuple(
                (
                    kernel.kind,
                    tensor_by_name[kernel.shape_name("fwhm")],
                    tensor_by_name[kernel.shape_name("fraction")],
                )
                for kernel in handle.line.kernels
            )
            template = _symbolic_profile(
                wavelength,
                profile=handle.profile,
                center=center,
                fwhm_kms=effective_fwhm,
                pt=pt,
                kernels=kernel_triples,
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
                shapes=tuple((name, f"{prefix}__{name}") for name, _ in shape_tensors),
            )
            line_variables[id(handle.line)] = names
            pm.Deterministic(names.flux, flux)
            pm.Deterministic(names.fwhm, fwhm)
            pm.Deterministic(names.center, center)
            pm.Deterministic(names.delta_v_kms, velocity)
            for (label, variable_name), (_, value) in zip(
                names.shapes, shape_tensors, strict=True
            ):
                pm.Deterministic(variable_name, value)

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
        continuum_layout=tuple(continuum_layout),
        frame_ids=workspace.frame_ids,
        frame_index=frame_index,
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
    if workspace.spectra[0].resolving_power is None:
        return
    if any(handle.profile != "gaussian" for handle in workspace.handles):
        raise NotImplementedError(
            "resolving_power is only implemented for gaussian line profiles."
        )


def _build_continuum(
    pm: Any,
    pt: Any,
    workspace: NoobFitWorkspace,
    *,
    wavelength: np.ndarray,
    frame_index: np.ndarray,
    model_scale: float,
    wavelength_scale: float,
    n_frames: int,
) -> tuple[Any, list[ContinuumVariableSpec]]:
    """Build the per-pixel continuum tensor and its posterior layout.

    A single-frame or ``"shared"`` continuum keeps one scalar coefficient per
    degree (byte-identical to the single-spectrum model). ``"independent"``
    gives each frame a free coefficient; ``"pooled"`` draws each frame's
    coefficient from a non-centered ``mu + tau * z`` hyperprior.
    """
    spec = workspace.continuum
    powers = [
        (wavelength - spec.lambda_0) ** degree for degree in range(spec.order + 1)
    ]
    continuum = pt.zeros_like(pt.as_tensor_variable(wavelength))
    layout: list[ContinuumVariableSpec] = []
    scalar = n_frames == 1 or spec.sharing == "shared"
    for degree, name in enumerate(spec.parameter_names):
        factor = model_scale / wavelength_scale**degree
        if scalar:
            raw = pm.Normal(
                f"continuum__{name}__normalized", mu=0.0, sigma=CONTINUUM_PRIOR_SIGMA
            )
            physical = pm.Deterministic(f"continuum__{name}", raw * factor)
            continuum = continuum + physical * powers[degree] / model_scale
            layout.append(
                ContinuumVariableSpec(
                    parameter_name=name,
                    value_variable=f"continuum__{name}",
                    per_frame=False,
                )
            )
            continue

        if spec.sharing == "independent":
            raw = pm.Normal(
                f"continuum__{name}__normalized",
                mu=0.0,
                sigma=CONTINUUM_PRIOR_SIGMA,
                dims="frame",
            )
            physical = pm.Deterministic(
                f"continuum__{name}", raw * factor, dims="frame"
            )
            layout.append(
                ContinuumVariableSpec(
                    parameter_name=name,
                    value_variable=f"continuum__{name}",
                    per_frame=True,
                )
            )
        else:
            mu = pm.Normal(
                f"continuum__{name}__mu_normalized", mu=0.0, sigma=CONTINUUM_PRIOR_SIGMA
            )
            tau = pm.HalfNormal(
                f"continuum__{name}__tau_normalized", sigma=CONTINUUM_TAU_SIGMA
            )
            z = pm.Normal(f"continuum__{name}__z", mu=0.0, sigma=1.0, dims="frame")
            pm.Deterministic(f"continuum__{name}__mu", mu * factor)
            pm.Deterministic(f"continuum__{name}__tau", tau * factor)
            physical = pm.Deterministic(
                f"continuum__{name}", (mu + tau * z) * factor, dims="frame"
            )
            layout.append(
                ContinuumVariableSpec(
                    parameter_name=name,
                    value_variable=f"continuum__{name}",
                    per_frame=True,
                    mu_variable=f"continuum__{name}__mu",
                    tau_variable=f"continuum__{name}__tau",
                )
            )
        continuum = continuum + physical[frame_index] * powers[degree] / model_scale
    return continuum, layout


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
    wavelength: np.ndarray,
    *,
    profile: str,
    center: Any,
    fwhm_kms: Any,
    pt: Any,
    kernels: tuple[tuple[str, Any, Any], ...] = (),
) -> Any:
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
        delta = wavelength - center
        branches: list[tuple[Any, Any, Any | None]] = [(1.0, fwhm_kms, None)]
        for kind, width, fraction in kernels:
            updated: list[tuple[Any, Any, Any | None]] = []
            for weight, gauss_fwhm, laplace_fwhm in branches:
                updated.append((weight * (1.0 - fraction), gauss_fwhm, laplace_fwhm))
                if kind == "gaussian":
                    updated.append(
                        (
                            weight * fraction,
                            pt.sqrt(gauss_fwhm**2 + width**2),
                            laplace_fwhm,
                        )
                    )
                else:
                    updated.append((weight * fraction, gauss_fwhm, width))
            branches = updated
        total: Any = 0.0
        for weight, gauss_fwhm, laplace_fwhm in branches:
            sigma = center * gauss_fwhm / C_KMS / _GAUSSIAN_FWHM_TO_SIGMA
            if laplace_fwhm is None:
                total = total + weight * pt.exp(-0.5 * (delta / sigma) ** 2) / (
                    sigma * sqrt(2.0 * pi)
                )
            else:
                scale = center * laplace_fwhm / C_KMS / (2.0 * log(2.0))
                u_plus = (sigma / scale + delta / sigma) / sqrt(2.0)
                u_minus = (sigma / scale - delta / sigma) / sqrt(2.0)
                total = total + weight * (
                    pt.exp(-0.5 * (delta / sigma) ** 2)
                    / (4.0 * scale)
                    * (pt.erfcx(u_plus) + pt.erfcx(u_minus))
                )
        return total

    fwhm_wavelength = center * fwhm_kms / C_KMS
    if profile == "lorentzian":
        gamma = 0.5 * fwhm_wavelength
        return (gamma / pi) / ((wavelength - center) ** 2 + gamma**2)
    if profile == "exponential":
        scale = fwhm_wavelength / (2.0 * log(2.0))
        return pt.exp(-pt.abs(wavelength - center) / scale) / (2.0 * scale)
    raise ValueError(f"Unsupported profile: {profile!r}.")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "line"
