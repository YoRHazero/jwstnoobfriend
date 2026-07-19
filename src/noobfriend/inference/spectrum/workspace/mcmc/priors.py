"""Translated physical priors exposed by spectrum MCMC results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from html import escape
from math import sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    BASE_SHAPE,
    C_KMS,
    contribution_sign,
    evaluate_expression,
    flux_bounds,
    profile_template,
)
from noobfriend.inference.spectrum.workspace.compiler.expressions import (
    line_kernels_kms,
)
from noobfriend.inference.spectrum.workspace.mcmc.posterior import CONTINUUM_COMPONENT

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.compiler import (
        CompiledLineGraph,
        ParameterExpression,
    )


FLUX_PRIOR_MULTIPLIER = 5.0
CONTINUUM_PRIOR_SIGMA = 5.0
# Half-normal scale (normalized units) for the per-coefficient pooling spread in
# a multi-frame continuum; validated on GN-4014 (tau reproduces inter-frame scatter).
CONTINUUM_TAU_SIGMA = 0.5


@dataclass(frozen=True, slots=True)
class FluxSourcePrior:
    """Internal physical prior required to build one independent flux source."""

    source_id: int
    line_id: str
    family: str
    reference_flux: float
    scale: float | None
    bounds: tuple[float, float] | None


@dataclass(frozen=True, slots=True, repr=False)
class MCMCParameterPrior:
    """Translated physical prior for one component parameter."""

    name: str
    family: str
    bounds: tuple[float, float] | None = None
    location: float | None = None
    scale: float | None = None
    value: float | None = None
    reference: float | None = None
    target: str | None = None
    ratio: float | None = None

    def __repr__(self) -> str:
        """Return a compact prior representation."""
        details = self._details()
        suffix = "" if not details else f", {details}"
        return f"MCMCParameterPrior(name={self.name!r}, family={self.family!r}{suffix})"

    def _repr_html_(self) -> str:
        """Return a notebook HTML representation."""
        return f"""<table class="noob-mcmc-parameter-prior">
  <thead><tr><th>Parameter</th><th>Family</th><th>Definition</th></tr></thead>
  <tbody><tr>
    <td><code>{escape(self.name)}</code></td>
    <td>{escape(self.family)}</td>
    <td>{escape(self._details() or "—")}</td>
  </tr></tbody>
</table>"""

    def _details(self) -> str:
        values: list[str] = []
        if self.bounds is not None:
            values.append(f"bounds={self.bounds!r}")
        if self.location is not None:
            values.append(f"location={self.location:.8g}")
        if self.scale is not None:
            values.append(f"scale={self.scale:.8g}")
        if self.value is not None:
            values.append(f"value={self.value:.8g}")
        if self.reference is not None:
            values.append(f"reference={self.reference:.8g}")
        if self.target is not None:
            values.append(f"target={self.target!r}")
        if self.ratio is not None:
            values.append(f"ratio={self.ratio:.8g}")
        return ", ".join(values)


@dataclass(frozen=True, slots=True, repr=False)
class MCMCComponentPrior(Mapping[str, MCMCParameterPrior]):
    """Immutable translated prior mapping for one model component."""

    name: str
    _values: Mapping[str, MCMCParameterPrior]

    def __post_init__(self) -> None:
        """Copy values into an immutable, insertion-ordered mapping."""
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    @property
    def parameters(self) -> tuple[str, ...]:
        """Valid parameter keys in display order."""
        return tuple(self._values)

    def __getitem__(self, parameter: str) -> MCMCParameterPrior:
        """Return a parameter prior and report valid keys on failure."""
        try:
            return self._values[parameter]
        except KeyError as error:
            valid = ", ".join(self.parameters)
            raise KeyError(
                f"unknown prior parameter {parameter!r} for component {self.name!r}; "
                f"valid parameters: {valid}"
            ) from error

    def __iter__(self) -> Iterator[str]:
        """Iterate over valid parameter keys."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._values)

    def __repr__(self) -> str:
        """Represent this component together with its valid keys."""
        return f"MCMCComponentPrior(name={self.name!r}, parameters={self.parameters!r})"

    def _repr_html_(self) -> str:
        """Return a notebook table of translated priors."""
        rows = "\n".join(
            f"""    <tr>
      <td><code>{escape(name)}</code></td>
      <td>{escape(value.family)}</td>
      <td>{escape(value._details() or "—")}</td>
    </tr>"""
            for name, value in self._values.items()
        )
        return f"""<section class="noob-mcmc-component-prior">
  <h3><code>{escape(self.name)}</code></h3>
  <p><strong>parameters:</strong> {escape(", ".join(self.parameters))}</p>
  <table>
    <thead><tr><th>Parameter</th><th>Family</th><th>Definition</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


@dataclass(frozen=True, slots=True, repr=False)
class MCMCPriors(Mapping[str, MCMCComponentPrior]):
    """Immutable translated priors with the same component interface as posterior."""

    _values: Mapping[str, MCMCComponentPrior]

    def __post_init__(self) -> None:
        """Copy components into an immutable, insertion-ordered mapping."""
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    @property
    def components(self) -> tuple[str, ...]:
        """Valid component keys in workspace order, followed by continuum."""
        return tuple(self._values)

    def __getitem__(self, component: str) -> MCMCComponentPrior:
        """Return component priors and report valid keys on failure."""
        try:
            return self._values[component]
        except KeyError as error:
            valid = ", ".join(self.components)
            raise KeyError(
                f"unknown prior component {component!r}; valid components: {valid}"
            ) from error

    def __iter__(self) -> Iterator[str]:
        """Iterate over valid component keys."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._values)

    def __repr__(self) -> str:
        """Represent translated priors together with all valid component keys."""
        return f"MCMCPriors(components={self.components!r})"

    def _repr_html_(self) -> str:
        """Return a notebook table of components and their valid parameters."""
        rows = "\n".join(
            f"""    <tr>
      <td><code>{escape(name)}</code></td>
      <td>{escape(", ".join(component.parameters))}</td>
    </tr>"""
            for name, component in self._values.items()
        )
        return f"""<section class="noob-mcmc-priors">
  <h2>Translated MCMC priors</h2>
  <p><strong>components:</strong> {escape(", ".join(self.components))}</p>
  <table>
    <thead><tr><th>Component</th><th>Parameters</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def build_flux_source_priors(
    workspace: NoobFitWorkspace,
    graph: CompiledLineGraph,
    source_ids: tuple[int, ...],
    data_amplitude: float,
    *,
    wavelength: np.ndarray,
    resolving_power: float | None,
) -> tuple[FluxSourcePrior, ...]:
    """Resolve data-scaled priors for independent flux sources.

    ``wavelength`` and ``resolving_power`` describe the (concatenated) fitting
    pixels; the shared line flux scale is set from the peak response there.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    center_values = {
        source_id: spec.initial for source_id, spec in graph.center_sources.items()
    }
    fwhm_values = {
        source_id: sqrt(spec.lower * spec.upper)
        for source_id, spec in graph.fwhm_sources.items()
    }
    templates = {source_id: np.zeros_like(wavelength) for source_id in source_ids}
    for handle, flux_expression, center_expression, shapes in zip(
        workspace.handles,
        graph.flux_expressions,
        graph.center_expressions,
        graph.shape_expressions,
        strict=True,
    ):
        velocity = evaluate_expression(center_expression, center_values)
        fwhm = evaluate_expression(shapes[BASE_SHAPE], fwhm_values)
        center = handle.observed_wavelength * (1.0 + velocity / C_KMS)
        template = contribution_sign(handle) * profile_template(
            wavelength,
            handle=handle,
            center=center,
            fwhm_kms=fwhm,
            resolving_power=resolving_power,
            kernels=line_kernels_kms(handle, shapes),
        )
        for source_id, coefficient in flux_expression.terms.items():
            templates[source_id] += coefficient * template

    priors: list[FluxSourcePrior] = []
    for source_id in source_ids:
        handle = graph.handle_by_line[source_id]
        peak = float(np.max(np.abs(templates[source_id])))
        if not np.isfinite(peak) or peak <= 0.0:
            raise RuntimeError(
                f"flux source {handle.id!r} has no finite reference response."
            )
        reference = data_amplitude / peak
        if handle.line.flux_rule.is_bounded:
            priors.append(
                FluxSourcePrior(
                    source_id=source_id,
                    line_id=handle.id,
                    family="uniform",
                    reference_flux=reference,
                    scale=None,
                    bounds=flux_bounds(handle.line),
                )
            )
        else:
            priors.append(
                FluxSourcePrior(
                    source_id=source_id,
                    line_id=handle.id,
                    family="halfnormal",
                    reference_flux=reference,
                    scale=FLUX_PRIOR_MULTIPLIER * reference,
                    bounds=None,
                )
            )
    return tuple(priors)


def build_translated_priors(
    workspace: NoobFitWorkspace,
    graph: CompiledLineGraph,
    flux_sources: tuple[FluxSourcePrior, ...],
    *,
    model_scale: float,
    wavelength_scale: float,
) -> MCMCPriors:
    """Translate declarations into line and continuum physical prior metadata."""
    flux_by_source = {prior.source_id: prior for prior in flux_sources}
    components: dict[str, MCMCComponentPrior] = {}
    for handle, center_expression, shapes in zip(
        workspace.handles,
        graph.center_expressions,
        graph.shape_expressions,
        strict=True,
    ):
        values = {
            "flux": _line_flux_prior(handle, graph, flux_by_source),
            "fwhm": _line_fwhm_prior(handle, graph, shapes[BASE_SHAPE]),
            "center": _line_center_prior(handle, graph, center_expression),
            "delta_v_kms": _line_delta_v_prior(handle, graph, center_expression),
        }
        for kernel in handle.line.kernels:
            for (_, rule), name in zip(kernel.rules, kernel.shape_names(), strict=True):
                values[name] = _kernel_shape_prior(
                    name, shapes[name], rule=rule, graph=graph
                )
        components[handle.id] = MCMCComponentPrior(name=handle.id, _values=values)
    components[CONTINUUM_COMPONENT] = MCMCComponentPrior(
        name=CONTINUUM_COMPONENT,
        _values=_continuum_priors(
            workspace, model_scale=model_scale, wavelength_scale=wavelength_scale
        ),
    )
    return MCMCPriors(components)


def _continuum_priors(
    workspace: NoobFitWorkspace,
    *,
    model_scale: float,
    wavelength_scale: float,
) -> dict[str, MCMCParameterPrior]:
    """Translate the continuum prior, honoring frame count and sharing mode.

    A single-frame (or ``"shared"``) continuum keeps one normal prior per
    coefficient. ``"pooled"`` exposes the hyperprior on each coefficient's
    mean (normal) and spread (half-normal); ``"independent"`` keeps the
    per-frame normal, described once since it is identical across frames.
    """
    spec = workspace.continuum
    single = workspace.n_frames == 1
    values: dict[str, MCMCParameterPrior] = {}
    for degree, name in enumerate(spec.parameter_names):
        scale = CONTINUUM_PRIOR_SIGMA * model_scale / wavelength_scale**degree
        if single or spec.sharing in ("shared", "independent"):
            values[name] = MCMCParameterPrior(
                name=name, family="normal", location=0.0, scale=scale
            )
            continue
        tau_scale = CONTINUUM_TAU_SIGMA * model_scale / wavelength_scale**degree
        values[f"{name}__mu"] = MCMCParameterPrior(
            name=f"{name}__mu", family="normal", location=0.0, scale=scale
        )
        values[f"{name}__tau"] = MCMCParameterPrior(
            name=f"{name}__tau", family="halfnormal", scale=tau_scale
        )
    return values


def _line_flux_prior(
    handle, graph, sources: Mapping[int, FluxSourcePrior]
) -> MCMCParameterPrior:
    rule = handle.line.flux_rule
    if rule.is_fixed:
        return MCMCParameterPrior(name="flux", family="fixed", value=rule.value)
    if rule.is_ratio:
        target = graph.handle_by_line[id(handle.line.base)].id
        return MCMCParameterPrior(
            name="flux",
            family="ratio",
            target=target,
            ratio=rule.value,
        )
    source = sources[id(handle.line)]
    return MCMCParameterPrior(
        name="flux",
        family=source.family,
        bounds=source.bounds,
        scale=source.scale,
        reference=source.reference_flux if source.family == "halfnormal" else None,
    )


def _line_fwhm_prior(
    handle,
    graph,
    expression: ParameterExpression,
) -> MCMCParameterPrior:
    rule = handle.line.fwhm_rule
    if rule.is_locked:
        return MCMCParameterPrior(
            name="fwhm",
            family="locked",
            target=graph.handle_by_line[id(rule.target)].id,
        )
    if rule.is_fixed:
        return MCMCParameterPrior(name="fwhm", family="fixed", value=expression.fixed)
    spec = next(iter(expression.specs.values()))
    return MCMCParameterPrior(
        name="fwhm",
        family="loguniform",
        bounds=(spec.lower, spec.upper),
    )


def _line_center_prior(
    handle,
    graph,
    expression: ParameterExpression,
) -> MCMCParameterPrior:
    rule = handle.line.center_rule
    if rule.is_locked:
        return MCMCParameterPrior(
            name="center",
            family="locked",
            target=graph.handle_by_line[id(rule.target)].id,
        )
    if rule.is_fixed:
        center = handle.observed_wavelength * (1.0 + expression.fixed / C_KMS)
        return MCMCParameterPrior(name="center", family="fixed", value=center)
    spec = next(iter(expression.specs.values()))
    bounds = (
        handle.observed_wavelength * (1.0 + spec.lower / C_KMS),
        handle.observed_wavelength * (1.0 + spec.upper / C_KMS),
    )
    return MCMCParameterPrior(name="center", family="uniform", bounds=bounds)


def _line_delta_v_prior(
    handle,
    graph,
    expression: ParameterExpression,
) -> MCMCParameterPrior:
    """Translate the center rule into a velocity-offset prior in km/s.

    This mirrors :func:`_line_center_prior` in the velocity coordinate the
    model samples, so the ``delta_v_kms`` posterior has a matching prior.
    """
    rule = handle.line.center_rule
    if rule.is_locked:
        return MCMCParameterPrior(
            name="delta_v_kms",
            family="locked",
            target=graph.handle_by_line[id(rule.target)].id,
        )
    if rule.is_fixed:
        return MCMCParameterPrior(
            name="delta_v_kms", family="fixed", value=expression.fixed
        )
    spec = next(iter(expression.specs.values()))
    return MCMCParameterPrior(
        name="delta_v_kms", family="uniform", bounds=(spec.lower, spec.upper)
    )


def _kernel_shape_prior(
    name: str,
    expression: ParameterExpression,
    *,
    rule,
    graph,
) -> MCMCParameterPrior:
    """Translate one kernel shape rule into prior metadata."""
    if rule.is_locked:
        return MCMCParameterPrior(
            name=name,
            family="locked",
            target=graph.handle_by_line[id(rule.target)].id,
        )
    if not expression.terms:
        return MCMCParameterPrior(name=name, family="fixed", value=expression.fixed)
    spec = next(iter(expression.specs.values()))
    return MCMCParameterPrior(
        name=name, family="loguniform", bounds=(spec.lower, spec.upper)
    )
