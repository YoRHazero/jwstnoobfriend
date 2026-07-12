"""Unified component and parameter views of an MCMC posterior."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from html import escape
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.mcmc.model import MCMCModelMetadata


CONTINUUM_COMPONENT = "continuum"


@dataclass(frozen=True, slots=True, repr=False)
class MCMCParameterPosterior:
    """Posterior samples and a 94% highest-density interval for one scalar."""

    name: str
    samples: np.ndarray
    median: float
    mean: float
    standard_deviation: float
    hdi_lower: float
    hdi_upper: float

    def __repr__(self) -> str:
        """Return a compact statistical representation."""
        return (
            f"MCMCParameterPosterior(name={self.name!r}, median={self.median:.8g}, "
            f"hdi94=({self.hdi_lower:.8g}, {self.hdi_upper:.8g}), "
            f"shape={self.samples.shape!r})"
        )

    def _repr_html_(self) -> str:
        """Return a notebook HTML representation."""
        return f"""<table class="noob-mcmc-parameter-posterior">
  <thead><tr><th>Parameter</th><th>Median</th><th>Mean</th><th>Std.</th><th>94% HDI</th></tr></thead>
  <tbody><tr>
    <td><code>{escape(self.name)}</code></td>
    <td>{self.median:.8g}</td>
    <td>{self.mean:.8g}</td>
    <td>{self.standard_deviation:.8g}</td>
    <td>[{self.hdi_lower:.8g}, {self.hdi_upper:.8g}]</td>
  </tr></tbody>
</table>"""


@dataclass(frozen=True, slots=True, repr=False)
class MCMCComponentPosterior(Mapping[str, MCMCParameterPosterior]):
    """Immutable posterior parameter mapping for one model component."""

    name: str
    _values: Mapping[str, MCMCParameterPosterior]

    def __post_init__(self) -> None:
        """Copy values into an immutable, insertion-ordered mapping."""
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    @property
    def parameters(self) -> tuple[str, ...]:
        """Valid parameter keys in display order."""
        return tuple(self._values)

    def __getitem__(self, parameter: str) -> MCMCParameterPosterior:
        """Return a parameter posterior and report valid keys on failure."""
        try:
            return self._values[parameter]
        except KeyError as error:
            valid = ", ".join(self.parameters)
            raise KeyError(
                f"unknown parameter {parameter!r} for component {self.name!r}; "
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
        return f"MCMCComponentPosterior(name={self.name!r}, parameters={self.parameters!r})"

    def _repr_html_(self) -> str:
        """Return a notebook table of parameter summaries."""
        rows = "\n".join(
            f"""    <tr>
      <td><code>{escape(name)}</code></td>
      <td>{value.median:.8g}</td>
      <td>[{value.hdi_lower:.8g}, {value.hdi_upper:.8g}]</td>
    </tr>"""
            for name, value in self._values.items()
        )
        return f"""<section class="noob-mcmc-component-posterior">
  <h3><code>{escape(self.name)}</code></h3>
  <p><strong>parameters:</strong> {escape(", ".join(self.parameters))}</p>
  <table>
    <thead><tr><th>Parameter</th><th>Median</th><th>94% HDI</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


@dataclass(frozen=True, slots=True, repr=False)
class MCMCPosterior(Mapping[str, MCMCComponentPosterior]):
    """Immutable mapping of line and continuum component posteriors."""

    _values: Mapping[str, MCMCComponentPosterior]

    def __post_init__(self) -> None:
        """Copy components into an immutable, insertion-ordered mapping."""
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    @property
    def components(self) -> tuple[str, ...]:
        """Valid component keys in workspace order, followed by continuum."""
        return tuple(self._values)

    def __getitem__(self, component: str) -> MCMCComponentPosterior:
        """Return a component posterior and report valid keys on failure."""
        try:
            return self._values[component]
        except KeyError as error:
            valid = ", ".join(self.components)
            raise KeyError(
                f"unknown posterior component {component!r}; valid components: {valid}"
            ) from error

    def __iter__(self) -> Iterator[str]:
        """Iterate over valid component keys."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._values)

    def __repr__(self) -> str:
        """Represent this posterior together with all valid component keys."""
        return f"MCMCPosterior(components={self.components!r})"

    def _repr_html_(self) -> str:
        """Return a notebook table of components and their valid parameters."""
        rows = "\n".join(
            f"""    <tr>
      <td><code>{escape(name)}</code></td>
      <td>{escape(", ".join(component.parameters))}</td>
    </tr>"""
            for name, component in self._values.items()
        )
        return f"""<section class="noob-mcmc-posterior">
  <h2>MCMC posterior</h2>
  <p><strong>components:</strong> {escape(", ".join(self.components))}</p>
  <table>
    <thead><tr><th>Component</th><th>Parameters</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def build_mcmc_posterior(
    workspace: NoobFitWorkspace,
    idata: Any,
    metadata: MCMCModelMetadata,
) -> MCMCPosterior:
    """Materialize unified line and continuum posterior mappings."""
    components: dict[str, MCMCComponentPosterior] = {}
    for handle in workspace.handles:
        names = metadata.line_variables[id(handle.line)]
        components[handle.id] = MCMCComponentPosterior(
            name=handle.id,
            _values={
                "flux": _parameter(idata, names.flux, name="flux"),
                "fwhm": _parameter(idata, names.fwhm, name="fwhm"),
                "center": _parameter(idata, names.center, name="center"),
            },
        )
    components[CONTINUUM_COMPONENT] = MCMCComponentPosterior(
        name=CONTINUUM_COMPONENT,
        _values={
            parameter_name: _parameter(idata, variable_name, name=parameter_name)
            for parameter_name, variable_name in zip(
                workspace.continuum.parameter_names,
                metadata.continuum_variables,
                strict=True,
            )
        },
    )
    return MCMCPosterior(components)


def _parameter(idata: Any, variable_name: str, *, name: str) -> MCMCParameterPosterior:
    samples = _readonly(np.asarray(idata.posterior[variable_name], dtype=float))
    flattened = samples.reshape(-1)
    lower, upper = _highest_density_interval(flattened, probability=0.94)
    return MCMCParameterPosterior(
        name=name,
        samples=samples,
        median=float(np.median(flattened)),
        mean=float(np.mean(flattened)),
        standard_deviation=float(np.std(flattened, ddof=1)),
        hdi_lower=lower,
        hdi_upper=upper,
    )


def _highest_density_interval(
    samples: np.ndarray, *, probability: float
) -> tuple[float, float]:
    ordered = np.sort(samples)
    interval_count = max(int(np.floor(probability * ordered.size)), 1)
    if interval_count >= ordered.size:
        return float(ordered[0]), float(ordered[-1])
    widths = ordered[interval_count:] - ordered[: ordered.size - interval_count]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + interval_count])


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
