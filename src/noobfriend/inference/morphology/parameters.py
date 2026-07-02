"""Small declarative parameter and prior objects for morphology scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prior:
    """Base prior descriptor."""


@dataclass(frozen=True)
class Uniform(Prior):
    """Uniform prior on ``[low, high]``."""

    low: float
    high: float


@dataclass(frozen=True)
class Normal(Prior):
    """Normal prior with ``loc`` and ``scale``."""

    loc: float
    scale: float


@dataclass(frozen=True)
class TruncNormal(Prior):
    """Normal prior truncated to ``[low, high]``."""

    loc: float
    scale: float
    low: float
    high: float


@dataclass(frozen=True)
class LogUniform(Prior):
    """Prior uniform in log-space on positive ``[low, high]``."""

    low: float
    high: float


@dataclass(frozen=True)
class Parameter:
    """Named scalar parameter in a scene."""

    name: str
    prior: Prior | None = None
    init: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameter name."""
        if not self.name:
            raise ValueError("Parameter.name must be non-empty.")


ScalarSpec = float | int | str | Parameter


def resolve_scalar(spec: ScalarSpec, values: dict[str, float]) -> float:
    """Resolve a scalar spec against sampled/fixed parameter values."""
    if isinstance(spec, Parameter):
        return float(values[spec.name] if spec.name in values else spec.init)
    if isinstance(spec, str):
        return float(values[spec])
    if isinstance(spec, bool):
        raise TypeError("boolean values are not valid scalar specs.")
    return float(spec)


@dataclass(frozen=True)
class PerBand:
    """Per-band scalar values resolved by image name."""

    prefix: str | None = None
    values: dict[str, ScalarSpec] | None = None
    default: ScalarSpec | None = None

    def for_band(self, band: str, values: dict[str, float]) -> float:
        """Resolve this value for one band."""
        if self.values and band in self.values:
            return resolve_scalar(self.values[band], values)
        if self.prefix is not None:
            return float(values[f"{self.prefix}_{band}"])
        if self.default is not None:
            return resolve_scalar(self.default, values)
        raise KeyError(f"no value for band {band!r}")


@dataclass(frozen=True)
class FluxPart:
    """One part of a total/fraction flux split."""

    split: "TotalFractionFlux"
    part_name: str

    def for_band(self, band: str, values: dict[str, float]) -> float:
        """Resolve this flux part for one band."""
        total = self.split.total.for_band(band, values)
        fraction = np.clip(self.split.fraction.for_band(band, values), 0.0, 1.0)
        if self.part_name == "host":
            return float(total * fraction)
        if self.part_name == "point":
            return float(total * (1.0 - fraction))
        raise ValueError(f"unknown flux part {self.part_name!r}")


@dataclass(frozen=True)
class TotalFractionFlux:
    """Shared per-band ``total_flux`` plus ``host_fraction`` parameterisation."""

    total: PerBand
    fraction: PerBand

    def part(self, name: str) -> FluxPart:
        """Return the point or host part of this split."""
        if name not in {"point", "host"}:
            raise ValueError("flux part must be 'point' or 'host'.")
        return FluxPart(self, name)


FluxSpec = PerBand | FluxPart | dict[str, ScalarSpec] | ScalarSpec


def resolve_flux(spec: FluxSpec, band: str, values: dict[str, float]) -> float:
    """Resolve a component flux for one band."""
    if isinstance(spec, PerBand | FluxPart):
        return spec.for_band(band, values)
    if isinstance(spec, dict):
        return resolve_scalar(spec[band], values)
    return resolve_scalar(spec, values)


def uv_backend_message() -> str:
    """Return the install hint used by lazy sampler backends."""
    return (
        "JAX/NumPyro backend is not installed.\n\n"
        "Install a CPU backend with:\n"
        '  uv add "jax[cpu]" numpyro\n\n'
        "For CUDA, install the JAX build matching your CUDA version, for example:\n"
        "  uv add numpyro\n"
        "  uv add --index-url "
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "
        '"jax[cuda12]"'
    )
