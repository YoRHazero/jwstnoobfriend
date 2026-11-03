"""Structural types ``prepare`` accepts.

The workspace sits below the declaration layers, so it knows lines,
continua and spectra only through these runtime-checkable protocols.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ParameterRuleLike(Protocol):
    """A line parameter rule (``kind`` and ``value``)."""

    kind: str
    value: Any


@runtime_checkable
class KernelLike(Protocol):
    """A line kernel declaration (``None`` parameters are locked)."""

    kind: str
    fwhm: Any
    fraction: Any


@runtime_checkable
class LineLike(Protocol):
    """A line declaration."""

    name: str
    rest: float | None
    obs: float | None
    unit: str
    component: str
    contribution: str
    profile: str
    base: Any
    center_rule: ParameterRuleLike
    fwhm_rule: ParameterRuleLike
    flux_rule: ParameterRuleLike
    kernels: Sequence[KernelLike]


@runtime_checkable
class ContinuumLike(Protocol):
    """A continuum declaration."""

    kind: str
    order: int
    lambda_0: float
    sharing: str
    declarations: Mapping[str, Any]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names in evaluation order."""
        ...

    def domain(self, name: str) -> Any:
        """Domain of one parameter."""
        ...

    def evaluate(self, xp: Any, wavelength: Any, values: Sequence[Any]) -> Any:
        """Evaluate the continuum."""
        ...


@runtime_checkable
class FrameLike(Protocol):
    """One spectrum frame."""

    id: str
    obs: Any
    flux: Any
    error: Any
    mask: Any
    resolving_power: Any
    noise: Any
    source_2d: Any

    @property
    def usable(self) -> Any:
        """Valid, unmasked pixels."""
        ...

    def resolving_power_at(self, wavelength: float) -> float | None:
        """Return the resolving power at one wavelength."""
        ...


@runtime_checkable
class SpectrumLike(Protocol):
    """A spectrum: frames plus source-level metadata."""

    frames: Mapping[str, FrameLike]
    z: float | None
    unit: str
