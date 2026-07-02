"""Scene components and centre constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from noobfriend.inference.morphology.parameters import (
    FluxSpec,
    ScalarSpec,
    resolve_scalar,
)


class Center(Protocol):
    """Protocol for objects resolving a component centre in arcsec."""

    def xy(self, scene: "SceneProtocol", values: dict[str, float]) -> tuple[float, float]:
        """Return ``(x, y)`` in the shared arcsec frame."""


class SceneProtocol(Protocol):
    """Minimal scene lookup protocol used by centre constraints."""

    def component_center(
        self, name: str, values: dict[str, float]
    ) -> tuple[float, float]:
        """Resolve a component's centre."""


@dataclass(frozen=True)
class FixedCenter:
    """Fixed centre in arcsec."""

    x: float = 0.0
    y: float = 0.0

    def xy(self, scene: SceneProtocol, values: dict[str, float]) -> tuple[float, float]:
        """Return the fixed centre."""
        return float(self.x), float(self.y)


@dataclass(frozen=True)
class FreeCenter:
    """Centre resolved from named scalar parameters."""

    x: ScalarSpec = "x"
    y: ScalarSpec = "y"

    def xy(self, scene: SceneProtocol, values: dict[str, float]) -> tuple[float, float]:
        """Resolve the free centre."""
        return resolve_scalar(self.x, values), resolve_scalar(self.y, values)


@dataclass(frozen=True)
class EllipticalOffsetFrom:
    """Constrain this centre so a reference point lies inside its Sersic ellipse.

    ``rho`` is the normalised elliptical radius of the reference point measured
    from this centre. Values in ``[0, 1]`` place the reference inside one
    effective-radius ellipse.
    """

    reference: str
    r_eff: ScalarSpec
    q: ScalarSpec
    theta: ScalarSpec
    rho: ScalarSpec = "host_offset_rho"
    phi: ScalarSpec = "host_offset_phi"

    def xy(self, scene: SceneProtocol, values: dict[str, float]) -> tuple[float, float]:
        """Return this centre in arcsec."""
        ref_x, ref_y = scene.component_center(self.reference, values)
        dx, dy = elliptical_offset_xy(
            r_eff=resolve_scalar(self.r_eff, values),
            q=resolve_scalar(self.q, values),
            theta_deg=resolve_scalar(self.theta, values),
            rho=resolve_scalar(self.rho, values),
            phi_deg=resolve_scalar(self.phi, values),
        )
        return ref_x + dx, ref_y + dy


def elliptical_offset_xy(
    *, r_eff: float, q: float, theta_deg: float, rho: float, phi_deg: float
) -> tuple[float, float]:
    """Map ellipse coordinates to arcsec ``dx, dy``."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    dx = r_eff * rho * (
        np.cos(phi) * np.cos(theta) - q * np.sin(phi) * np.sin(theta)
    )
    dy = r_eff * rho * (
        np.cos(phi) * np.sin(theta) + q * np.sin(phi) * np.cos(theta)
    )
    return float(dx), float(dy)


@dataclass(frozen=True)
class SersicShape:
    """Sersic shape parameters in arcsec/degrees."""

    r_eff: ScalarSpec
    n: ScalarSpec
    q: ScalarSpec
    theta: ScalarSpec

    def resolved(self, values: dict[str, float]) -> dict[str, float]:
        """Return resolved numeric shape parameters."""
        return {
            "r_eff": resolve_scalar(self.r_eff, values),
            "n": resolve_scalar(self.n, values),
            "q": resolve_scalar(self.q, values),
            "theta": resolve_scalar(self.theta, values),
        }


@dataclass(frozen=True)
class Component:
    """Base component contract."""

    name: str
    center: Center
    flux: FluxSpec

    kind: str = "component"


@dataclass(frozen=True)
class Background:
    """Uniform per-band background component."""

    name: str
    level: FluxSpec
    kind: str = "background"


@dataclass(frozen=True)
class Point(Component):
    """Point-source component."""

    kind: str = "point"


@dataclass(frozen=True)
class Sersic(Component):
    """Sersic surface-brightness component."""

    shape: SersicShape | None = None
    kind: str = "sersic"

    def __post_init__(self) -> None:
        """Require a Sersic shape."""
        if self.shape is None:
            raise ValueError("Sersic requires a SersicShape.")
