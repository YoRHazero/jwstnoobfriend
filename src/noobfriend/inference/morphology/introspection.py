"""Scene introspection helpers that do not depend on sampler backends."""

from __future__ import annotations

from typing import Any

from noobfriend.inference.morphology.components import (
    Background,
    EllipticalOffsetFrom,
    FreeCenter,
    Sersic,
)
from noobfriend.inference.morphology.parameters import FluxPart, Parameter, PerBand
from noobfriend.inference.morphology.scene import Scene


def scene_parameters(scene: Scene) -> tuple[Parameter, ...]:
    """Collect unique named ``Parameter`` objects from a scene."""
    found: dict[str, Parameter] = {}

    def add(param: Parameter) -> None:
        existing = found.get(param.name)
        if existing is not None and existing != param:
            raise ValueError(f"conflicting definitions for parameter {param.name!r}")
        found[param.name] = param

    def visit_scalar(value: Any) -> None:
        if isinstance(value, Parameter):
            add(value)

    def visit_flux(value: Any) -> None:
        if isinstance(value, FluxPart):
            visit_flux(value.split.total)
            visit_flux(value.split.fraction)
        elif isinstance(value, PerBand):
            if value.values:
                for item in value.values.values():
                    visit_scalar(item)
            if value.default is not None:
                visit_scalar(value.default)
        elif isinstance(value, dict):
            for item in value.values():
                visit_scalar(item)
        else:
            visit_scalar(value)

    for component in scene:
        if isinstance(component, Background):
            visit_flux(component.level)
            continue
        center = component.center
        if isinstance(center, FreeCenter):
            visit_scalar(center.x)
            visit_scalar(center.y)
        elif isinstance(center, EllipticalOffsetFrom):
            for value in [center.r_eff, center.q, center.theta, center.rho, center.phi]:
                visit_scalar(value)
        visit_flux(component.flux)
        if isinstance(component, Sersic) and component.shape is not None:
            for value in [
                component.shape.r_eff,
                component.shape.n,
                component.shape.q,
                component.shape.theta,
            ]:
                visit_scalar(value)
    return tuple(found.values())
