"""Composable morphology scene contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from typing import Any


@dataclass(frozen=True)
class Scene(Sequence[Any]):
    """Ordered component list with no hidden physical ties."""

    components: tuple[Any, ...]

    def __init__(self, components: Iterable[Any]) -> None:
        """Create a scene from an ordered component iterable."""
        items = tuple(components)
        if not items:
            raise ValueError("Scene needs at least one component.")
        names = [component.name for component in items]
        if len(set(names)) != len(names):
            raise ValueError(f"component names must be unique, got {names}.")
        object.__setattr__(self, "components", items)

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self.components)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over components in rendering order."""
        return iter(self.components)

    def __getitem__(self, index: int) -> Any:
        """Return a component by integer index."""
        return self.components[index]

    def component(self, name: str) -> Any:
        """Return one component by name."""
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)

    def component_center(
        self, name: str, values: dict[str, float]
    ) -> tuple[float, float]:
        """Resolve one component centre in arcsec."""
        component = self.component(name)
        return component.center.xy(self, values)
