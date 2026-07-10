"""Typed handles for a prepared line combination."""

from __future__ import annotations

from dataclasses import dataclass

from noobfriend.inference.spectrum.line import ComponentName, ContributionName, NoobLine, ProfileName


@dataclass(frozen=True, slots=True)
class LineHandle:
    """A line's identity inside one prepared workspace."""

    id: str
    index: int
    line: NoobLine
    observed_wavelength: float
    rest_wavelength: float | None
    component: ComponentName
    contribution: ContributionName
    profile: ProfileName
