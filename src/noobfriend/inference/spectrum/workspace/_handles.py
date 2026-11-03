"""Read-only handles for the components of a workspace (``ws[name]``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, eq=False)
class LineHandle:
    """One resolved line of a workspace.

    Attributes
    ----------
    name
        The line's name.
    component, contribution, profile
        As declared.
    base
        The base line's name, or ``None``.
    nominal
        Nominal observed wavelength in the spectrum's unit (rest lines
        shifted by the spectrum's ``z``).
    sign
        ``+1`` for emission, ``-1`` for absorption.
    kernels
        Kernel kinds, in declaration order.
    lsf_fwhm_kms
        Gaussian LSF FWHM (km/s) of this line in each frame, ``None`` for a
        frame without resolving power.
    parameters
        Names of the table rows this line owns.
    declaration
        The ``NoobLine`` the handle was resolved from.
    """

    name: str
    component: str
    contribution: str
    profile: str
    base: str | None
    nominal: float
    sign: float
    kernels: tuple[str, ...]
    lsf_fwhm_kms: Mapping[str, float | None]
    parameters: tuple[str, ...]
    declaration: Any

    @property
    def quantities(self) -> tuple[str, ...]:
        """Flat names of the physical quantities this line reports."""
        names = [
            f"{self.name}.{role}" for role in ("center", "delta_v", "fwhm", "flux")
        ]
        for kind in self.kernels:
            names += [f"{self.name}.{kind}.fwhm", f"{self.name}.{kind}.fraction"]
        return tuple(names)


@dataclass(frozen=True, slots=True, eq=False)
class ContinuumHandle:
    """The resolved continuum of a workspace (``ws["continuum"]``).

    Attributes
    ----------
    kind, order, lambda_0, sharing
        As declared (``sharing`` is ``"shared"`` for a single frame).
    parameter_names
        Model parameter names (``"c0"``, ... or ``"level"``, ``"index"``).
    frame_parameters
        For each frame id, the table rows feeding that frame's continuum, in
        ``parameter_names`` order.
    parameters
        Every table row the continuum owns (including per-frame rows).
    declaration
        The ``NoobContinuum`` the handle was resolved from.
    """

    kind: str
    order: int
    lambda_0: float
    sharing: str
    parameter_names: tuple[str, ...]
    frame_parameters: Mapping[str, tuple[str, ...]]
    parameters: tuple[str, ...]
    declaration: Any

    @property
    def name(self) -> str:
        """The reserved component name ``"continuum"``."""
        return "continuum"
