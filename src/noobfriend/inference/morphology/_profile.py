"""Profile components: the building blocks of a spatial decomposition.

A :class:`Profile` is a *declaration*, independent of any image (like
:class:`~noobfriend.inference.spectrum.NoobLine`): a named analytic
surface-brightness component whose geometry parameters are shared across bands and
whose flux is per-band. Typed subclasses -- :class:`Sersic`, :class:`Exponential`,
:class:`PointSource` -- give each profile a real signature with only the axes it
has, rather than a ``profile="..."`` string over a generic parameter bag.

Each geometry axis takes a :data:`~noobfriend.inference.morphology._param.Spec`
(nothing / a fixed number / an interval / a :class:`Prior` / a shared
:class:`Param` / an expression) and is canonicalised to a
:class:`~noobfriend.inference.morphology._param.ParamNode` at construction. Flux is
held as a :class:`FluxSpec`: a single template applied independently per band, or
an explicit per-band mapping, expanded against the actual bands only in the model
layer (where each band gets its own free flux unless an explicit shared
:class:`Param` was passed).

This module is internal; the profile classes are re-exported from
:mod:`noobfriend.inference.morphology`.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from noobfriend.inference.morphology._param import (
    Param,
    ParamNode,
    Spec,
    parse_spec,
)


@dataclass(frozen=True)
class FluxSpec:
    """How a component's per-band flux is specified.

    The bands are not known when a profile is declared, so flux is kept as a recipe
    and expanded in the model layer. A ``template`` spec is applied *independently*
    to each band (a fresh free flux per band) unless it is a shared
    :class:`Param`, which links the bands. A ``per_band`` mapping gives each named
    band its own spec.

    Attributes
    ----------
    template : Spec
        One spec for every band; ``None`` is the default (free positive per band).
        Mutually exclusive with ``per_band``.
    per_band : tuple[tuple[str, Spec], ...] or None
        Explicit ``(band, spec)`` pairs; ``None`` when a template is used.
    """

    template: Spec = None
    per_band: tuple[tuple[str, Spec], ...] | None = None


class Profile(ABC):
    """A named analytic light profile with shared geometry and per-band flux.

    Construct a typed subclass (:class:`Sersic`, :class:`Exponential`,
    :class:`PointSource`); :class:`Profile` itself is the shared base and the type
    to annotate against. Treated as immutable after construction.

    Attributes
    ----------
    name : str
        Component name, used to label parameters and look results up.
    params : dict[str, ParamNode]
        Canonicalised geometry parameters, one per axis in :attr:`GEOMETRY`.
    flux : FluxSpec
        The per-band flux recipe.
    """

    #: The geometry axes this profile exposes, in canonical order.
    GEOMETRY: ClassVar[tuple[str, ...]]

    name: str
    params: dict[str, ParamNode]
    flux: FluxSpec

    def __init__(
        self,
        name: str,
        *,
        flux: Spec | Mapping[str, Spec] = None,
        **specs: Spec,
    ) -> None:
        """Validate the name, canonicalise geometry, and normalise flux.

        Raises
        ------
        ValueError
            If ``name`` is empty, an unknown axis is given, or a geometry / flux
            spec is malformed.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Profile name must be a non-empty string.")
        unknown = set(specs) - set(self.GEOMETRY)
        if unknown:
            raise ValueError(
                f"{name}: unknown parameter(s) {sorted(unknown)} for "
                f"{type(self).__name__}; axes are {list(self.GEOMETRY)}."
            )
        self.name = name
        self.params = {
            axis: parse_spec(specs.get(axis), where=f"{name}.{axis}")
            for axis in self.GEOMETRY
        }
        self.flux = _normalize_flux(name, flux)

    def free_geometry_params(self) -> tuple[Param, ...]:
        """Return the free geometry parameters, unique by identity, in axis order.

        Flux parameters are not included -- they are per-band and only materialise
        when the model is built against an image's bands.
        """
        out: list[Param] = []
        for node in self.params.values():
            node._collect(out)
        return tuple(out)

    def __repr__(self) -> str:
        bits = [repr(self.name)]
        bits += [f"{axis}={node!r}" for axis, node in self.params.items()]
        if self.flux.per_band is not None:
            bits.append(f"flux={{{len(self.flux.per_band)} bands}}")
        elif self.flux.template is not None:
            bits.append(f"flux={self.flux.template!r}")
        return f"{type(self).__name__}({', '.join(bits)})"


class Sersic(Profile):
    """A Sérsic profile (free index ``n``).

    Parameters
    ----------
    name : str
        Component name.
    x0, y0 : Spec
        Sky-frame centre offset.
    r_eff : Spec
        Half-light radius (arcsec).
    n : Spec
        Sérsic index.
    q : Spec
        Axis ratio (minor/major).
    theta : Spec
        Position angle (degrees, east of north).
    flux : Spec or mapping
        Per-band flux recipe; see :class:`FluxSpec`.
    """

    GEOMETRY: ClassVar[tuple[str, ...]] = ("x0", "y0", "r_eff", "n", "q", "theta")

    def __init__(
        self,
        name: str,
        *,
        x0: Spec = None,
        y0: Spec = None,
        r_eff: Spec = None,
        n: Spec = None,
        q: Spec = None,
        theta: Spec = None,
        flux: Spec | Mapping[str, Spec] = None,
    ) -> None:
        super().__init__(
            name, flux=flux, x0=x0, y0=y0, r_eff=r_eff, n=n, q=q, theta=theta
        )


class Exponential(Profile):
    """An exponential disk -- a Sérsic with ``n`` fixed at 1.

    Parameters
    ----------
    name : str
        Component name.
    x0, y0 : Spec
        Sky-frame centre offset.
    r_eff : Spec
        Half-light radius (arcsec).
    q : Spec
        Axis ratio (minor/major).
    theta : Spec
        Position angle (degrees, east of north).
    flux : Spec or mapping
        Per-band flux recipe; see :class:`FluxSpec`.
    """

    GEOMETRY: ClassVar[tuple[str, ...]] = ("x0", "y0", "r_eff", "q", "theta")

    def __init__(
        self,
        name: str,
        *,
        x0: Spec = None,
        y0: Spec = None,
        r_eff: Spec = None,
        q: Spec = None,
        theta: Spec = None,
        flux: Spec | Mapping[str, Spec] = None,
    ) -> None:
        super().__init__(name, flux=flux, x0=x0, y0=y0, r_eff=r_eff, q=q, theta=theta)


class PointSource(Profile):
    """An unresolved point source -- the PSF itself, with only a centre and flux.

    Parameters
    ----------
    name : str
        Component name.
    x0, y0 : Spec
        Sky-frame centre offset.
    flux : Spec or mapping
        Per-band flux recipe; see :class:`FluxSpec`.
    """

    GEOMETRY: ClassVar[tuple[str, ...]] = ("x0", "y0")

    def __init__(
        self,
        name: str,
        *,
        x0: Spec = None,
        y0: Spec = None,
        flux: Spec | Mapping[str, Spec] = None,
    ) -> None:
        super().__init__(name, flux=flux, x0=x0, y0=y0)


def _normalize_flux(name: str, flux: Spec | Mapping[str, Spec]) -> FluxSpec:
    """Validate a flux input and freeze it into a :class:`FluxSpec`.

    Raises
    ------
    ValueError
        If a per-band mapping is empty, or any contained spec is malformed.
    """
    if isinstance(flux, Mapping):
        if not flux:
            raise ValueError(
                f"{name}: flux mapping is empty; omit flux for the default."
            )
        per_band = tuple((str(band), spec) for band, spec in flux.items())
        for band, spec in per_band:
            parse_spec(spec, where=f"{name}.flux[{band!r}]")  # validate, discard
        return FluxSpec(template=None, per_band=per_band)
    parse_spec(flux, where=f"{name}.flux")  # validate, discard
    return FluxSpec(template=flux, per_band=None)
