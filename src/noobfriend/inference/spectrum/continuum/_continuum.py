"""NoobContinuum: the smooth component under the lines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

from noobfriend.inference.spectrum.priors import (
    POSITIVE,
    REAL,
    Domain,
    ParameterValue,
    check_value,
)

type ContinuumKind = Literal["polynomial", "power_law"]
type ContinuumSharing = Literal["pooled", "independent", "shared"]

_SHARING = ("pooled", "independent", "shared")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class NoobContinuum:
    """Continuum model, built by :meth:`polynomial` or :meth:`power_law`.

    Parameters are anchored at ``lambda_0`` (observed frame, the spectrum's
    unit), which is always required so every parameter means one thing
    before ``prepare``. Undeclared parameters get explicit default priors at
    ``prepare``.

    Attributes
    ----------
    kind : str
        ``"polynomial"`` or ``"power_law"``.
    order : int
        Polynomial order (``0`` for a power law).
    lambda_0 : float
        Anchor wavelength.
    sharing : str
        How a multi-frame fit shares the parameters: ``"pooled"`` (each frame
        its own values drawn around a common mean with a fitted spread),
        ``"independent"`` (each frame its own values) or ``"shared"`` (one set
        for every frame). Irrelevant for a single frame. Pooling estimates a
        spread from the frames themselves, so it needs several frames; with
        two, the pooled mean and spread are barely identified and sampling
        usually fails the convergence checks (use ``"independent"`` or
        ``"shared"`` there).
    declarations : Mapping[str, float or Prior]
        Declared parameter values by name (``"c0"``, ``"c1"``, ... or
        ``"level"``, ``"index"``).
    """

    kind: ContinuumKind
    order: int
    lambda_0: float
    sharing: ContinuumSharing
    declarations: Mapping[str, ParameterValue]

    def __init__(self) -> None:
        """Reject direct construction."""
        raise TypeError(
            "build a continuum with NoobContinuum.polynomial or .power_law."
        )

    @classmethod
    def polynomial(
        cls,
        order: int = 1,
        *,
        lambda_0: float,
        sharing: ContinuumSharing = "pooled",
    ) -> NoobContinuum:
        """Polynomial ``F = sum_n c_n (lambda - lambda_0)**n``, ``n = 0 ... order``.

        Parameters
        ----------
        order
            Highest power, ``>= 0``. Data already continuum-subtracted:
            ``polynomial(order=0, lambda_0=...).coefficient(0, 0.0)``.
        lambda_0
            Anchor wavelength (observed frame, spectrum unit).
        sharing
            ``"pooled"``, ``"independent"`` or ``"shared"``.

        Returns
        -------
        NoobContinuum
            Coefficient ``c_n`` is in flux units per wavelength unit ``**n``.
        """
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError(f"order must be a non-negative integer; got {order!r}.")
        return _new(cls, "polynomial", order, lambda_0, sharing, {})

    @classmethod
    def power_law(
        cls, *, lambda_0: float, sharing: ContinuumSharing = "pooled"
    ) -> NoobContinuum:
        """Power law ``F = level * (lambda / lambda_0) ** index``.

        The level must be positive. The index follows whatever flux-density
        convention (``f_lambda`` or ``f_nu``) the spectrum uses.

        Parameters
        ----------
        lambda_0
            Anchor wavelength (observed frame, spectrum unit).
        sharing
            ``"pooled"``, ``"independent"`` or ``"shared"``.

        Returns
        -------
        NoobContinuum
            Parameters ``level`` (flux at ``lambda_0``) and ``index``.
        """
        return _new(cls, "power_law", 0, lambda_0, sharing, {})

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names in evaluation order."""
        if self.kind == "power_law":
            return ("level", "index")
        return tuple(f"c{degree}" for degree in range(self.order + 1))

    def domain(self, name: str) -> Domain:
        """Return the domain of one parameter."""
        if name not in self.parameter_names:
            raise KeyError(f"{name!r} is not a parameter of this continuum.")
        return POSITIVE if name == "level" else REAL

    def coefficient(self, degree: int, spec: ParameterValue) -> NoobContinuum:
        """Set polynomial coefficient ``c_degree`` (fixed or a prior).

        Returns
        -------
        NoobContinuum
            A copy with the declaration.
        """
        if self.kind != "polynomial":
            raise ValueError(
                "coefficient() applies to a polynomial; use level() / index()."
            )
        if (
            isinstance(degree, bool)
            or not isinstance(degree, int)
            or not 0 <= degree <= self.order
        ):
            raise ValueError(
                f"degree must be an integer in 0 ... {self.order}; got {degree!r}."
            )
        return self._declare(f"c{degree}", spec)

    def level(self, spec: ParameterValue) -> NoobContinuum:
        """Set the power-law flux at ``lambda_0`` (positive; fixed or a prior)."""
        if self.kind != "power_law":
            raise ValueError("level() applies to a power law; use coefficient().")
        return self._declare("level", spec)

    def index(self, spec: ParameterValue) -> NoobContinuum:
        """Set the power-law index (fixed or a prior)."""
        if self.kind != "power_law":
            raise ValueError("index() applies to a power law; use coefficient().")
        return self._declare("index", spec)

    def evaluate(self, xp: Any, wavelength: Any, values: Sequence[Any]) -> Any:
        """Evaluate the continuum.

        Parameters
        ----------
        xp
            Array namespace.
        wavelength
            Observed wavelengths.
        values
            Parameter values in :attr:`parameter_names` order; arrays
            broadcast against ``wavelength``.

        Returns
        -------
        array
            Continuum flux density.
        """
        if self.kind == "power_law":
            level, index = values
            return level * xp.exp(index * xp.log(wavelength / self.lambda_0))
        x = wavelength - self.lambda_0
        total = values[-1] + 0.0 * x
        for value in reversed(values[:-1]):
            total = total * x + value
        return total

    def __repr__(self) -> str:
        """Summarize the model."""
        shape = f"order={self.order}" if self.kind == "polynomial" else "power_law"
        return (
            f"NoobContinuum({self.kind}, {shape}, lambda_0={self.lambda_0:g}, "
            f"sharing={self.sharing!r})"
        )

    def _declare(self, name: str, spec: ParameterValue) -> NoobContinuum:
        value = check_value(spec, self.domain(name), f"continuum.{name}")
        declarations = dict(self.declarations) | {name: value}
        copy = object.__new__(type(self))
        for entry in fields(self):
            object.__setattr__(copy, entry.name, getattr(self, entry.name))
        object.__setattr__(copy, "declarations", MappingProxyType(declarations))
        return copy


def _new(
    cls: type[NoobContinuum],
    kind: ContinuumKind,
    order: int,
    lambda_0: object,
    sharing: object,
    declarations: dict[str, ParameterValue],
) -> NoobContinuum:
    if not isinstance(lambda_0, Real) or isinstance(lambda_0, bool):
        raise TypeError("lambda_0 must be a number.")
    if not isfinite(lambda_0) or lambda_0 <= 0:
        raise ValueError(f"lambda_0 must be a positive wavelength; got {lambda_0!r}.")
    if sharing not in _SHARING:
        raise ValueError(f"sharing must be one of {_SHARING}; got {sharing!r}.")
    continuum = object.__new__(cls)
    object.__setattr__(continuum, "kind", kind)
    object.__setattr__(continuum, "order", order)
    object.__setattr__(continuum, "lambda_0", float(lambda_0))
    object.__setattr__(continuum, "sharing", sharing)
    object.__setattr__(continuum, "declarations", MappingProxyType(declarations))
    return continuum
