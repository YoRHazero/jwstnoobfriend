"""Parameter rules for line fitting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Literal

from noobfriend.inference.spectrum.line.distributions import Prior, prior_support
from noobfriend.inference.spectrum.line.types import ParameterSpec

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line.line import NoobLine


class _ParameterMode(Enum):
    """How a parameter is constrained during fitting."""

    FREE = auto()
    FIXED = auto()
    BOUNDED = auto()
    PRIOR = auto()
    LOCKED = auto()
    RATIO = auto()


@dataclass(frozen=True, slots=True)
class _ParameterRule:
    """A resolved rule for one fitted line parameter."""

    mode: _ParameterMode
    value: float | None = None
    bounds: tuple[float, float] | None = None
    prior: Prior | None = None
    target: NoobLine | None = None
    offset_unit: Literal["km/s", "wavelength"] | None = None

    @classmethod
    def free(cls) -> _ParameterRule:
        return cls(_ParameterMode.FREE)

    @classmethod
    def locked(cls, target: NoobLine) -> _ParameterRule:
        return cls(_ParameterMode.LOCKED, target=target)

    @classmethod
    def fixed(
        cls, value: float, *, offset_unit: Literal["km/s", "wavelength"] | None = None
    ) -> _ParameterRule:
        return cls(_ParameterMode.FIXED, value=value, offset_unit=offset_unit)

    @classmethod
    def bounded(
        cls,
        bounds: tuple[float, float],
        *,
        offset_unit: Literal["km/s", "wavelength"] | None = None,
    ) -> _ParameterRule:
        return cls(_ParameterMode.BOUNDED, bounds=bounds, offset_unit=offset_unit)

    @classmethod
    def declared(
        cls,
        prior: Prior,
        *,
        offset_unit: Literal["km/s", "wavelength"] | None = None,
    ) -> _ParameterRule:
        return cls(_ParameterMode.PRIOR, prior=prior, offset_unit=offset_unit)

    @classmethod
    def ratio(cls, value: float) -> _ParameterRule:
        return cls(_ParameterMode.RATIO, value=value)

    @property
    def is_free(self) -> bool:
        return self.mode is _ParameterMode.FREE

    @property
    def is_fixed(self) -> bool:
        return self.mode is _ParameterMode.FIXED

    @property
    def is_bounded(self) -> bool:
        return self.mode is _ParameterMode.BOUNDED

    @property
    def is_prior(self) -> bool:
        return self.mode is _ParameterMode.PRIOR

    @property
    def is_locked(self) -> bool:
        return self.mode is _ParameterMode.LOCKED

    @property
    def is_ratio(self) -> bool:
        return self.mode is _ParameterMode.RATIO


def _rule_from_fixed_or_bounded(
    value: ParameterSpec | None,
    name: str,
    *,
    offset_unit: Literal["km/s", "wavelength"] | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> _ParameterRule:
    """Build a fixed, bounded, or declared-prior rule from the value contract."""
    mode, parsed = _parse_fixed_or_bounded(
        value, name, positive=positive, nonnegative=nonnegative
    )
    if mode is _ParameterMode.PRIOR:
        return _ParameterRule.declared(parsed, offset_unit=offset_unit)
    if mode is _ParameterMode.FIXED:
        return _ParameterRule.fixed(parsed, offset_unit=offset_unit)
    return _ParameterRule.bounded(parsed, offset_unit=offset_unit)


def _parse_fixed_or_bounded(
    value: ParameterSpec | None,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[_ParameterMode, float | tuple[float, float] | Prior]:
    """Parse the public fixed, bounded, or prior input shape."""
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, Prior):
        _check_prior_domain(value, name, positive=positive, nonnegative=nonnegative)
        return _ParameterMode.PRIOR, value
    if _is_number(value):
        parsed = float(value)
        _check_limit(parsed, name, positive=positive, nonnegative=nonnegative)
        return _ParameterMode.FIXED, parsed
    if isinstance(value, tuple) and len(value) == 2:
        lower = _required_float(value[0], name=f"{name}[0]")
        upper = _required_float(value[1], name=f"{name}[1]")
        if lower >= upper:
            raise ValueError(f"{name} bounds must be increasing.")
        _check_limit(lower, f"{name}[0]", positive=positive, nonnegative=nonnegative)
        _check_limit(upper, f"{name}[1]", positive=positive, nonnegative=nonnegative)
        return _ParameterMode.BOUNDED, (lower, upper)
    raise TypeError(
        f"{name} must be a number, a two-item tuple of numbers, or a prior."
    )


def _check_prior_domain(
    prior: Prior, name: str, *, positive: bool, nonnegative: bool
) -> None:
    """Reject a prior whose support leaves the parameter's own domain.

    Widths, fluxes, and fractions are restricted to positive (or nonnegative)
    values, so an unbounded ``Normal`` cannot describe them: a negative width
    has no profile and a negative flux has no meaning. Those parameters take
    ``TruncatedNormal`` instead, whose bounds state the restriction
    explicitly.
    """
    if not (positive or nonnegative):
        return
    lower, _ = prior_support(prior)
    domain = "positive" if positive else "nonnegative"
    if not isfinite(lower):
        raise ValueError(
            f"{name} cannot take an unbounded {type(prior).__name__} prior "
            f"because this parameter must be {domain}; use "
            "TruncatedNormal(mu, sigma, low=..., high=...) instead."
        )
    if positive and lower <= 0.0:
        raise ValueError(f"{name} prior support must be positive.")
    if nonnegative and lower < 0.0:
        raise ValueError(f"{name} prior support must be nonnegative.")


def _required_float(value: float, *, name: str) -> float:
    """Convert a finite real number to ``float``."""
    if not _is_number(value):
        raise TypeError(f"{name} must be a finite number.")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _is_number(value: object) -> bool:
    """Whether ``value`` is a real number but not ``bool``."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _check_limit(value: float, name: str, *, positive: bool, nonnegative: bool) -> None:
    """Validate sign constraints."""
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
