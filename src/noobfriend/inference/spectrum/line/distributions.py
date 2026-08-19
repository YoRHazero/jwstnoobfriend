"""Declarative priors accepted by line parameter rules.

A line parameter is declared with a fixed value, a ``(lower, upper)`` range,
or one of the priors below. The first two forms stay the fast path and keep
their original meaning; a prior object states the distribution explicitly
instead of leaving the choice to the fit backend.

These are plain value objects: they carry the declaration and nothing else.
How a prior becomes an optimizer box or a PyMC variable is decided by the fit
backends, so this module stays free of both ``scipy`` and ``pymc``.

Every prior needs a finite range. ``Uniform``, ``LogUniform``, and
``TruncatedNormal`` carry one directly; ``Normal`` is the sole unbounded
prior and is therefore only accepted on parameters whose domain is the whole
real line — in practice a line's velocity offset. Widths, fluxes, and
fractions are positive quantities, so they take ``TruncatedNormal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Prior:
    """Base class for declarative parameter priors."""


@dataclass(frozen=True, slots=True)
class Normal(Prior):
    """Normal prior over the whole real line.

    Parameters
    ----------
    mu
        Distribution mean.
    sigma
        Standard deviation; must be positive.

    Raises
    ------
    ValueError
        If ``mu`` is not finite or ``sigma`` is not finite and positive.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        """Validate and normalize the location and scale."""
        _finite(self.mu, "Normal.mu")
        _positive(self.sigma, "Normal.sigma")
        _coerce(self, "mu", "sigma")


@dataclass(frozen=True, slots=True)
class TruncatedNormal(Prior):
    """Normal prior restricted to a finite range.

    Parameters
    ----------
    mu, sigma
        Location and scale of the untruncated normal. ``mu`` may lie outside
        ``[low, high]``.
    low, high
        Finite truncation bounds with ``low < high``.

    Raises
    ------
    ValueError
        If any value is not finite, ``sigma`` is not positive, or the bounds
        are not increasing.
    """

    mu: float
    sigma: float
    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the location, scale, and truncation bounds."""
        _finite(self.mu, "TruncatedNormal.mu")
        _positive(self.sigma, "TruncatedNormal.sigma")
        _range(self.low, self.high, "TruncatedNormal")
        _coerce(self, "mu", "sigma", "low", "high")


@dataclass(frozen=True, slots=True)
class Uniform(Prior):
    """Uniform prior over a finite range.

    Parameters
    ----------
    low, high
        Finite bounds with ``low < high``.

    Raises
    ------
    ValueError
        If the bounds are not finite and increasing.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate and normalize the bounds."""
        _range(self.low, self.high, "Uniform")
        _coerce(self, "low", "high")


@dataclass(frozen=True, slots=True)
class LogUniform(Prior):
    """Prior uniform in the logarithm of the parameter, over a finite range.

    The natural default for a positive scale parameter such as a line width,
    where the plausible values span orders of magnitude.

    Parameters
    ----------
    low, high
        Finite bounds with ``0 < low < high``.

    Raises
    ------
    ValueError
        If the bounds are not finite and increasing, or ``low`` is not
        positive.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the bounds and their positivity."""
        _range(self.low, self.high, "LogUniform")
        _positive(self.low, "LogUniform.low")
        _coerce(self, "low", "high")


def prior_support(prior: Prior) -> tuple[float, float]:
    """Return the range a prior is supported on.

    ``Normal`` is unbounded and reports ``(-inf, inf)``; every other prior
    reports its declared finite range.

    Parameters
    ----------
    prior
        The declared prior.

    Returns
    -------
    tuple of float
        Lower and upper edge of the support.
    """
    if isinstance(prior, Normal):
        return float("-inf"), float("inf")
    if isinstance(prior, TruncatedNormal):
        return prior.low, prior.high
    if isinstance(prior, (Uniform, LogUniform)):
        return prior.low, prior.high
    raise TypeError(f"unsupported prior type: {type(prior).__name__}.")


def _coerce(prior: Prior, *names: str) -> None:
    """Store every named field as a plain ``float``."""
    for name in names:
        object.__setattr__(prior, name, float(getattr(prior, name)))


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _range(low: float, high: float, name: str) -> None:
    _finite(low, f"{name}.low")
    _finite(high, f"{name}.high")
    if low >= high:
        raise ValueError(f"{name} bounds must be increasing.")


__all__ = [
    "LogUniform",
    "Normal",
    "Prior",
    "TruncatedNormal",
    "Uniform",
    "prior_support",
]
