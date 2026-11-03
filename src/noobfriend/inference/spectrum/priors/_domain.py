"""Parameter domains and their unconstrained transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Literal

type TransformName = Literal["identity", "log", "logit"]


@dataclass(frozen=True, slots=True)
class Domain:
    """The interval a parameter's values live in.

    Attributes
    ----------
    name
        Short label used in messages (``"real"``, ``"positive"``, ...).
    low, high
        Interval ends; infinite ends are open.
    low_closed, high_closed
        Whether the finite ends belong to the domain.
    """

    name: str
    low: float = -inf
    high: float = inf
    low_closed: bool = False
    high_closed: bool = False

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in the domain."""
        if not isfinite(value):
            return False
        above = value >= self.low if self.low_closed else value > self.low
        below = value <= self.high if self.high_closed else value < self.high
        return above and below

    def contains_interval(self, low: float, high: float) -> bool:
        """Whether the closure-compatible interval ``[low, high]`` fits inside.

        An end may sit exactly on an open domain end (a prior bounded at
        ``0`` on a positive parameter), since the endpoint itself carries no
        probability.
        """
        return self.low <= low and high <= self.high

    @property
    def transform(self) -> TransformName:
        """The map to unconstrained coordinates used for joint priors.

        ``log`` for positive and nonnegative parameters, ``logit`` for
        fractions in ``(0, 1]``, identity otherwise.
        """
        if self.low == 0.0 and self.high == inf:
            return "log"
        if self.low == 0.0 and self.high == 1.0:
            return "logit"
        return "identity"

    def to_unconstrained(self, xp: Any, value: Any) -> Any:
        """Map values from the domain to unconstrained coordinates."""
        transform = self.transform
        if transform == "log":
            return xp.log(value)
        if transform == "logit":
            return xp.log(value) - xp.log1p(-value)
        return value

    def from_unconstrained(self, xp: Any, value: Any) -> Any:
        """Map unconstrained coordinates back into the domain."""
        transform = self.transform
        if transform == "log":
            return xp.exp(value)
        if transform == "logit":
            return 1.0 / (1.0 + xp.exp(-value))
        return value

    def __str__(self) -> str:
        """Return interval notation."""
        left = "[" if self.low_closed else "("
        right = "]" if self.high_closed else ")"
        return f"{left}{self.low:g}, {self.high:g}{right}"


REAL = Domain("real")
POSITIVE = Domain("positive", 0.0, inf)
NONNEGATIVE = Domain("nonnegative", 0.0, inf, low_closed=True)
FRACTION = Domain("fraction", 0.0, 1.0, high_closed=True)
CORRELATION = Domain("correlation", -1.0, 1.0)
