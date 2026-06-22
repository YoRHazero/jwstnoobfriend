"""Priors, parameters, and the small expression algebra that ties them.

A profile parameter (a Sérsic ``n``, an ``r_eff``, a centre ``x0``) is given by
the user as one of: nothing (a data-driven default), a fixed number, an ``(lo,
hi)`` interval, an explicit :class:`Prior`, a named :class:`Param`, or an
*expression* of parameters. :func:`parse_spec` canonicalises any of these into a
:class:`ParamNode`.

The two ideas that make this spatial-native (rather than a transplant of the line
module's fixed-physics ties):

* **Sharing by identity.** A :class:`Param` is a named free parameter; *reusing
  the same object* in two places links them to one latent. An anonymous prior
  (a bare :class:`Prior` or an interval) yields a *fresh* independent parameter
  each time -- structural equality never links.
* **Ordering by reparameterisation.** Parameters support ``+``, ``-`` and scaling
  by a constant, so an identifiability constraint such as "the disk is larger than
  the bulge" is written ``r_eff_bulge + gap`` with ``gap`` a positive parameter --
  no label switching, no fixed physics baked in.

The prior objects are declarative value types; their translation to numpyro
distributions and the resolution of default (``prior=None``) parameters against
the data both happen later, in the model layer. This module is internal; the
public priors, :class:`Param`, and the spec alias are re-exported from
:mod:`noobfriend.inference.morphology`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #


class Prior(ABC):
    """Base class for a declarative prior distribution.

    Subclasses are immutable value objects describing a distribution; they carry
    no sampling machinery here (the numpyro translation lives in the model layer).
    """


@dataclass(frozen=True)
class Uniform(Prior):
    """A flat prior on ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Finite bounds with ``low < high``.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the bounds."""
        _require_finite(low=self.low, high=self.high)
        if not self.low < self.high:
            raise ValueError(
                f"Uniform needs low < high, got ({self.low}, {self.high})."
            )

    def __repr__(self) -> str:
        return f"Uniform({self.low:g}, {self.high:g})"


@dataclass(frozen=True)
class Normal(Prior):
    """A Gaussian prior.

    Parameters
    ----------
    loc : float
        Mean.
    scale : float
        Standard deviation (``> 0``).
    """

    loc: float
    scale: float

    def __post_init__(self) -> None:
        """Validate the scale."""
        _require_finite(loc=self.loc, scale=self.scale)
        if self.scale <= 0:
            raise ValueError(f"Normal needs scale > 0, got {self.scale}.")

    def __repr__(self) -> str:
        return f"Normal({self.loc:g}, {self.scale:g})"


@dataclass(frozen=True)
class TruncNormal(Prior):
    """A Gaussian truncated to ``[low, high]``.

    Parameters
    ----------
    loc, scale : float
        Mean and standard deviation (``scale > 0``) of the untruncated Gaussian.
    low, high : float
        Finite truncation bounds with ``low < high``.
    """

    loc: float
    scale: float
    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the scale and bounds."""
        _require_finite(loc=self.loc, scale=self.scale, low=self.low, high=self.high)
        if self.scale <= 0:
            raise ValueError(f"TruncNormal needs scale > 0, got {self.scale}.")
        if not self.low < self.high:
            raise ValueError(
                f"TruncNormal needs low < high, got ({self.low}, {self.high})."
            )

    def __repr__(self) -> str:
        return f"TruncNormal({self.loc:g}, {self.scale:g}, {self.low:g}, {self.high:g})"


@dataclass(frozen=True)
class LogUniform(Prior):
    """A prior flat in the logarithm, on ``[low, high]`` with ``low > 0``.

    Parameters
    ----------
    low, high : float
        Finite positive bounds with ``0 < low < high``.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the bounds."""
        _require_finite(low=self.low, high=self.high)
        if not 0 < self.low < self.high:
            raise ValueError(
                f"LogUniform needs 0 < low < high, got ({self.low}, {self.high})."
            )

    def __repr__(self) -> str:
        return f"LogUniform({self.low:g}, {self.high:g})"


def _require_finite(**named: float) -> None:
    """Raise if any keyword value is not a finite real number."""
    for name, value in named.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")


# --------------------------------------------------------------------------- #
# Parameter expression nodes
# --------------------------------------------------------------------------- #


class ParamNode(ABC):
    """An element of a parameter expression.

    A constant, a free parameter, or a combination of them. Supports ``+``, ``-``,
    unary ``-`` and scaling by a constant; ``*`` between two parameters is rejected
    (only linear reparameterisation is intended).
    """

    @abstractmethod
    def evaluate(self, env: Mapping[Param, float]) -> float:
        """Evaluate the node given a value for each free :class:`Param`.

        Parameters
        ----------
        env : mapping of Param to float
            A value for every free parameter the node depends on.

        Returns
        -------
        float

        Raises
        ------
        KeyError
            If a depended-on parameter has no value in ``env``.
        """

    @abstractmethod
    def _collect(self, out: list[Param]) -> None:
        """Append the free parameters of this node to ``out``, by identity."""

    def __add__(self, other: object) -> ParamNode:
        node = _coerce_operand(other)
        return NotImplemented if node is None else Add(self, node)

    def __radd__(self, other: object) -> ParamNode:
        node = _coerce_operand(other)
        return NotImplemented if node is None else Add(node, self)

    def __sub__(self, other: object) -> ParamNode:
        node = _coerce_operand(other)
        return NotImplemented if node is None else Add(self, -node)

    def __rsub__(self, other: object) -> ParamNode:
        node = _coerce_operand(other)
        return NotImplemented if node is None else Add(node, -self)

    def __mul__(self, other: object) -> ParamNode:
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            return NotImplemented
        return Mul(self, float(other))

    def __rmul__(self, other: object) -> ParamNode:
        return self.__mul__(other)

    def __neg__(self) -> ParamNode:
        return Mul(self, -1.0)


@dataclass(frozen=True, eq=False)
class Constant(ParamNode):
    """A fixed scalar -- a parameter the user pinned to a number (held, not fit)."""

    value: float

    def evaluate(self, env: Mapping[Param, float]) -> float:
        return self.value

    def _collect(self, out: list[Param]) -> None:
        return None

    def __repr__(self) -> str:
        return f"{self.value:g}"


@dataclass(frozen=True, eq=False)
class Param(ParamNode):
    """A named free parameter, identified by object identity for linking.

    Reuse the *same* ``Param`` instance across components to tie them to one
    latent; build distinct instances to keep them independent. ``prior=None`` means
    "fill from the default prior policy against the data" -- the common case for a
    parameter you name only so you can share it.

    Parameters
    ----------
    name : str or None
        Label for the latent (non-empty). ``None`` for an anonymous parameter,
        which the model layer names after the component and axis it sits on.
    prior : Prior or None
        The prior, or ``None`` to defer to the data-driven default policy.
    """

    name: str | None = None
    prior: Prior | None = None

    def __post_init__(self) -> None:
        """Validate the name."""
        if self.name is not None and not str(self.name):
            raise ValueError("Param name must be a non-empty string or None.")

    def evaluate(self, env: Mapping[Param, float]) -> float:
        return env[self]

    def _collect(self, out: list[Param]) -> None:
        if all(p is not self for p in out):
            out.append(self)

    def __repr__(self) -> str:
        body = "default" if self.prior is None else repr(self.prior)
        return f"{self.name or ''}~{body}"


@dataclass(frozen=True, eq=False)
class Add(ParamNode):
    """The sum of two parameter nodes."""

    left: ParamNode
    right: ParamNode

    def evaluate(self, env: Mapping[Param, float]) -> float:
        return self.left.evaluate(env) + self.right.evaluate(env)

    def _collect(self, out: list[Param]) -> None:
        self.left._collect(out)
        self.right._collect(out)

    def __repr__(self) -> str:
        return f"({self.left!r} + {self.right!r})"


@dataclass(frozen=True, eq=False)
class Mul(ParamNode):
    """A parameter node scaled by a constant factor."""

    node: ParamNode
    factor: float

    def evaluate(self, env: Mapping[Param, float]) -> float:
        return self.node.evaluate(env) * self.factor

    def _collect(self, out: list[Param]) -> None:
        self.node._collect(out)

    def __repr__(self) -> str:
        return f"({self.node!r} * {self.factor:g})"


def _coerce_operand(value: object) -> ParamNode | None:
    """Coerce an algebra operand to a node, or ``None`` if it is not combinable."""
    if isinstance(value, ParamNode):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return Constant(float(value))
    return None


def collect_params(node: ParamNode) -> tuple[Param, ...]:
    """Return the free parameters a node depends on, unique by identity, in order."""
    out: list[Param] = []
    node._collect(out)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #

#: What a profile parameter accepts: nothing (default), a fixed number, an
#: ``(lo, hi)`` interval, an explicit prior, or a named parameter / expression.
Spec = float | int | tuple[float, float] | list[float] | Prior | ParamNode | None


def parse_spec(value: Spec, *, where: str) -> ParamNode:
    """Canonicalise one parameter spec into a :class:`ParamNode`.

    Parameters
    ----------
    value : Spec
        ``None`` (data-driven default), a number (fixed), a length-2 ``(lo, hi)``
        sequence (a :class:`Uniform`), a :class:`Prior` (an anonymous free
        parameter with that prior), or a :class:`ParamNode` (a named parameter or
        an expression, passed through).
    where : str
        Context for error messages (e.g. ``"bulge.n"``).

    Returns
    -------
    ParamNode
        ``Constant`` for a fixed number; an anonymous ``Param`` for ``None`` / a
        prior / an interval; the given node otherwise.

    Raises
    ------
    ValueError
        If ``value`` is a bool, a malformed interval, or an unrecognised type.
    """
    if value is None:
        return Param(name=None, prior=None)
    if isinstance(value, ParamNode):
        return value
    if isinstance(value, Prior):
        return Param(name=None, prior=value)
    if isinstance(value, bool):
        raise ValueError(f"{where}: bool is not a valid spec, got {value!r}.")
    if isinstance(value, (int, float)):
        return Constant(float(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Param(name=None, prior=Uniform(float(value[0]), float(value[1])))
    raise ValueError(
        f"{where}: expected None, a number, an (lo, hi) pair, a Prior, or a "
        f"Param/expression; got {value!r}."
    )
