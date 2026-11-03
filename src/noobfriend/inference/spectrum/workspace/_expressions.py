"""Physical quantities as small expressions over the sampled parameters.

A derived line's physical width is ``base width * ratio``, its velocity
offset ``base offset + difference``, and so on down the derive tree. The
expressions evaluate with plain arithmetic, so NumPy arrays, JAX tracers and
broadcast draw stacks all work, and render as text for the summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class Expression:
    """Base class of physical-quantity expressions."""

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Evaluate with free-parameter values by flat name."""
        raise NotImplementedError

    def text(self) -> str:
        """Render as text."""
        raise NotImplementedError

    def references(self) -> tuple[str, ...]:
        """Names of the free or fixed parameters the expression reads."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Ref(Expression):
    """A free parameter by flat name."""

    name: str

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Look the value up."""
        return values[self.name]

    def text(self) -> str:
        """Return the name."""
        return self.name

    def references(self) -> tuple[str, ...]:
        """Return the name."""
        return (self.name,)


@dataclass(frozen=True, slots=True)
class Const(Expression):
    """A fixed value, remembered with the parameter it came from."""

    value: float
    name: str | None = None

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Return the value (a plain float keeps NumPy/JAX shapes scalar)."""
        return self.value

    def text(self) -> str:
        """Render the value (and its parameter name)."""
        return (
            f"{self.value:.6g}"
            if self.name is None
            else f"{self.name}(={self.value:.6g})"
        )

    def references(self) -> tuple[str, ...]:
        """Return the source parameter, if any."""
        return () if self.name is None else (self.name,)


@dataclass(frozen=True, slots=True)
class Sum(Expression):
    """``left + right``."""

    left: Expression
    right: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Add the operands."""
        return self.left.evaluate(values) + self.right.evaluate(values)

    def text(self) -> str:
        """Render ``a + b``."""
        return f"{self.left.text()} + {self.right.text()}"

    def references(self) -> tuple[str, ...]:
        """Combine operand references."""
        return self.left.references() + self.right.references()


@dataclass(frozen=True, slots=True)
class Product(Expression):
    """``left * right``."""

    left: Expression
    right: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Multiply the operands."""
        return self.left.evaluate(values) * self.right.evaluate(values)

    def text(self) -> str:
        """Render ``a * b``."""
        return f"{_wrap(self.left)} * {_wrap(self.right)}"

    def references(self) -> tuple[str, ...]:
        """Combine operand references."""
        return self.left.references() + self.right.references()


@dataclass(frozen=True, slots=True)
class Scale(Expression):
    """``factor * term`` with a constant factor."""

    factor: float
    term: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Scale the operand."""
        return self.factor * self.term.evaluate(values)

    def text(self) -> str:
        """Render ``k * a``."""
        return f"{self.factor:.6g} * {_wrap(self.term)}"

    def references(self) -> tuple[str, ...]:
        """Return the operand's references."""
        return self.term.references()


def simplify(expression: Expression) -> Expression:
    """Fold constant subtrees into one :class:`Const`.

    Keeps a bound quantity whose inputs are all fixed (``fwhm`` locked to a
    fixed base width) a plain float, so profiles can prune exact-zero kernel
    branches.
    """
    if isinstance(expression, (Sum, Product)):
        left, right = simplify(expression.left), simplify(expression.right)
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(type(expression)(left, right).evaluate({}))
        return type(expression)(left, right)
    if isinstance(expression, Scale):
        term = simplify(expression.term)
        if isinstance(term, Const):
            return Const(expression.factor * term.value)
        return Scale(expression.factor, term)
    return expression


def _wrap(expression: Expression) -> str:
    text = expression.text()
    return f"({text})" if isinstance(expression, Sum) else text
