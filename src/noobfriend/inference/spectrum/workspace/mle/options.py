"""Validated advanced options for workspace maximum-likelihood fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class MLEOptions:
    """Normalized internal representation of public MLE options."""

    absorption_bound_multipliers: tuple[float, ...]
    cancellation_threshold: float | None
    relative_likelihood_min: float
    random_seed: int


def build_mle_options(
    *,
    absorption_bound_multipliers: Sequence[float],
    cancellation_threshold: float | None,
    relative_likelihood_min: float,
    random_seed: int,
) -> MLEOptions:
    """Validate public MLE options and normalize immutable values."""
    multipliers = _normalize_multipliers(absorption_bound_multipliers)
    threshold = None if cancellation_threshold is None else _finite_float(
        cancellation_threshold,
        name="cancellation_threshold",
    )
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("cancellation_threshold must be between 0 and 1, or None.")

    likelihood = _finite_float(relative_likelihood_min, name="relative_likelihood_min")
    if not 0.0 < likelihood <= 1.0:
        raise ValueError("relative_likelihood_min must be greater than 0 and at most 1.")

    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be a nonnegative integer.")
    seed = int(random_seed)
    if seed < 0:
        raise ValueError("random_seed must be a nonnegative integer.")

    return MLEOptions(
        absorption_bound_multipliers=multipliers,
        cancellation_threshold=threshold,
        relative_likelihood_min=likelihood,
        random_seed=seed,
    )


def _normalize_multipliers(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("absorption_bound_multipliers must be a sequence of positive numbers.")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError("absorption_bound_multipliers must be a sequence of positive numbers.") from error
    parsed = tuple(_finite_float(value, name="absorption_bound_multipliers") for value in iterator)
    if not parsed:
        raise ValueError("absorption_bound_multipliers must not be empty.")
    if any(value <= 0.0 for value in parsed):
        raise ValueError("absorption_bound_multipliers must contain only positive values.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("absorption_bound_multipliers must contain unique values.")
    return tuple(sorted(parsed))


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        message = f"{name} must contain only finite numbers." if name.endswith("multipliers") else f"{name} must be a finite number."
        raise TypeError(message)
    parsed = float(value)
    if not isfinite(parsed):
        message = f"{name} must contain only finite values." if name.endswith("multipliers") else f"{name} must be finite."
        raise ValueError(message)
    return parsed
