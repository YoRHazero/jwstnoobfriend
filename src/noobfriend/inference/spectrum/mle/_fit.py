"""Entry point of the MLE backend."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral, Real
from time import perf_counter
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.mle._result import NoobMLEResult, options_mapping
from noobfriend.inference.spectrum.mle._search import search

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


def fit(
    workspace: NoobFitWorkspace,
    *,
    absorption_bound_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    cancellation_threshold: float | None = 0.37,
    relative_likelihood_min: float = 0.5,
    random_seed: int = 1729,
) -> NoobMLEResult:
    """Fit a single-frame workspace by maximum likelihood (see ``ws.mle``)."""
    multipliers = tuple(float(value) for value in absorption_bound_multipliers)
    if not multipliers or not all(
        isfinite(value) and value > 0 for value in multipliers
    ):
        raise ValueError("absorption_bound_multipliers must be positive numbers.")
    if cancellation_threshold is not None and not 0.0 <= cancellation_threshold <= 1.0:
        raise ValueError("cancellation_threshold must lie in [0, 1] or be None.")
    if (
        not isinstance(relative_likelihood_min, Real)
        or not 0.0 < relative_likelihood_min <= 1.0
    ):
        raise ValueError("relative_likelihood_min must lie in (0, 1].")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer.")
    started = perf_counter()
    found = search(
        workspace,
        absorption_bound_multipliers=multipliers,
        cancellation_threshold=cancellation_threshold,
        relative_likelihood_min=float(relative_likelihood_min),
        random_seed=int(random_seed),
    )
    options = options_mapping(
        multipliers,
        cancellation_threshold,
        float(relative_likelihood_min),
        int(random_seed),
    )
    return NoobMLEResult(workspace, found, options, perf_counter() - started)
