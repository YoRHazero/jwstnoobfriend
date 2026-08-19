"""Shared numeric helpers for spectrum inference.

Small utilities that several unrelated layers need — array hygiene, posterior
summary statistics, and draw subsampling. They live here rather than in any
one backend so the MCMC results, the MLE results, and the plotting adapters
can all reach them without depending on each other.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import ArrayLike


def readonly(value: ArrayLike) -> np.ndarray:
    """Return an immutable float copy of ``value``.

    Result objects hand arrays to callers directly; copying them read-only
    keeps a caller from mutating state that other views share.
    """
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def highest_density_interval(
    samples: np.ndarray, *, probability: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the narrowest interval covering ``probability`` along axis 0.

    Works pointwise: a 1-D array of draws yields the scalar bounds of one
    parameter, while a ``(draws, points)`` array yields one interval per
    point. The interval is the shortest window of sorted draws that contains
    the requested probability mass, which is the highest-density interval for
    a unimodal posterior.

    Parameters
    ----------
    samples
        Posterior draws along the leading axis.
    probability
        Interval mass in ``(0, 1)``.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper bounds, shaped like ``samples`` without its leading
        axis.
    """
    ordered = np.sort(np.asarray(samples, dtype=float), axis=0)
    n_draws = ordered.shape[0]
    width = max(int(np.floor(probability * n_draws)), 1)
    if width >= n_draws:
        return ordered[0], ordered[-1]
    widths = ordered[width:] - ordered[: n_draws - width]
    starts = np.asarray(np.argmin(widths, axis=0))
    lower = np.take_along_axis(ordered, starts[None, ...], axis=0)[0]
    upper = np.take_along_axis(ordered, (starts + width)[None, ...], axis=0)[0]
    return lower, upper


def draw_indices(
    total_draws: int,
    *,
    max_draws: int | None,
    random_seed: int,
    name: str = "max_draws",
) -> np.ndarray:
    """Return sorted indices of a posterior draw subsample.

    Every draw is used when ``max_draws`` is ``None`` or exceeds the number
    available; otherwise a seeded sample without replacement is drawn and
    sorted, so a given seed always selects the same draws.

    Parameters
    ----------
    total_draws
        Number of posterior draws available.
    max_draws
        Cap on the draws to use, or ``None`` for all of them.
    random_seed
        Nonnegative seed for the subsample.
    name
        Name of ``max_draws`` in the caller's public signature, used in error
        messages.

    Raises
    ------
    TypeError
        If ``max_draws`` is not an integer or ``None``.
    ValueError
        If ``max_draws`` is not positive, or ``random_seed`` is not a
        nonnegative integer.
    """
    if max_draws is not None:
        if isinstance(max_draws, bool) or not isinstance(max_draws, int):
            raise TypeError(f"{name} must be a positive integer or None.")
        if max_draws <= 0:
            raise ValueError(f"{name} must be a positive integer or None.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer.")
    if max_draws is None or max_draws >= total_draws:
        return np.arange(total_draws)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(total_draws, size=max_draws, replace=False))


def validate_probability(value: float) -> float:
    """Return a validated interval probability strictly inside ``(0, 1)``."""
    probability = float(value)
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("hdi_probability must be finite and between 0 and 1.")
    return probability
