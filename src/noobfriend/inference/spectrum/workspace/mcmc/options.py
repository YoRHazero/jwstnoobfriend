"""Validated sampling options for workspace MCMC fitting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class MCMCOptions:
    """Normalized internal representation of public sampling options."""

    draws: int
    tune: int
    chains: int
    cores: int
    target_accept: float
    random_seed: int
    progressbar: bool


def build_mcmc_options(
    *,
    draws: int,
    tune: int,
    chains: int,
    cores: int | None,
    target_accept: float,
    random_seed: int,
    progressbar: bool,
) -> MCMCOptions:
    """Validate public MCMC options and resolve the usable core count."""
    parsed_draws = _positive_integer(draws, name="draws")
    parsed_tune = _nonnegative_integer(tune, name="tune")
    parsed_chains = _positive_integer(chains, name="chains")
    available = max(os.cpu_count() or 1, 1)
    requested_cores = (
        available if cores is None else _positive_integer(cores, name="cores")
    )
    parsed_cores = min(requested_cores, parsed_chains, available)

    if isinstance(target_accept, bool) or not isinstance(target_accept, Real):
        raise TypeError("target_accept must be a finite number.")
    parsed_target = float(target_accept)
    if not isfinite(parsed_target):
        raise ValueError("target_accept must be finite.")
    if not 0.8 <= parsed_target < 1.0:
        raise ValueError("target_accept must be at least 0.8 and less than 1.")

    parsed_seed = _nonnegative_integer(random_seed, name="random_seed")
    if not isinstance(progressbar, bool):
        raise TypeError("progressbar must be bool.")

    return MCMCOptions(
        draws=parsed_draws,
        tune=parsed_tune,
        chains=parsed_chains,
        cores=parsed_cores,
        target_accept=parsed_target,
        random_seed=parsed_seed,
        progressbar=progressbar,
    )


def _positive_integer(value: int, *, name: str) -> int:
    parsed = _nonnegative_integer(value, name=name)
    if parsed == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return parsed
