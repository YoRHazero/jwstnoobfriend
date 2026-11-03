"""Multi-start search and cancellation-aware selection.

For each absorption-bound variant: five structured starts (centres at zero,
widths at five anchors across their range), four random starts and, with a
free broad flux, one broad high-peak start; every start gets its linear
parameters from the bounded linear subproblem. Each start is optimized by
bounded trust-region least squares. Among candidates within the relative-
likelihood region of the best one, the least emission/absorption
cancellation wins; if that still cancels more than the threshold, an SLSQP
pass minimizes cancellation subject to staying inside the region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

from noobfriend.inference.spectrum.mle._problem import (
    Problem,
    build_problem,
    free_absorption,
    initial_vector,
    width_range,
)

_STRUCTURED = 5
_RANDOM = 4
_BROAD_PEAK_FRACTION = 0.5
_MAX_NFEV = 250
_REFINE_MAXITER = 200


@dataclass(frozen=True, slots=True)
class Start:
    """One starting point of one problem variant."""

    problem: Problem
    kind: str
    x: np.ndarray


@dataclass(frozen=True, slots=True)
class Candidate:
    """One converged optimum.

    Attributes
    ----------
    problem
        The variant it was found in.
    kind
        Start kind (``structured_2``, ``random_0``, ``refinement_1``, ...).
    x
        Physical parameter vector.
    chi2
        Chi-square.
    cancellation_fraction
        Emission/absorption cancellation (see ``Problem``).
    """

    problem: Problem
    kind: str
    x: np.ndarray
    chi2: float
    cancellation_fraction: float

    def relative_likelihood(self, minimum_chi2: float) -> float:
        """``exp(-(chi2 - minimum) / 2)``."""
        return exp(-0.5 * max(0.0, self.chi2 - minimum_chi2))


@dataclass(frozen=True, slots=True)
class Search:
    """Outcome of the multi-start search.

    Attributes
    ----------
    likelihood_optimum
        Lowest chi-square candidate.
    selected
        The cancellation-aware choice (possibly refined).
    start_count, converged_count
        Starts tried and converged.
    refinement_applied
        Whether the SLSQP cancellation pass ran.
    refinement_attempted, refinement_converged
        Its starts and successes.
    """

    likelihood_optimum: Candidate
    selected: Candidate
    start_count: int
    converged_count: int
    refinement_applied: bool
    refinement_attempted: int
    refinement_converged: int


def search(
    workspace: object,
    *,
    absorption_bound_multipliers: Sequence[float],
    cancellation_threshold: float | None,
    relative_likelihood_min: float,
    random_seed: int,
) -> Search:
    """Run every start, select and conditionally refine."""
    probe = build_problem(workspace, absorption_bound_multiplier=1.0)  # type: ignore[arg-type]
    multipliers = (
        tuple(absorption_bound_multipliers) if free_absorption(probe) else (1.0,)
    )
    starts: list[Start] = []
    for variant, multiplier in enumerate(multipliers):
        problem = (
            probe
            if multiplier == 1.0
            else build_problem(workspace, absorption_bound_multiplier=multiplier)  # type: ignore[arg-type]
        )
        starts += _structured(problem)
        starts += _random(problem, np.random.default_rng(random_seed + variant))
        broad = _broad(problem)
        if broad is not None:
            starts.append(broad)
    candidates = [c for start in starts if (c := _optimize(start)) is not None]
    if not candidates:
        raise RuntimeError("MLE: no start converged.")
    optimum = min(candidates, key=lambda c: c.chi2)
    limit = optimum.chi2 - 2.0 * log(relative_likelihood_min)
    eligible = [c for c in candidates if c.chi2 <= limit + 1e-9]
    selected = min(eligible, key=lambda c: (c.cancellation_fraction, c.chi2))
    applied, attempted, converged = False, 0, 0
    if (
        cancellation_threshold is not None
        and selected.cancellation_fraction >= cancellation_threshold
    ):
        applied = True
        widest = max(
            (c.problem for c in candidates), key=lambda p: p.absorption_bound_multiplier
        )
        seeds = _refinement_seeds(optimum, eligible)
        refined = [
            r
            for index, seed in enumerate(seeds)
            if (r := _refine(widest, seed, limit, index)) is not None
        ]
        attempted, converged = len(seeds), len(refined)
        selected = min(
            (selected, *refined), key=lambda c: (c.cancellation_fraction, c.chi2)
        )
    return Search(
        likelihood_optimum=optimum,
        selected=selected,
        start_count=len(starts),
        converged_count=len(candidates),
        refinement_applied=applied,
        refinement_attempted=attempted,
        refinement_converged=converged,
    )


def _structured(problem: Problem) -> list[Start]:
    base = initial_vector(problem)
    widths = [i for i in range(problem.size) if problem.kind(i) == "width"]
    anchors = {i: _anchors(*width_range(problem, i)) for i in widths}
    starts = []
    for index in range(_STRUCTURED):
        x = base.copy()
        for position, i in enumerate(widths):
            x[i] = anchors[i][(index + position) % _STRUCTURED]
        starts.append(
            Start(problem, f"structured_{index}", problem.linearized_start(x))
        )
    return starts


def _random(problem: Problem, rng: np.random.Generator) -> list[Start]:
    base = problem.linearized_start(initial_vector(problem))
    starts = []
    for index in range(_RANDOM):
        x = base.copy()
        for i in range(problem.size):
            kind = problem.kind(i)
            lower, upper, scale = problem.lower[i], problem.upper[i], problem.scale[i]
            if kind == "width":
                low, high = width_range(problem, i)
                x[i] = np.exp(rng.uniform(np.log(low), np.log(high)))
            elif kind in ("center", "fraction", "ratio") and (
                np.isfinite(lower) and np.isfinite(upper)
            ):
                x[i] = rng.uniform(lower, upper)
            elif kind == "center":
                x[i] = rng.normal(0.0, scale)
        for i in np.flatnonzero(problem.linear):
            if problem.rows[i].owner == "continuum":
                x[i] = base[i] + rng.normal(0.0, 0.25) * problem.scale[i]
            elif np.isfinite(problem.upper[i]):
                x[i] = rng.uniform(problem.lower[i], problem.upper[i])
            else:
                x[i] = problem.flux_for_peak(x, i, rng.uniform(0.05, 0.8))
        starts.append(Start(problem, f"random_{index}", problem.clip(x)))
    return starts


def _broad(problem: Problem) -> Start | None:
    lines = {line.name: line for line in problem.workspace.lines}
    broad_fluxes = [
        i
        for i in np.flatnonzero(problem.linear)
        if problem.rows[i].owner in lines
        and lines[problem.rows[i].owner].component == "broad"
    ]
    if not broad_fluxes:
        return None
    x = initial_vector(problem)
    for i in range(problem.size):
        if problem.kind(i) == "width":
            anchors = _anchors(*width_range(problem, i))
            owner = lines.get(problem.rows[i].owner)
            x[i] = (
                anchors[-1]
                if owner is not None and owner.component == "broad"
                else anchors[2]
            )
    x = problem.linearized_start(x)
    for i in broad_fluxes:
        x[i] = problem.flux_for_peak(x, i, _BROAD_PEAK_FRACTION)
    return Start(problem, "broad_high_peak", problem.clip(x))


def _optimize(start: Start) -> Candidate | None:
    from scipy.optimize import least_squares

    problem = start.problem
    lower, upper = problem.lower / problem.scale, problem.upper / problem.scale
    initial = problem.clip(start.x) / problem.scale
    try:
        fitted = least_squares(
            problem.residual_normalized,
            initial,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=_MAX_NFEV,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None
    x = fitted.x * problem.scale
    chi2 = float(fitted.fun @ fitted.fun)
    cancellation = problem.cancellation_fraction(x)
    if not (np.isfinite(chi2) and np.isfinite(cancellation)):
        return None
    return Candidate(problem, start.kind, x, chi2, cancellation)


def _refinement_seeds(optimum: Candidate, eligible: list[Candidate]) -> list[Candidate]:
    best: dict[float, Candidate] = {}
    for candidate in eligible:
        key = candidate.problem.absorption_bound_multiplier
        current = best.get(key)
        if current is None or (candidate.cancellation_fraction, candidate.chi2) < (
            current.cancellation_fraction,
            current.chi2,
        ):
            best[key] = candidate
    unique: list[Candidate] = []
    for candidate in (optimum, *best.values()):
        if not any(
            np.allclose(candidate.x, other.x, rtol=1e-9, atol=1e-12) for other in unique
        ):
            unique.append(candidate)
    return unique


def _refine(
    problem: Problem, seed: Candidate, limit: float, index: int
) -> Candidate | None:
    from scipy.optimize import minimize

    lower, upper = problem.lower / problem.scale, problem.upper / problem.scale
    initial = problem.clip(seed.x) / problem.scale

    def objective(z: np.ndarray) -> float:
        value = problem.cancellation_fraction(z * problem.scale)
        return value if np.isfinite(value) else 1e12

    def inside(z: np.ndarray) -> float:
        return limit - problem.chi2(z * problem.scale)

    try:
        fitted = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=tuple(zip(lower, upper, strict=True)),
            constraints=({"type": "ineq", "fun": inside},),
            options={"maxiter": _REFINE_MAXITER, "ftol": 1e-10, "disp": False},
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None
    x = fitted.x * problem.scale
    chi2 = problem.chi2(x)
    if not np.isfinite(chi2) or chi2 > limit + max(1e-5, abs(limit) * 1e-7):
        return None
    cancellation = problem.cancellation_fraction(x)
    if not np.isfinite(cancellation):
        return None
    return Candidate(problem, f"refinement_{index}", x, chi2, cancellation)


def _anchors(lower: float, upper: float) -> tuple[float, ...]:
    """Five widths across ``[lower, upper]``: edges, harmonic, geometric, arithmetic."""
    span = upper - lower
    return (
        lower + 0.05 * span,
        2.0 * lower * upper / (lower + upper),
        sqrt(lower * upper),
        0.5 * (lower + upper),
        upper - 0.05 * span,
    )
