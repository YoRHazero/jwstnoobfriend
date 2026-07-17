"""Candidate optimization and cancellation-aware MLE selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy.optimize import least_squares, minimize

from noobfriend.inference.spectrum.workspace.mle.problem import (
    MLEProblem,
    ModelEvaluation,
)
from noobfriend.inference.spectrum.workspace.mle.starts import MLEStart


_FIRST_LAYER_MAX_NFEV = 250
_SECOND_LAYER_MAXITER = 200


@dataclass(frozen=True, slots=True)
class MLECandidate:
    """One converged first- or second-layer optimizer candidate."""

    problem: MLEProblem
    kind: str
    physical: np.ndarray
    evaluation: ModelEvaluation
    chi2: float
    nfev: int

    def relative_likelihood(self, minimum_chi2: float) -> float:
        """Return likelihood relative to the global first-layer optimum."""
        return exp(-0.5 * max(0.0, self.chi2 - minimum_chi2))


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """First-layer likelihood optimum and cancellation-aware discrete choice."""

    likelihood_optimum: MLECandidate
    selected: MLECandidate
    converged: tuple[MLECandidate, ...]


@dataclass(frozen=True, slots=True)
class RefinementSelection:
    """Conditional second-layer outcome."""

    applied: bool
    selected: MLECandidate
    attempted: int
    converged: int


def optimize_start(start: MLEStart) -> MLECandidate | None:
    """Run bounded nonlinear least squares for one start."""
    problem = start.problem
    lower, upper = problem.normalized_bounds
    initial = problem.to_normalized(problem.clip(start.physical))
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
            max_nfev=_FIRST_LAYER_MAX_NFEV,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None
    physical = problem.to_physical(fitted.x)
    evaluation = problem.evaluate(physical)
    chi2 = float(np.dot(fitted.fun, fitted.fun))
    if not np.isfinite(chi2) or not np.isfinite(evaluation.cancellation_fraction):
        return None
    return MLECandidate(
        problem=problem,
        kind=start.kind,
        physical=physical,
        evaluation=evaluation,
        chi2=chi2,
        nfev=int(fitted.nfev),
    )


def select_candidates(
    candidates: tuple[MLECandidate, ...],
    *,
    relative_likelihood_min: float,
) -> CandidateSelection:
    """Select minimum cancellation inside the accepted likelihood region."""
    if not candidates:
        raise RuntimeError("MLE did not produce any converged candidates.")
    optimum = min(candidates, key=lambda candidate: candidate.chi2)
    chi2_limit = optimum.chi2 - 2.0 * log(relative_likelihood_min)
    eligible = tuple(
        candidate for candidate in candidates if candidate.chi2 <= chi2_limit + 1e-9
    )
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate.evaluation.cancellation_fraction,
            candidate.chi2,
        ),
    )
    return CandidateSelection(
        likelihood_optimum=optimum,
        selected=selected,
        converged=candidates,
    )


def refine_cancellation(
    selection: CandidateSelection,
    *,
    cancellation_threshold: float | None,
    relative_likelihood_min: float,
) -> RefinementSelection:
    """Conditionally minimize cancellation inside the relative-likelihood region."""
    selected = selection.selected
    if (
        cancellation_threshold is None
        or selected.evaluation.cancellation_fraction < cancellation_threshold
    ):
        return RefinementSelection(
            applied=False, selected=selected, attempted=0, converged=0
        )

    widest_problem = max(
        (candidate.problem for candidate in selection.converged),
        key=lambda problem: problem.absorption_bound_multiplier,
    )
    chi2_limit = selection.likelihood_optimum.chi2 - 2.0 * log(relative_likelihood_min)
    starts = _refinement_starts(selection, chi2_limit=chi2_limit)
    refined: list[MLECandidate] = []
    for start_index, candidate in enumerate(starts):
        output = _refine_one(
            widest_problem,
            candidate,
            chi2_limit=chi2_limit,
            start_index=start_index,
        )
        if output is not None:
            refined.append(output)
    final = min(
        (selected, *refined),
        key=lambda candidate: (
            candidate.evaluation.cancellation_fraction,
            candidate.chi2,
        ),
    )
    return RefinementSelection(
        applied=True,
        selected=final,
        attempted=len(starts),
        converged=len(refined),
    )


def _refinement_starts(
    selection: CandidateSelection, *, chi2_limit: float
) -> tuple[MLECandidate, ...]:
    eligible = [
        candidate
        for candidate in selection.converged
        if candidate.chi2 <= chi2_limit + 1e-9
    ]
    by_multiplier: dict[float, MLECandidate] = {}
    for candidate in eligible:
        multiplier = candidate.problem.absorption_bound_multiplier
        current = by_multiplier.get(multiplier)
        if current is None or (
            candidate.evaluation.cancellation_fraction,
            candidate.chi2,
        ) < (
            current.evaluation.cancellation_fraction,
            current.chi2,
        ):
            by_multiplier[multiplier] = candidate

    starts = [selection.likelihood_optimum, *by_multiplier.values()]
    unique: list[MLECandidate] = []
    for candidate in starts:
        if any(
            np.allclose(candidate.physical, item.physical, rtol=1e-9, atol=1e-12)
            for item in unique
        ):
            continue
        unique.append(candidate)
    return tuple(unique)


def _refine_one(
    problem: MLEProblem,
    start: MLECandidate,
    *,
    chi2_limit: float,
    start_index: int,
) -> MLECandidate | None:
    lower, upper = problem.normalized_bounds
    initial = problem.to_normalized(problem.clip(start.physical))

    def objective(normalized: np.ndarray) -> float:
        value = problem.evaluate(problem.to_physical(normalized)).cancellation_fraction
        return value if np.isfinite(value) else 1e12

    def likelihood_constraint(normalized: np.ndarray) -> float:
        return chi2_limit - problem.chi2(problem.to_physical(normalized))

    try:
        fitted = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=tuple(zip(lower, upper, strict=True)),
            constraints=({"type": "ineq", "fun": likelihood_constraint},),
            options={"maxiter": _SECOND_LAYER_MAXITER, "ftol": 1e-10, "disp": False},
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None
    physical = problem.to_physical(fitted.x)
    chi2 = problem.chi2(physical)
    tolerance = max(1e-5, abs(chi2_limit) * 1e-7)
    if not np.isfinite(chi2) or chi2 > chi2_limit + tolerance:
        return None
    evaluation = problem.evaluate(physical)
    if not np.isfinite(evaluation.cancellation_fraction):
        return None
    return MLECandidate(
        problem=problem,
        kind=f"cancellation_refinement_{start_index}",
        physical=physical,
        evaluation=evaluation,
        chi2=chi2,
        nfev=int(getattr(fitted, "nfev", 0)),
    )
