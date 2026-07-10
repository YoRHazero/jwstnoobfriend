"""Public orchestration for workspace maximum-likelihood fitting."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.workspace.mle.options import build_mle_options
from noobfriend.inference.spectrum.workspace.mle.result import MLEFitResult, build_solution
from noobfriend.inference.spectrum.workspace.mle.selection import (
    optimize_start,
    refine_cancellation,
    select_candidates,
)
from noobfriend.inference.spectrum.workspace.mle.starts import build_mle_starts

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


def fit_workspace_mle(
    workspace: NoobFitWorkspace,
    *,
    absorption_bound_multipliers: Sequence[float],
    cancellation_threshold: float | None,
    relative_likelihood_min: float,
    random_seed: int,
) -> MLEFitResult:
    """Fit one prepared workspace with the built-in robust MLE workflow."""
    started = perf_counter()
    options = build_mle_options(
        absorption_bound_multipliers=absorption_bound_multipliers,
        cancellation_threshold=cancellation_threshold,
        relative_likelihood_min=relative_likelihood_min,
        random_seed=random_seed,
    )
    starts = build_mle_starts(workspace, options)
    candidates = tuple(candidate for start in starts if (candidate := optimize_start(start)) is not None)
    selection = select_candidates(candidates, relative_likelihood_min=options.relative_likelihood_min)
    refinement = refine_cancellation(
        selection,
        cancellation_threshold=options.cancellation_threshold,
        relative_likelihood_min=options.relative_likelihood_min,
    )
    minimum_chi2 = selection.likelihood_optimum.chi2
    return MLEFitResult(
        workspace=workspace,
        likelihood_optimum=build_solution(
            workspace,
            selection.likelihood_optimum,
            minimum_chi2=minimum_chi2,
        ),
        solution=build_solution(
            workspace,
            refinement.selected,
            minimum_chi2=minimum_chi2,
        ),
        candidate_count=len(starts),
        converged_candidate_count=len(candidates),
        refinement_applied=refinement.applied,
        refinement_attempt_count=refinement.attempted,
        refinement_converged_count=refinement.converged,
        absorption_bound_multipliers=options.absorption_bound_multipliers,
        cancellation_threshold=options.cancellation_threshold,
        relative_likelihood_min=options.relative_likelihood_min,
        random_seed=options.random_seed,
        elapsed_seconds=perf_counter() - started,
    )
