"""Deterministic multi-bound and multi-start generation for MLE."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.workspace.mle.problem import (
    MLEProblem,
    build_mle_problem,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.mle.options import MLEOptions


_STRUCTURED_STARTS = 5
_RANDOM_STARTS = 4
_BROAD_HIGH_PEAK_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class MLEStart:
    """One physical optimizer start associated with an internal bound variant."""

    problem: MLEProblem
    kind: str
    physical: np.ndarray


def build_mle_starts(
    workspace: NoobFitWorkspace, options: MLEOptions
) -> tuple[MLEStart, ...]:
    """Build all structured, random, and broad exploration starts."""
    probe = build_mle_problem(workspace, absorption_bound_multiplier=1.0)
    multipliers = (
        options.absorption_bound_multipliers if probe.has_free_absorption else (1.0,)
    )
    starts: list[MLEStart] = []
    for variant_index, multiplier in enumerate(multipliers):
        problem = (
            probe
            if multiplier == 1.0
            else build_mle_problem(
                workspace,
                absorption_bound_multiplier=multiplier,
            )
        )
        starts.extend(_structured_starts(problem))
        starts.extend(_random_starts(problem, seed=options.random_seed + variant_index))
        broad = _broad_upper_high_peak_start(problem)
        if broad is not None:
            starts.append(broad)
    return tuple(starts)


def _structured_starts(problem: MLEProblem) -> tuple[MLEStart, ...]:
    anchors = [
        _width_anchors(problem.lower[index], problem.upper[index])
        for index in _indices(problem.shapes_slice)
    ]
    starts: list[MLEStart] = []
    for start_index in range(_STRUCTURED_STARTS):
        physical = problem.initial.copy()
        for local_index, index in enumerate(_indices(problem.center_slice)):
            physical[index] = np.clip(0.0, problem.lower[index], problem.upper[index])
        for local_index, index in enumerate(_indices(problem.shapes_slice)):
            physical[index] = anchors[local_index][
                (start_index + local_index) % _STRUCTURED_STARTS
            ]
        physical = problem.linearized_start(physical)
        starts.append(
            MLEStart(
                problem=problem, kind=f"structured_{start_index}", physical=physical
            )
        )
    return tuple(starts)


def _random_starts(problem: MLEProblem, *, seed: int) -> tuple[MLEStart, ...]:
    rng = np.random.default_rng(seed)
    starts: list[MLEStart] = []
    for start_index in range(_RANDOM_STARTS):
        physical = problem.initial.copy()
        continuum = physical[problem.continuum_slice]
        physical[problem.continuum_slice] = (
            continuum
            + rng.normal(0.0, 0.25, continuum.size)
            * problem.scale[problem.continuum_slice]
        )
        for index in _indices(problem.center_slice):
            physical[index] = rng.uniform(problem.lower[index], problem.upper[index])
        for index in _indices(problem.shapes_slice):
            lower = problem.lower[index]
            upper = problem.upper[index]
            physical[index] = np.exp(rng.uniform(np.log(lower), np.log(upper)))
        for local_index, source_id in enumerate(problem.flux_source_ids):
            index = problem.offsets.flux + local_index
            fwhm = _source_fwhm(problem, source_id, physical)
            upper = problem.upper[index]
            if np.isfinite(upper):
                physical[index] = rng.uniform(problem.lower[index], upper)
            else:
                peak_fraction = rng.uniform(0.05, 0.8)
                physical[index] = problem.flux_for_peak(
                    source_id,
                    fwhm_kms=fwhm,
                    peak_fraction=peak_fraction,
                )
        starts.append(
            MLEStart(
                problem=problem,
                kind=f"random_{start_index}",
                physical=problem.clip(physical),
            )
        )
    return tuple(starts)


def _broad_upper_high_peak_start(problem: MLEProblem) -> MLEStart | None:
    if not problem.has_free_broad_flux:
        return None
    physical = problem.initial.copy()
    for index in _indices(problem.center_slice):
        physical[index] = np.clip(0.0, problem.lower[index], problem.upper[index])
    for source_id, index in zip(
        problem.fwhm_source_ids, _indices(problem.fwhm_slice), strict=True
    ):
        handle = problem.source_handle(source_id)
        anchors = _width_anchors(problem.lower[index], problem.upper[index])
        physical[index] = anchors[-1] if handle.component == "broad" else anchors[2]
    physical = problem.linearized_start(physical)
    for local_index, source_id in enumerate(problem.flux_source_ids):
        handle = problem.source_handle(source_id)
        if handle.component != "broad":
            continue
        index = problem.offsets.flux + local_index
        fwhm = _source_fwhm(problem, source_id, physical)
        physical[index] = problem.flux_for_peak(
            source_id,
            fwhm_kms=fwhm,
            peak_fraction=_BROAD_HIGH_PEAK_FRACTION,
        )
    return MLEStart(
        problem=problem, kind="broad_upper_high_peak", physical=problem.clip(physical)
    )


def _source_fwhm(problem: MLEProblem, source_id: int, physical: np.ndarray) -> float:
    handle = problem.source_handle(source_id)
    expression = problem.graph.fwhm_expressions[handle.index]
    values = dict(
        zip(problem.fwhm_source_ids, physical[problem.fwhm_slice], strict=True)
    )
    return expression.fixed + sum(
        scale * values[key] for key, scale in expression.terms.items()
    )


def _width_anchors(lower: float, upper: float) -> tuple[float, ...]:
    span = upper - lower
    near_lower = lower + 0.05 * span
    near_upper = upper - 0.05 * span
    harmonic = 2.0 * lower * upper / (lower + upper)
    geometric = sqrt(lower * upper)
    arithmetic = 0.5 * (lower + upper)
    return near_lower, harmonic, geometric, arithmetic, near_upper


def _indices(value: slice) -> range:
    if value.start is None or value.stop is None:
        raise RuntimeError("optimizer group slices must have finite endpoints.")
    return range(value.start, value.stop)
