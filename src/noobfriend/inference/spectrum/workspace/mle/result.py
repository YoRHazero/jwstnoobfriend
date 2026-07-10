"""Public result objects for workspace maximum-likelihood fitting."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.workspace import LineHandle, NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.mle.selection import MLECandidate


@dataclass(frozen=True, slots=True)
class MLELineResult:
    """Fitted physical quantities and signed model for one workspace line."""

    handle: LineHandle
    flux: float
    fwhm: float
    center: float
    delta_v_kms: float
    model_flux: np.ndarray


@dataclass(frozen=True, slots=True)
class MLESolution:
    """One complete fitted spectrum solution.

    ``aic`` and ``bic`` use the Gaussian known-error forms ``chi2 + 2k``
    and ``chi2 + k * log(n)``. The shared likelihood normalization constant
    is omitted, so these values are intended for models fit to the same data
    and error arrays. Use the enclosing result's ``likelihood_optimum`` values
    for model comparison; a cancellation-selected solution need not be the
    maximum-likelihood point.
    """

    chi2: float
    reduced_chi2: float
    aic: float
    bic: float
    relative_likelihood: float
    cancellation: float
    cancellation_fraction: float
    continuum_flux: np.ndarray
    model_flux: np.ndarray
    residual: np.ndarray
    lines: tuple[MLELineResult, ...]

    def for_line(self, line: NoobLine) -> MLELineResult:
        """Return a line result by the original ``NoobLine`` object identity."""
        for fitted in self.lines:
            if fitted.handle.line is line:
                return fitted
        raise KeyError("line is not part of this fitted solution.")


@dataclass(frozen=True, slots=True)
class MLEFitResult:
    """Maximum-likelihood result with an explicit cancellation-aware selection."""

    workspace: NoobFitWorkspace
    likelihood_optimum: MLESolution
    solution: MLESolution
    candidate_count: int
    converged_candidate_count: int
    refinement_applied: bool
    refinement_attempt_count: int
    refinement_converged_count: int
    absorption_bound_multipliers: tuple[float, ...]
    cancellation_threshold: float | None
    relative_likelihood_min: float
    random_seed: int
    elapsed_seconds: float

    def summary(self) -> str:
        """Return an HTML summary of fit diagnostics and selected line values."""
        rows = "\n".join(
            f"""      <tr>
        <td><code>{escape(line.handle.id)}</code></td>
        <td>{line.flux:.8g}</td>
        <td>{line.fwhm:.8g}</td>
        <td>{line.center:.8g}</td>
        <td>{line.delta_v_kms:.8g}</td>
      </tr>"""
            for line in self.solution.lines
        )
        refinement = "yes" if self.refinement_applied else "no"
        threshold = "None" if self.cancellation_threshold is None else f"{self.cancellation_threshold:.8g}"
        multipliers = ", ".join(f"{value:.8g}" for value in self.absorption_bound_multipliers)
        return f"""<section class="noob-mle-summary">
  <h2>Maximum-likelihood fit</h2>
  <p>
    <strong>chi2:</strong> {self.solution.chi2:.8g};
    <strong>reduced chi2:</strong> {self.solution.reduced_chi2:.8g};
    <strong>MLE AIC:</strong> {self.likelihood_optimum.aic:.8g};
    <strong>MLE BIC:</strong> {self.likelihood_optimum.bic:.8g};
    <strong>relative likelihood:</strong> {self.solution.relative_likelihood:.8g}
  </p>
  <p>
    <strong>cancellation fraction:</strong> {self.solution.cancellation_fraction:.8g};
    <strong>conditional refinement:</strong> {refinement};
    <strong>candidates:</strong> {self.converged_candidate_count}/{self.candidate_count};
    <strong>elapsed:</strong> {self.elapsed_seconds:.3f} s
  </p>
  <p>
    <strong>absorption bound multipliers:</strong> ({multipliers});
    <strong>cancellation threshold:</strong> {threshold};
    <strong>relative likelihood min:</strong> {self.relative_likelihood_min:.8g};
    <strong>random seed:</strong> {self.random_seed}
  </p>
  <table>
    <thead>
      <tr><th>Line</th><th>Flux</th><th>FWHM (km/s)</th><th>Center</th><th>Delta v (km/s)</th></tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def build_solution(
    workspace: NoobFitWorkspace,
    candidate: MLECandidate,
    *,
    minimum_chi2: float,
) -> MLESolution:
    """Materialize one internal candidate on the full spectrum grid."""
    spectrum = workspace.spectrum
    evaluation = candidate.problem.evaluate(candidate.physical, wavelength=spectrum.obs)
    valid_count = int(spectrum.valid_mask.sum())
    parameter_count = candidate.problem.size
    dof = valid_count - parameter_count
    reduced_chi2 = candidate.chi2 / dof if dof > 0 else float("nan")
    aic = candidate.chi2 + 2.0 * parameter_count
    bic = candidate.chi2 + parameter_count * log(valid_count)
    residual = np.full_like(spectrum.flux, np.nan, dtype=float)
    residual[spectrum.valid_mask] = (
        spectrum.flux[spectrum.valid_mask] - evaluation.model[spectrum.valid_mask]
    ) / spectrum.error[spectrum.valid_mask]

    lines = tuple(
        MLELineResult(
            handle=handle,
            flux=flux,
            fwhm=fwhm,
            center=center,
            delta_v_kms=velocity,
            model_flux=_readonly(model),
        )
        for handle, flux, fwhm, center, velocity, model in zip(
            workspace.handles,
            evaluation.line_fluxes,
            evaluation.fwhm_kms,
            evaluation.centers,
            evaluation.velocity_offsets_kms,
            evaluation.line_models,
            strict=True,
        )
    )
    return MLESolution(
        chi2=candidate.chi2,
        reduced_chi2=reduced_chi2,
        aic=aic,
        bic=bic,
        relative_likelihood=candidate.relative_likelihood(minimum_chi2),
        cancellation=evaluation.cancellation,
        cancellation_fraction=evaluation.cancellation_fraction,
        continuum_flux=_readonly(evaluation.continuum),
        model_flux=_readonly(evaluation.model),
        residual=_readonly(residual),
        lines=lines,
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
