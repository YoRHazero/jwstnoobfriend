"""NoobMLEResult: the selected solution, its error bars and diagnostics.

Error bars come from the Jacobian ``J`` of the error-normalized residuals
with respect to every optimized parameter (linear ones included) at the
selected solution: covariance ``(J^T J)^-1``. A derived quantity (a bound
line's width, a centre) gets its error by the delta method. They are quick
error bars only: they are flagged unreliable when a parameter sits on a
bound or ``J^T J`` is ill-conditioned, and they are never a source for
``as_prior`` (which needs an MCMC posterior).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._text import format_rows
from noobfriend.inference.spectrum.mle._search import Candidate, Search

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.workspace import LineHandle, NoobFitWorkspace

#: Condition number of the scaled ``J^T J`` beyond which errors are unreliable.
CONDITION_LIMIT = 1e12
_STEP = 1e-6


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One fitted value (no arithmetic: use MCMC for derived posteriors).

    Attributes
    ----------
    value
        The fitted value.
    stderr
        Standard error from the Jacobian (delta method for derived values);
        ``nan`` when the covariance is singular.
    at_bound
        Whether the value (or a parameter it depends on) sits on a bound.
    """

    value: float
    stderr: float
    at_bound: bool


@dataclass(frozen=True, slots=True)
class MLESolution:
    """One complete solution.

    ``aic = chi2 + 2k`` and ``bic = chi2 + k ln n`` (Gaussian, known
    errors, constants dropped): compare models on the same data only, and use
    the likelihood optimum's values for model comparison.

    Attributes
    ----------
    x
        Optimized parameter values by flat name.
    values
        Every parameter and line quantity by flat name.
    chi2, reduced_chi2, aic, bic
        Goodness of fit.
    relative_likelihood
        ``exp(-(chi2 - chi2_min) / 2)`` against the likelihood optimum.
    cancellation_fraction
        Emission/absorption cancellation.
    kind
        The start the solution came from.
    """

    x: Mapping[str, float]
    values: Mapping[str, float]
    chi2: float
    reduced_chi2: float
    aic: float
    bic: float
    relative_likelihood: float
    cancellation_fraction: float
    kind: str


class ComponentEstimate:
    """One line's fitted quantities, read like its declaration."""

    _RELATIVE = ("delta_v_from_base", "fwhm_ratio_to_base", "flux_ratio_to_base")

    def __init__(self, result: NoobMLEResult, handle: LineHandle) -> None:
        self._result = result
        self._handle = handle

    @property
    def center(self) -> ParameterEstimate:
        """Line centre (data unit)."""
        return self._result.estimate(f"{self._handle.name}.center")

    @property
    def delta_v(self) -> ParameterEstimate:
        """Velocity offset (km/s)."""
        return self._result.estimate(f"{self._handle.name}.delta_v")

    @property
    def fwhm(self) -> ParameterEstimate:
        """Intrinsic FWHM (km/s)."""
        return self._result.estimate(f"{self._handle.name}.fwhm")

    @property
    def flux(self) -> ParameterEstimate:
        """Integrated flux."""
        return self._result.estimate(f"{self._handle.name}.flux")

    def __getattr__(self, attribute: str) -> Any:
        """Free relative parameters and kernels by kind."""
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        name = f"{self._handle.name}.{attribute}"
        if attribute in self._RELATIVE:
            if name not in self._result.solution.x:
                raise AttributeError(f"{name} is not a free parameter of this fit.")
            return self._result.estimate(name)
        if attribute in self._handle.kernels:
            return _Kernel(self._result, name)
        raise AttributeError(f"line {self._handle.name!r} has no {attribute!r}.")


class _Kernel:
    def __init__(self, result: NoobMLEResult, prefix: str) -> None:
        self._result = result
        self._prefix = prefix

    @property
    def fwhm(self) -> ParameterEstimate:
        """Kernel FWHM (km/s)."""
        return self._result.estimate(f"{self._prefix}.fwhm")

    @property
    def fraction(self) -> ParameterEstimate:
        """Convolved fraction."""
        return self._result.estimate(f"{self._prefix}.fraction")


class ContinuumEstimate:
    """Continuum parameters."""

    def __init__(self, result: NoobMLEResult) -> None:
        self._result = result

    def coefficient(self, degree: int) -> ParameterEstimate:
        """Polynomial coefficient ``c_degree``."""
        return self._result.estimate(f"continuum.c{degree}")

    @property
    def level(self) -> ParameterEstimate:
        """Power-law flux at ``lambda_0``."""
        return self._result.estimate("continuum.level")

    @property
    def index(self) -> ParameterEstimate:
        """Power-law index."""
        return self._result.estimate("continuum.index")


class NoobMLEResult:
    """Maximum-likelihood fit of a single-frame workspace.

    ``result["Ha"].fwhm``, ``result["continuum"].coefficient(0)`` and flat
    names (``result["Ha.fwhm"]``) return :class:`ParameterEstimate` objects
    of the selected (cancellation-aware) solution.

    Attributes
    ----------
    workspace : NoobFitWorkspace
        The fit plan.
    solution : MLESolution
        The selected solution.
    likelihood_optimum : MLESolution
        The lowest chi-square solution (use its AIC/BIC for model comparison).
    parameter_names : tuple of str
        Optimized parameters, the order of :attr:`covariance`.
    covariance : numpy.ndarray
        ``(J^T J)^-1`` at the selected solution.
    reliable : bool
        ``False`` when a parameter sits on a bound or ``J^T J`` is
        ill-conditioned.
    reliability_notes : tuple of str
        Why the error bars are unreliable.
    start_count, converged_count : int
        Starts tried and converged.
    refinement_applied : bool
        Whether the cancellation refinement ran.
    refinement_attempted, refinement_converged : int
        Its starts and successes.
    options : Mapping
        The ``mle()`` arguments.
    elapsed_seconds : float
        Wall time.
    """

    def __init__(
        self,
        workspace: NoobFitWorkspace,
        found: Search,
        options: Mapping[str, Any],
        elapsed_seconds: float,
    ) -> None:
        self.workspace = workspace
        self._problem = found.selected.problem
        minimum = found.likelihood_optimum.chi2
        self.solution = _solution(found.selected, minimum)
        self.likelihood_optimum = _solution(found.likelihood_optimum, minimum)
        self.parameter_names = self._problem.names
        self._x = np.asarray(found.selected.x, dtype=float)
        self.covariance, notes = _covariance(self._problem, self._x)
        self._at_bound = _at_bound(self._problem, self._x)
        if self._at_bound:
            notes.append("at a bound: " + ", ".join(self._at_bound))
        self.reliable = not notes
        self.reliability_notes = tuple(notes)
        self.start_count = found.start_count
        self.converged_count = found.converged_count
        self.refinement_applied = found.refinement_applied
        self.refinement_attempted = found.refinement_attempted
        self.refinement_converged = found.refinement_converged
        self.options = MappingProxyType(dict(options))
        self.elapsed_seconds = elapsed_seconds

    @property
    def aic(self) -> float:
        """AIC of the likelihood optimum."""
        return self.likelihood_optimum.aic

    @property
    def bic(self) -> float:
        """BIC of the likelihood optimum."""
        return self.likelihood_optimum.bic

    def __getitem__(
        self, name: str
    ) -> ComponentEstimate | ContinuumEstimate | ParameterEstimate:
        """Return a component by name, or one value by flat dotted name."""
        if "." in name:
            return self.estimate(name)
        if name == "continuum":
            return ContinuumEstimate(self)
        return ComponentEstimate(self, self.workspace[name])  # type: ignore[arg-type]

    def estimate(self, name: str) -> ParameterEstimate:
        """Value, standard error and bound flag of any parameter or quantity."""
        values = self.solution.values
        if name not in values:
            raise KeyError(f"no {name!r} in this fit.")
        gradient = self._gradient(name)
        variance = (
            float(gradient @ self.covariance @ gradient) if gradient.size else 0.0
        )
        depends = [
            other
            for other, g in zip(self.parameter_names, gradient, strict=True)
            if g != 0.0
        ]
        return ParameterEstimate(
            value=float(values[name]),
            stderr=float(np.sqrt(variance)) if variance >= 0 else float("nan"),
            at_bound=any(other in self._at_bound for other in depends),
        )

    def summary(self) -> str:
        """Return the plan, fit statistics and estimates as text."""
        solution = self.solution
        lines = [
            self.workspace.summary(),
            "",
            "Maximum likelihood (raw errors, white noise, priors as bounds only)",
            f"  chi2 {solution.chi2:.6g} (reduced {solution.reduced_chi2:.4g}), "
            f"relative likelihood {solution.relative_likelihood:.4g}, "
            f"cancellation {solution.cancellation_fraction:.4g}",
            f"  likelihood optimum: chi2 {self.likelihood_optimum.chi2:.6g}, "
            f"AIC {self.aic:.6g}, BIC {self.bic:.6g}",
            f"  starts {self.converged_count}/{self.start_count} converged; refinement "
            + (
                f"{self.refinement_converged}/{self.refinement_attempted}"
                if self.refinement_applied
                else "not needed"
            )
            + f"; {self.elapsed_seconds:.2f} s",
            "  error bars: "
            + (
                "reliable"
                if self.reliable
                else "UNRELIABLE (" + "; ".join(self.reliability_notes) + ")"
            ),
            "",
            "Estimates (selected solution)",
        ]
        units = {row.name: row.unit for row in self.workspace.parameters.values()}
        units |= {q.name: q.unit for q in self.workspace.parameters.quantities.values()}
        names = list(self.workspace.parameters.quantities) + [
            name
            for name in self.parameter_names
            if name not in self.workspace.parameters.quantities
        ]
        rows = []
        for name in names:
            estimate = self.estimate(name)
            rows.append(
                [
                    name,
                    units.get(name, "") or "-",
                    f"{estimate.value:.6g}",
                    f"{estimate.stderr:.3g}",
                    "yes" if estimate.at_bound else "",
                ]
            )
        lines += format_rows(["name", "unit", "value", "stderr", "at bound"], rows)
        return "\n".join(lines)

    def plot(self) -> Figure:
        """Plot data, the selected model, its components and the residual.

        Returns
        -------
        matplotlib.figure.Figure
            Axes: spectrum, then residual (in units of the error).
        """
        from noobfriend.inference.spectrum.mle._plot import plot_mle

        return plot_mle(self)

    def __repr__(self) -> str:
        """Show chi-square and reliability."""
        return (
            f"NoobMLEResult({self.workspace!r}, chi2={self.solution.chi2:.6g}, "
            f"reliable={self.reliable})"
        )

    def _gradient(self, name: str) -> np.ndarray:
        """Differentiate one value with respect to the optimized parameters."""
        problem = self._problem
        if name in self.parameter_names:
            gradient = np.zeros(problem.size)
            gradient[self.parameter_names.index(name)] = 1.0
            return gradient
        gradient = np.empty(problem.size)
        for i in range(problem.size):
            step = _STEP * max(abs(self._x[i]), problem.scale[i])
            up, down = self._x.copy(), self._x.copy()
            up[i] += step
            down[i] -= step
            gradient[i] = (problem.values(up)[name] - problem.values(down)[name]) / (
                2 * step
            )
        return gradient


def _solution(candidate: Candidate, minimum_chi2: float) -> MLESolution:
    problem = candidate.problem
    n = problem.data.size
    k = problem.size
    values = problem.values(candidate.x)
    return MLESolution(
        x=MappingProxyType(
            dict(zip(problem.names, map(float, candidate.x), strict=True))
        ),
        values=MappingProxyType({key: float(value) for key, value in values.items()}),
        chi2=candidate.chi2,
        reduced_chi2=candidate.chi2 / max(n - k, 1),
        aic=candidate.chi2 + 2.0 * k,
        bic=candidate.chi2 + k * log(n),
        relative_likelihood=candidate.relative_likelihood(minimum_chi2),
        cancellation_fraction=candidate.cancellation_fraction,
        kind=candidate.kind,
    )


def _covariance(problem: Any, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """``(J^T J)^-1`` in physical units, from central differences."""
    z = x / problem.scale
    jacobian = np.empty((problem.data.size, problem.size))
    for i in range(problem.size):
        step = _STEP * max(1.0, abs(z[i]))
        up, down = z.copy(), z.copy()
        up[i] = min(z[i] + step, problem.upper[i] / problem.scale[i])
        down[i] = max(z[i] - step, problem.lower[i] / problem.scale[i])
        jacobian[:, i] = (
            problem.residual_normalized(up) - problem.residual_normalized(down)
        ) / (up[i] - down[i])
    information = jacobian.T @ jacobian
    notes = []
    condition = np.linalg.cond(information) if information.size else 1.0
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        notes.append(f"J^T J is ill-conditioned (condition {condition:.3g})")
    try:
        scaled = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        notes.append("J^T J is singular")
        scaled = np.full_like(information, np.nan)
    # J is taken in scaled coordinates z = x / scale; map back to physical units.
    return scaled * np.outer(problem.scale, problem.scale), notes


def _at_bound(problem: Any, x: np.ndarray) -> tuple[str, ...]:
    names = []
    for i, name in enumerate(problem.names):
        tolerance = 1e-6 * max(problem.scale[i], abs(x[i]))
        if (
            abs(x[i] - problem.lower[i]) <= tolerance
            or abs(problem.upper[i] - x[i]) <= tolerance
        ):
            names.append(name)
    return tuple(names)


def options_mapping(
    absorption_bound_multipliers: Sequence[float],
    cancellation_threshold: float | None,
    relative_likelihood_min: float,
    random_seed: int,
) -> dict[str, Any]:
    """Return the ``mle()`` arguments as a plain mapping."""
    return {
        "absorption_bound_multipliers": tuple(absorption_bound_multipliers),
        "cancellation_threshold": cancellation_threshold,
        "relative_likelihood_min": relative_likelihood_min,
        "random_seed": random_seed,
    }
