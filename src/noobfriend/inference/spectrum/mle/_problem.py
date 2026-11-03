"""The bounded least-squares problem of a single-frame workspace.

Reading rules (the MLE is a quick-look backend):

- the frame's raw error array with white noise; the ``Noise`` model is
  ignored;
- joint priors are ignored; their members are bounded by their domain;
- every other prior contributes only its support, as optimizer bounds
  (``Uniform()`` / ``LogUniform()`` without bounds are unbounded or
  half-bounded);
- fixed values and derive-tree bindings apply as usual.

The vector being optimized holds every free row except noise rows, in table
order, divided by each row's natural scale. Continuum coefficients (the
power-law level included) and independent line fluxes enter the model
linearly; they form the linear subproblem used to build starting points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._xp import numpy_namespace

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace import (
        FrameModel,
        NoobFitWorkspace,
        Parameter,
    )

_CENTER_ROLES = ("delta_v", "delta_wavelength", "delta_v_from_base")


@dataclass(frozen=True, slots=True)
class Problem:
    """One optimizer problem (one internal absorption-bound variant).

    Attributes
    ----------
    workspace
        The fit plan.
    frame_id
        The single frame.
    rows
        Optimized parameters, in order.
    lower, upper
        Physical bounds.
    scale
        Natural scales; the optimizer works on ``value / scale``.
    linear
        Which parameters enter the model linearly.
    wavelength, data, error
        Fitted pixels.
    absorption_bound_multiplier
        Multiplier of the internal upper bound on free absorption fluxes.
    """

    workspace: NoobFitWorkspace
    frame_id: str
    rows: tuple[Parameter, ...]
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray
    linear: np.ndarray
    wavelength: np.ndarray
    data: np.ndarray
    error: np.ndarray
    absorption_bound_multiplier: float

    @property
    def size(self) -> int:
        """Number of optimized parameters."""
        return len(self.rows)

    @property
    def names(self) -> tuple[str, ...]:
        """Flat names of the optimized parameters."""
        return tuple(row.name for row in self.rows)

    def kind(self, index: int) -> str:
        """Role group of one parameter: center, width, fraction, ratio, linear or other."""
        row = self.rows[index]
        role = row.role
        if self.linear[index]:
            return "linear"
        if role in _CENTER_ROLES:
            return "center"
        if role == "fwhm" or role.endswith(".fwhm"):
            return "width"
        if role.endswith(".fraction"):
            return "fraction"
        if role.endswith("_ratio_to_base"):
            return "ratio"
        return "other"

    def values(self, x: np.ndarray) -> dict[str, Any]:
        """Physical values of every parameter and line quantity."""
        values: dict[str, Any] = {
            name: float(v) for name, v in zip(self.names, x, strict=True)
        }
        for row in self.workspace.parameters.free():
            if row.name not in values:
                # Noise rows: the MLE uses raw errors and white noise.
                values[row.name] = 1.0 if row.role == "noise.scale" else 0.0
        return self.workspace.physical(values)

    def model(self, x: np.ndarray, wavelength: np.ndarray | None = None) -> FrameModel:
        """Mean model (continuum, components, total) at ``x``."""
        wavelength = self.wavelength if wavelength is None else wavelength
        return self.workspace.frame_model(
            numpy_namespace(), self.values(x), self.frame_id, wavelength
        )

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Error-normalized residuals ``(model - data) / error``."""
        return (np.asarray(self.model(x).total) - self.data) / self.error

    def residual_normalized(self, z: np.ndarray) -> np.ndarray:
        """Residuals as a function of scaled coordinates."""
        return self.residual(z * self.scale)

    def chi2(self, x: np.ndarray) -> float:
        """Chi-square at ``x``."""
        residual = self.residual(x)
        return float(residual @ residual)

    def cancellation_fraction(self, x: np.ndarray) -> float:
        """Emission flux cancelled by absorption, over the total line flux.

        ``2 * integral(min(emission, absorption))`` divided by the summed
        absolute line fluxes; zero without absorption.
        """
        values = self.values(x)
        components = self.model(x).components
        emission = np.zeros_like(self.wavelength)
        absorption = np.zeros_like(self.wavelength)
        total_flux = 0.0
        for line in self.workspace.lines:
            curve = np.abs(np.asarray(components[line.name], dtype=float))
            total_flux += abs(float(values[f"{line.name}.flux"]))
            if line.sign > 0:
                emission += curve
            else:
                absorption += curve
        if total_flux <= 0.0:
            return 0.0
        overlap = np.trapezoid(np.minimum(emission, absorption), self.wavelength)
        return 2.0 * float(overlap) / total_flux

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Move ``x`` strictly inside the bounds."""
        value = np.asarray(x, dtype=float).copy()
        for i in range(self.size):
            lower, upper, scale = self.lower[i], self.upper[i], self.scale[i]
            if np.isfinite(lower) and np.isfinite(upper):
                margin = min(
                    0.1 * (upper - lower), max(abs(lower), abs(upper), scale) * 1e-10
                )
                value[i] = np.clip(value[i], lower + margin, upper - margin)
            elif np.isfinite(lower):
                value[i] = max(value[i], lower + max(abs(lower), scale) * 1e-10)
            elif np.isfinite(upper):
                value[i] = min(value[i], upper - max(abs(upper), scale) * 1e-10)
        return value

    def linear_columns(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Model with the linear parameters at zero, and one column per linear parameter.

        The model is affine in the linear parameters at fixed nonlinear ones,
        so these columns are exact.
        """
        base_x = np.where(self.linear, 0.0, x)
        base = np.asarray(self.model(base_x).total, dtype=float)
        columns = []
        for i in np.flatnonzero(self.linear):
            unit = base_x.copy()
            unit[i] = 1.0
            columns.append(np.asarray(self.model(unit).total, dtype=float) - base)
        matrix = np.column_stack(columns) if columns else np.zeros((base.size, 0))
        return base, matrix

    def linearized_start(self, x: np.ndarray) -> np.ndarray:
        """Solve the bounded linear subproblem at fixed nonlinear parameters."""
        from scipy.optimize import lsq_linear

        start = self.clip(x)
        if not self.linear.any():
            return start
        base, matrix = self.linear_columns(start)
        weighted = matrix / self.error[:, None]
        target = (self.data - base) / self.error
        index = np.flatnonzero(self.linear)
        try:
            solution = lsq_linear(
                weighted,
                target,
                bounds=(self.lower[index], self.upper[index]),
                lsmr_tol="auto",
                max_iter=200,
            )
        except (ValueError, np.linalg.LinAlgError):
            return start
        if not solution.success or not np.all(np.isfinite(solution.x)):
            return start
        start[index] = solution.x
        return self.clip(start)

    def flux_for_peak(self, x: np.ndarray, index: int, fraction: float) -> float:
        """Linear-parameter value whose column peaks at ``fraction`` of the data range."""
        base, matrix = self.linear_columns(x)
        column = matrix[:, list(np.flatnonzero(self.linear)).index(index)]
        peak = float(np.max(np.abs(column)))
        amplitude = max(float(np.ptp(self.data)), float(np.median(self.error)))
        return fraction * amplitude / peak if peak > 0 else self.scale[index]


def build_problem(
    workspace: NoobFitWorkspace, *, absorption_bound_multiplier: float
) -> Problem:
    """Build the optimizer problem of a single-frame workspace.

    Raises
    ------
    ValueError
        If the workspace has several frames.
    """
    if len(workspace.frame_ids) != 1:
        raise ValueError(
            "mle() fits one frame; select one with spec.select(frame_id) and "
            "prepare again (use model().sample() for joint multi-frame fits)."
        )
    (frame_id,) = workspace.frame_ids
    frame = workspace.spectrum.frames[frame_id]
    mask = workspace.fitted[frame_id]
    wavelength = np.asarray(frame.obs)[mask]
    data = np.asarray(frame.flux)[mask]
    error = np.asarray(frame.error)[mask]
    line_names = {line.name: line for line in workspace.lines}
    rows = tuple(
        row for row in workspace.parameters.free() if not row.owner.startswith("frame.")
    )
    lower, upper, scale, linear = [], [], [], []
    for row in rows:
        low, high = row.prior.bounds()  # type: ignore[union-attr]
        is_linear = (row.owner == "continuum" and row.role != "index") or (
            row.owner in line_names and row.role == "flux"
        )
        lower.append(low)
        upper.append(high)
        scale.append(max(row.scale, np.finfo(float).tiny))
        linear.append(is_linear)
    problem = Problem(
        workspace=workspace,
        frame_id=frame_id,
        rows=rows,
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        scale=np.asarray(scale, dtype=float),
        linear=np.asarray(linear, dtype=bool),
        wavelength=wavelength,
        data=data,
        error=error,
        absorption_bound_multiplier=absorption_bound_multiplier,
    )
    return _bound_absorption(problem, line_names)


def free_absorption(problem: Problem) -> list[int]:
    """Return indices of independent absorption fluxes without a finite upper bound."""
    lines = {line.name: line for line in problem.workspace.lines}
    return [
        i
        for i, row in enumerate(problem.rows)
        if row.role == "flux"
        and row.owner in lines
        and lines[row.owner].sign < 0
        and not np.isfinite(problem.upper[i])
    ]


def _bound_absorption(problem: Problem, lines: Mapping[str, Any]) -> Problem:
    """Cap free absorption fluxes at ``multiplier`` times a data-scale flux.

    The cap is the flux at which the line, at the widest width it can take,
    reaches the data's peak-to-peak range: without it, emission and
    absorption can grow together without bound while cancelling.
    """
    indices = free_absorption(problem)
    if not indices:
        return problem
    widest = problem.clip(_widest(problem))
    base, matrix = problem.linear_columns(widest)
    linear_index = list(np.flatnonzero(problem.linear))
    amplitude = float(np.ptp(problem.data)) or float(np.median(problem.error))
    upper = problem.upper.copy()
    for i in indices:
        column = matrix[:, linear_index.index(i)]
        peak = float(np.max(np.abs(column)))
        if peak > 0:
            upper[i] = problem.absorption_bound_multiplier * amplitude / peak
    return Problem(
        workspace=problem.workspace,
        frame_id=problem.frame_id,
        rows=problem.rows,
        lower=problem.lower,
        upper=upper,
        scale=problem.scale,
        linear=problem.linear,
        wavelength=problem.wavelength,
        data=problem.data,
        error=problem.error,
        absorption_bound_multiplier=problem.absorption_bound_multiplier,
    )


def _widest(problem: Problem) -> np.ndarray:
    """Return a vector with every width at its upper bound (or 20x its scale)."""
    x = initial_vector(problem)
    for i in range(problem.size):
        if problem.kind(i) == "width":
            x[i] = (
                problem.upper[i]
                if np.isfinite(problem.upper[i])
                else 20.0 * problem.scale[i]
            )
    return x


def initial_vector(problem: Problem) -> np.ndarray:
    """Return a neutral starting vector before the linear solve."""
    x = np.empty(problem.size)
    for i in range(problem.size):
        lower, upper, scale = problem.lower[i], problem.upper[i], problem.scale[i]
        kind = problem.kind(i)
        if kind == "center" or problem.rows[i].role == "index":
            x[i] = 0.0
        elif kind == "fraction":
            x[i] = 0.5 * (lower + upper)
        elif kind == "ratio":
            x[i] = 0.5 * (lower + upper) if np.isfinite(upper) else 1.0
        elif kind == "width":
            x[i] = np.sqrt(lower * upper) if np.isfinite(upper) and lower > 0 else scale
        elif kind == "linear":
            x[i] = 0.25 * scale if lower >= 0 else 0.0
        else:
            x[i] = (
                0.5 * (lower + upper)
                if (np.isfinite(lower) and np.isfinite(upper))
                else scale
            )
    return problem.clip(x)


def width_range(problem: Problem, index: int) -> tuple[float, float]:
    """Finite range spanned by width starting points."""
    lower, upper, scale = (
        problem.lower[index],
        problem.upper[index],
        problem.scale[index],
    )
    low = lower if lower > 0 else scale / 20.0
    high = upper if np.isfinite(upper) else 20.0 * scale
    return max(low, np.finfo(float).tiny), max(high, low * (1 + 1e-6))
