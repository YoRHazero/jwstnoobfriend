"""Per-frame model reconstruction and goodness-of-fit for joint MCMC results.

A joint fit shares its line parameters across frames while giving each frame its
own continuum. These tools evaluate the posterior model on every frame's native
(valid) wavelength grid: the posterior-median curve, an optional pointwise
highest-density band (the model's parameter uncertainty), and a per-frame
chi-square. Comparing the per-frame chi-square is how inter-visit systematics
surface: a shared line model that fits one exposure but not another shows up as
an elevated chi-square on the discrepant frame.

The per-draw reconstruction helpers here (draw subsampling, line parameter
extraction, and per-draw line evaluation) are shared with the frame-level
leave-one-out machinery in :mod:`~.lofo`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    contribution_sign,
    profile_template,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.mcmc.result import MCMCFitResult


@dataclass(frozen=True, slots=True, repr=False)
class FrameFit:
    """Posterior model of one frame on its native valid pixels.

    ``components`` maps each line id to that line's contribution drawn on the
    continuum (continuum plus the signed line), matching the plotting
    convention. ``model`` is the posterior-median total model. ``model_lower``
    and ``model_upper`` hold the pointwise highest-density band of the total
    model when a band was requested, else ``None``. ``chi_square`` sums the
    squared standardized residuals of the median model over the valid pixels.
    """

    frame_id: str
    wavelength: np.ndarray
    data: np.ndarray
    error: np.ndarray
    continuum: np.ndarray
    components: Mapping[str, np.ndarray]
    model: np.ndarray
    chi_square: float
    n_pixels: int
    model_lower: np.ndarray | None = None
    model_upper: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Freeze the component mapping while preserving workspace order."""
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @property
    def chi_square_per_pixel(self) -> float:
        """Chi-square divided by the number of valid pixels."""
        return self.chi_square / self.n_pixels if self.n_pixels else float("nan")

    def __repr__(self) -> str:
        """Return a compact per-frame goodness-of-fit summary."""
        return (
            f"FrameFit(frame_id={self.frame_id!r}, n_pixels={self.n_pixels}, "
            f"chi_square={self.chi_square:.6g}, "
            f"chi_square_per_pixel={self.chi_square_per_pixel:.4g})"
        )


def build_frame_fits(
    result: MCMCFitResult,
    *,
    hdi_probability: float | None = None,
    max_draws: int | None = 1000,
    random_seed: int = 1729,
) -> tuple[FrameFit, ...]:
    """Evaluate the posterior model per frame on native valid pixels.

    Parameters
    ----------
    result
        A sampled MCMC result, single-frame or joint.
    hdi_probability
        When set (in ``(0, 1)``), also evaluate the total model over posterior
        draws and store its pointwise highest-density band on each frame.
        ``None`` (default) computes only the median model.
    max_draws
        Cap on posterior draws used for the band; ``None`` uses every draw.
    random_seed
        Seed for draw subsampling.

    Returns
    -------
    tuple of FrameFit
        One entry per frame in workspace order. A single-frame fit yields one
        entry.
    """
    workspace = result.inputs.workspace
    posterior = result.posterior
    continuum_posterior = posterior["continuum"]
    parameter_names = workspace.continuum.parameter_names
    per_frame = workspace.n_frames > 1 and workspace.continuum.sharing != "shared"

    line_medians = [
        (
            handle,
            posterior[handle.id]["flux"].median,
            posterior[handle.id]["fwhm"].median,
            posterior[handle.id]["center"].median,
            tuple(
                (
                    kernel.kind,
                    posterior[handle.id][kernel.shape_name("fwhm")].median,
                    posterior[handle.id][kernel.shape_name("fraction")].median,
                )
                for kernel in handle.line.kernels
            ),
        )
        for handle in workspace.handles
    ]

    band = hdi_probability is not None
    if band:
        probability = _validate_probability(hdi_probability)
        indices = _draw_indices(result, max_draws=max_draws, random_seed=random_seed)
        line_draws = _line_parameter_draws(result, indices)

    fits: list[FrameFit] = []
    for index, frame_id in enumerate(workspace.frame_ids):
        frame = workspace.spectra[index]
        wavelength = np.asarray(frame.valid_wavelength, dtype=float)
        data = np.asarray(frame.valid_flux, dtype=float)
        error = np.asarray(frame.valid_error, dtype=float)
        design = workspace.continuum.design(wavelength)

        coefficients = np.asarray(
            [
                continuum_posterior[f"{name}[{frame_id}]" if per_frame else name].median
                for name in parameter_names
            ],
            dtype=float,
        )
        continuum = design @ coefficients

        total = continuum.copy()
        components: dict[str, np.ndarray] = {}
        for handle, flux, fwhm, center, kernels in line_medians:
            signed = (
                contribution_sign(handle)
                * flux
                * profile_template(
                    wavelength,
                    handle=handle,
                    center=center,
                    fwhm_kms=fwhm,
                    resolving_power=frame.resolving_power,
                    kernels=kernels,
                )
            )
            total = total + signed
            components[handle.id] = continuum + signed

        model_lower = model_upper = None
        if band:
            coefficient_draws = np.stack(
                [
                    continuum_posterior[
                        f"{name}[{frame_id}]" if per_frame else name
                    ].samples.reshape(-1)[indices]
                    for name in parameter_names
                ]
            )
            total_draws = (design @ coefficient_draws).T + _line_model_draws(
                line_draws, wavelength, frame.resolving_power
            )
            model_lower, model_upper = _pointwise_hdi(total_draws, probability)

        standardized = (data - total) / error
        fits.append(
            FrameFit(
                frame_id=frame_id,
                wavelength=_readonly(wavelength),
                data=_readonly(data),
                error=_readonly(error),
                continuum=_readonly(continuum),
                components=components,
                model=_readonly(total),
                chi_square=float(np.sum(standardized**2)),
                n_pixels=int(wavelength.size),
                model_lower=None if model_lower is None else _readonly(model_lower),
                model_upper=None if model_upper is None else _readonly(model_upper),
            )
        )
    return tuple(fits)


def _draw_indices(
    result: MCMCFitResult, *, max_draws: int | None, random_seed: int
) -> np.ndarray:
    reference = result.posterior[result.posterior.components[0]]
    total = reference[reference.parameters[0]].samples.size
    if max_draws is None or max_draws >= total:
        return np.arange(total)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(total, size=max_draws, replace=False))


def _line_parameter_draws(
    result: MCMCFitResult, indices: np.ndarray
) -> tuple[tuple[Any, np.ndarray, np.ndarray, np.ndarray, tuple], ...]:
    draws = []
    for handle in result.inputs.workspace.handles:
        posterior = result.posterior[handle.id]
        flux = posterior["flux"].samples.reshape(-1)[indices]
        center = posterior["center"].samples.reshape(-1)[indices]
        fwhm = posterior["fwhm"].samples.reshape(-1)[indices]
        kernels = tuple(
            (
                kernel.kind,
                posterior[kernel.shape_name("fwhm")].samples.reshape(-1)[indices],
                posterior[kernel.shape_name("fraction")].samples.reshape(-1)[indices],
            )
            for kernel in handle.line.kernels
        )
        draws.append((handle, flux, center, fwhm, kernels))
    return tuple(draws)


def _line_model_draws(
    line_draws: tuple[tuple[Any, np.ndarray, np.ndarray, np.ndarray, tuple], ...],
    wavelength: np.ndarray,
    resolving_power: float | None,
) -> np.ndarray:
    n_draws = line_draws[0][1].size if line_draws else 0
    total = np.zeros((n_draws, wavelength.size), dtype=float)
    for handle, flux, center, fwhm, kernels in line_draws:
        sign = contribution_sign(handle)
        for draw in range(n_draws):
            draw_kernels = tuple(
                (kind, widths[draw], fractions[draw])
                for kind, widths, fractions in kernels
            )
            total[draw] += (
                sign
                * flux[draw]
                * profile_template(
                    wavelength,
                    handle=handle,
                    center=center[draw],
                    fwhm_kms=fwhm[draw],
                    resolving_power=resolving_power,
                    kernels=draw_kernels,
                )
            )
    return total


def _pointwise_hdi(
    samples: np.ndarray, probability: float
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(samples, axis=0)
    interval = max(int(np.floor(probability * ordered.shape[0])), 1)
    if interval >= ordered.shape[0]:
        return ordered[0], ordered[-1]
    widths = ordered[interval:] - ordered[: ordered.shape[0] - interval]
    starts = np.argmin(widths, axis=0)
    columns = np.arange(ordered.shape[1])
    return ordered[starts, columns], ordered[starts + interval, columns]


def _validate_probability(value: float | None) -> float:
    probability = float(value)  # type: ignore[arg-type]
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("hdi_probability must be finite and between 0 and 1.")
    return probability


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
