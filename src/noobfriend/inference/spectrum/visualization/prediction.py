"""Backend adapters that evaluate fit results on a dense plotting grid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    contribution_sign,
    profile_template,
    profile_template_stack,
)

type ModelView = Literal["observed", "emergent", "intrinsic"]

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult
    from noobfriend.inference.spectrum.workspace.mle import MLEFitResult, MLESolution


@dataclass(frozen=True, slots=True)
class CurvePrediction:
    """One evaluated curve with an optional pointwise interval."""

    value: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FitPrediction:
    """Continuum-based components and total model on one wavelength grid."""

    wavelength: np.ndarray
    continuum: CurvePrediction
    components: Mapping[str, CurvePrediction]
    total: CurvePrediction

    def __post_init__(self) -> None:
        """Freeze component lookup while preserving workspace order."""
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


def build_mle_prediction(
    result: MLEFitResult,
    *,
    solution: Literal["selected", "likelihood_optimum"],
    model_oversample: int,
) -> FitPrediction:
    """Evaluate one MLE solution exactly on a dense plotting grid."""
    selected = _select_mle_solution(result, solution)
    workspace = result.workspace
    wavelength = dense_wavelength(workspace.spectrum.obs, model_oversample)
    coefficients = np.asarray(
        [
            selected.continuum_parameters[name]
            for name in workspace.continuum.parameter_names
        ],
        dtype=float,
    )
    continuum = workspace.continuum.design(wavelength) @ coefficients
    total = continuum.copy()
    components: dict[str, CurvePrediction] = {}
    for fitted in selected.lines:
        handle = fitted.handle
        fitted_kernels = tuple(
            (
                kernel.kind,
                fitted.shapes[kernel.shape_name("fwhm")],
                fitted.shapes[kernel.shape_name("fraction")],
            )
            for kernel in handle.line.kernels
        )
        signed = (
            contribution_sign(handle)
            * fitted.flux
            * profile_template(
                wavelength,
                handle=handle,
                center=fitted.center,
                fwhm_kms=fitted.fwhm,
                resolving_power=workspace.spectrum.resolving_power,
                kernels=fitted_kernels,
            )
        )
        total += signed
        components[handle.id] = CurvePrediction(value=continuum + signed)
    return FitPrediction(
        wavelength=wavelength,
        continuum=CurvePrediction(value=continuum),
        components=components,
        total=CurvePrediction(value=total),
    )


def build_mcmc_prediction(
    result: MCMCFitResult,
    *,
    hdi_probability: float,
    posterior_draws: int | None,
    random_seed: int,
    model_oversample: int,
    view: ModelView = "observed",
) -> FitPrediction:
    """Evaluate physical posterior draws and summarize them pointwise.

    ``view`` selects the convolution layers drawn for the per-line component
    curves: ``"observed"`` applies convolution kernels and the instrumental
    LSF, ``"emergent"`` applies kernels only (the source's emergent
    spectrum), and ``"intrinsic"`` shows the bare base profile. The total
    model (and therefore the residual reference) is always evaluated in the
    observed view, because only that is comparable with the data.
    """
    if view not in ("observed", "emergent", "intrinsic"):
        raise ValueError(
            f"view must be 'observed', 'emergent', or 'intrinsic'; got {view!r}."
        )
    probability = _validate_probability(hdi_probability)
    workspace = result.inputs.workspace
    wavelength = dense_wavelength(workspace.spectrum.obs, model_oversample)
    indices = _posterior_indices(
        result.posterior[result.posterior.components[0]][
            result.posterior[result.posterior.components[0]].parameters[0]
        ].samples.size,
        posterior_draws=posterior_draws,
        random_seed=random_seed,
    )
    n_draws = indices.size
    continuum_draws = np.zeros((n_draws, wavelength.size), dtype=float)
    continuum_posterior = result.posterior["continuum"]
    design = workspace.continuum.design(wavelength)
    for column, parameter_name in enumerate(workspace.continuum.parameter_names):
        samples = continuum_posterior[parameter_name].samples.reshape(-1)[indices]
        continuum_draws += samples[:, None] * design[None, :, column]

    total_draws = continuum_draws.copy()
    components: dict[str, CurvePrediction] = {}
    for handle in workspace.handles:
        posterior = result.posterior[handle.id]
        flux = posterior["flux"].samples.reshape(-1)[indices]
        fwhm = posterior["fwhm"].samples.reshape(-1)[indices]
        center = posterior["center"].samples.reshape(-1)[indices]
        kernel_draws = tuple(
            (
                kernel.kind,
                posterior[kernel.shape_name("fwhm")].samples.reshape(-1)[indices],
                posterior[kernel.shape_name("fraction")].samples.reshape(-1)[indices],
            )
            for kernel in handle.line.kernels
        )
        line_draws = np.empty_like(continuum_draws)
        view_draws = line_draws if view == "observed" else np.empty_like(line_draws)
        sign = contribution_sign(handle)
        for draw_index in range(n_draws):
            draw_kernels = tuple(
                (kind, widths[draw_index], fractions[draw_index])
                for kind, widths, fractions in kernel_draws
            )
            line_draws[draw_index] = (
                sign
                * flux[draw_index]
                * profile_template(
                    wavelength,
                    handle=handle,
                    center=center[draw_index],
                    fwhm_kms=fwhm[draw_index],
                    resolving_power=workspace.spectrum.resolving_power,
                    kernels=draw_kernels,
                )
            )
            if view != "observed":
                view_draws[draw_index] = (
                    sign
                    * flux[draw_index]
                    * profile_template(
                        wavelength,
                        handle=handle,
                        center=center[draw_index],
                        fwhm_kms=fwhm[draw_index],
                        resolving_power=None,
                        kernels=draw_kernels if view == "emergent" else (),
                    )
                )
        total_draws += line_draws
        components[handle.id] = CurvePrediction(
            value=np.median(continuum_draws + view_draws, axis=0)
        )

    lower, upper = _pointwise_hdi(total_draws, probability=probability)
    return FitPrediction(
        wavelength=wavelength,
        continuum=CurvePrediction(value=np.median(continuum_draws, axis=0)),
        components=components,
        total=CurvePrediction(
            value=np.median(total_draws, axis=0),
            lower=lower,
            upper=upper,
        ),
    )


def build_mcmc_frame_predictions(
    result: MCMCFitResult,
    *,
    hdi_probability: float | None,
    posterior_draws: int | None,
    random_seed: int,
    model_oversample: int,
    view: ModelView = "observed",
) -> tuple[FitPrediction, ...]:
    """Evaluate posterior draws per frame on dense native wavelength grids.

    Mirrors :func:`build_mcmc_prediction` for every frame of a fit: the shared
    line draws are evaluated on each frame's oversampled grid together with
    that frame's continuum coefficients. ``hdi_probability=None`` skips the
    pointwise total-model band. A single-frame result yields one prediction.
    """
    from noobfriend.inference.spectrum.workspace.mcmc.frames import (
        _line_parameter_draws,
    )

    if view not in ("observed", "emergent", "intrinsic"):
        raise ValueError(
            f"view must be 'observed', 'emergent', or 'intrinsic'; got {view!r}."
        )
    probability = (
        None if hdi_probability is None else _validate_probability(hdi_probability)
    )
    workspace = result.inputs.workspace
    continuum_posterior = result.posterior["continuum"]
    parameter_names = workspace.continuum.parameter_names
    per_frame = workspace.n_frames > 1 and workspace.continuum.sharing != "shared"
    reference = result.posterior[result.posterior.components[0]]
    indices = _posterior_indices(
        reference[reference.parameters[0]].samples.size,
        posterior_draws=posterior_draws,
        random_seed=random_seed,
    )
    line_draws = _line_parameter_draws(result, indices)

    predictions: list[FitPrediction] = []
    for index, frame_id in enumerate(workspace.frame_ids):
        frame = workspace.spectra[index]
        wavelength = dense_wavelength(frame.obs, model_oversample)
        design = workspace.continuum.design(wavelength)
        coefficients = np.stack(
            [
                continuum_posterior[
                    f"{name}[{frame_id}]" if per_frame else name
                ].samples.reshape(-1)[indices]
                for name in parameter_names
            ]
        )
        continuum_draws = (design @ coefficients).T
        total_draws = continuum_draws.copy()
        components: dict[str, CurvePrediction] = {}
        for handle, flux, center, fwhm, kernels in line_draws:
            signed_flux = (contribution_sign(handle) * flux)[:, None]
            observed = signed_flux * profile_template_stack(
                wavelength,
                handle=handle,
                centers=center,
                fwhms_kms=fwhm,
                resolving_power=frame.resolving_power,
                kernels=kernels,
            )
            total_draws += observed
            if view == "observed":
                curve = observed
            else:
                curve = signed_flux * profile_template_stack(
                    wavelength,
                    handle=handle,
                    centers=center,
                    fwhms_kms=fwhm,
                    resolving_power=None,
                    kernels=kernels if view == "emergent" else (),
                )
            components[handle.id] = CurvePrediction(
                value=np.median(continuum_draws + curve, axis=0)
            )
        lower = upper = None
        if probability is not None:
            lower, upper = _pointwise_hdi(total_draws, probability=probability)
        predictions.append(
            FitPrediction(
                wavelength=wavelength,
                continuum=CurvePrediction(value=np.median(continuum_draws, axis=0)),
                components=components,
                total=CurvePrediction(
                    value=np.median(total_draws, axis=0),
                    lower=lower,
                    upper=upper,
                ),
            )
        )
    return tuple(predictions)


def dense_wavelength(wavelength: np.ndarray, oversample: int) -> np.ndarray:
    """Subdivide every input wavelength interval without assuming uniformity."""
    if isinstance(oversample, bool) or not isinstance(oversample, int):
        raise TypeError("model_oversample must be a positive integer.")
    if oversample <= 0:
        raise ValueError("model_oversample must be a positive integer.")
    source = np.asarray(wavelength, dtype=float)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("plot wavelength must be a non-empty 1-D array.")
    if source.size == 1:
        return _readonly(source)
    pieces = [
        np.linspace(left, right, oversample, endpoint=False)
        for left, right in zip(source[:-1], source[1:], strict=True)
    ]
    return _readonly(np.concatenate([*pieces, source[-1:]]))


def _select_mle_solution(
    result: MLEFitResult,
    solution: str,
) -> MLESolution:
    if solution == "selected":
        return result.solution
    if solution == "likelihood_optimum":
        return result.likelihood_optimum
    raise ValueError(
        f"solution must be 'selected' or 'likelihood_optimum', got {solution!r}."
    )


def _posterior_indices(
    total_draws: int,
    *,
    posterior_draws: int | None,
    random_seed: int,
) -> np.ndarray:
    if posterior_draws is not None:
        if isinstance(posterior_draws, bool) or not isinstance(posterior_draws, int):
            raise TypeError("posterior_draws must be a positive integer or None.")
        if posterior_draws <= 0:
            raise ValueError("posterior_draws must be a positive integer or None.")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a nonnegative integer.")
    if posterior_draws is None or posterior_draws >= total_draws:
        return np.arange(total_draws)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(total_draws, size=posterior_draws, replace=False))


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("hdi_probability must be finite and between 0 and 1.")
    return probability


def _pointwise_hdi(
    samples: np.ndarray,
    *,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(samples, axis=0)
    interval_count = max(int(np.floor(probability * ordered.shape[0])), 1)
    if interval_count >= ordered.shape[0]:
        return _readonly(ordered[0]), _readonly(ordered[-1])
    widths = ordered[interval_count:] - ordered[:-interval_count]
    starts = np.argmin(widths, axis=0)
    columns = np.arange(ordered.shape[1])
    lower = ordered[starts, columns]
    upper = ordered[starts + interval_count, columns]
    return _readonly(lower), _readonly(upper)


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
