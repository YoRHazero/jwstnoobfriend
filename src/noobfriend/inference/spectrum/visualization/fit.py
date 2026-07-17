"""Shared static renderer for MLE and MCMC fit predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.core.display.plot import ModelSpec, plot_spectrum1d, plot_spectrum2d
from noobfriend.inference.spectrum.visualization.prediction import (
    ModelView,
    FitPrediction,
    build_mcmc_prediction,
    build_mle_prediction,
)
from noobfriend.inference.spectrum.visualization.style import (
    CONTINUUM_COLOR,
    DATA_COLOR,
    TOTAL_COLOR,
    resolve_component_colors,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.data.types import DispersionAxis
    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult
    from noobfriend.inference.spectrum.workspace.mle import MLEFitResult


@dataclass(frozen=True, slots=True)
class _Spectrum2DPlotInput:
    flux: np.ndarray
    spatial: np.ndarray
    spatial_window: tuple[float, float] | None


def plot_mle_fit(
    result: MLEFitResult,
    *,
    solution: Literal["selected", "likelihood_optimum"] = "selected",
    show_residuals: bool = True,
    model_oversample: int = 8,
    flux_2d: ArrayLike | None = None,
    spatial: ArrayLike | None = None,
    dispersion: DispersionAxis | None = None,
    spatial_window: tuple[float, float] | None = None,
    component_colors: Mapping[str, str] | None = None,
    data_color: str = DATA_COLOR,
    continuum_color: str = CONTINUUM_COLOR,
    total_color: str = TOTAL_COLOR,
    cmap: str = "magma",
    pmin: float = 1.0,
    pmax: float = 99.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    size: int = 1000,
    title: str | None = None,
    legend_location: str = "best",
) -> Figure:
    """Plot one MLE solution with the shared spectrum-fit visual language."""
    prediction = build_mle_prediction(
        result,
        solution=solution,
        model_oversample=model_oversample,
    )
    return _plot_fit_prediction(
        prediction,
        spectrum=result.workspace.spectrum,
        total_label="total model",
        show_residuals=show_residuals,
        flux_2d=flux_2d,
        spatial=spatial,
        dispersion=dispersion,
        spatial_window=spatial_window,
        component_colors=component_colors,
        data_color=data_color,
        continuum_color=continuum_color,
        total_color=total_color,
        cmap=cmap,
        pmin=pmin,
        pmax=pmax,
        xlim=xlim,
        ylim=ylim,
        ylog=ylog,
        size=size,
        title=title,
        legend_location=legend_location,
    )


def plot_mcmc_fit(
    result: MCMCFitResult,
    *,
    hdi_probability: float = 0.94,
    posterior_draws: int | None = 1000,
    random_seed: int = 1729,
    show_residuals: bool = True,
    model_oversample: int = 8,
    flux_2d: ArrayLike | None = None,
    spatial: ArrayLike | None = None,
    dispersion: DispersionAxis | None = None,
    spatial_window: tuple[float, float] | None = None,
    component_colors: Mapping[str, str] | None = None,
    data_color: str = DATA_COLOR,
    continuum_color: str = CONTINUUM_COLOR,
    total_color: str = TOTAL_COLOR,
    cmap: str = "magma",
    pmin: float = 1.0,
    pmax: float = 99.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    size: int = 1000,
    title: str | None = None,
    legend_location: str = "best",
    view: ModelView = "observed",
) -> Figure:
    """Plot posterior model summaries with a pointwise total-model HDI.

    ``view`` controls the per-component curves only (``"observed"``,
    ``"emergent"``, or ``"intrinsic"``); the total model, its HDI band, and
    the residual panel are always evaluated in the observed view.
    """
    prediction = build_mcmc_prediction(
        result,
        hdi_probability=hdi_probability,
        posterior_draws=posterior_draws,
        random_seed=random_seed,
        model_oversample=model_oversample,
        view=view,
    )
    if view != "observed":
        prediction = FitPrediction(
            wavelength=prediction.wavelength,
            continuum=prediction.continuum,
            components={
                f"{name} ({view})": curve
                for name, curve in prediction.components.items()
            },
            total=prediction.total,
        )
    return _plot_fit_prediction(
        prediction,
        spectrum=result.inputs.data,
        total_label=f"total + {hdi_probability:.0%} HDI",
        show_residuals=show_residuals,
        flux_2d=flux_2d,
        spatial=spatial,
        dispersion=dispersion,
        spatial_window=spatial_window,
        component_colors=component_colors,
        data_color=data_color,
        continuum_color=continuum_color,
        total_color=total_color,
        cmap=cmap,
        pmin=pmin,
        pmax=pmax,
        xlim=xlim,
        ylim=ylim,
        ylog=ylog,
        size=size,
        title=title,
        legend_location=legend_location,
    )


def _plot_fit_prediction(
    prediction: FitPrediction,
    *,
    spectrum: NoobSpectrum,
    total_label: str,
    show_residuals: bool,
    flux_2d: ArrayLike | None,
    spatial: ArrayLike | None,
    dispersion: DispersionAxis | None,
    spatial_window: tuple[float, float] | None,
    component_colors: Mapping[str, str] | None,
    data_color: str,
    continuum_color: str,
    total_color: str,
    cmap: str,
    pmin: float,
    pmax: float,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    ylog: bool,
    size: int,
    title: str | None,
    legend_location: str,
) -> Figure:
    if not isinstance(show_residuals, bool):
        raise TypeError("show_residuals must be bool.")
    component_color_values = resolve_component_colors(
        tuple(prediction.components), component_colors
    )
    models: list[ModelSpec] = [
        {
            "wavelength": prediction.wavelength,
            "flux": prediction.continuum.value,
            "label": "continuum",
            "color": continuum_color,
            "linestyle": "--",
        }
    ]
    models.extend(
        {
            "wavelength": prediction.wavelength,
            "flux": curve.value,
            "label": component,
            "color": component_color_values[component],
        }
        for component, curve in prediction.components.items()
    )
    total_model: ModelSpec = {
        "wavelength": prediction.wavelength,
        "flux": prediction.total.value,
        "label": total_label,
        "color": total_color,
    }
    if prediction.total.lower is not None and prediction.total.upper is not None:
        total_model["error"] = (prediction.total.lower, prediction.total.upper)
    models.append(total_model)

    image = _resolve_2d_input(
        spectrum,
        flux_2d=flux_2d,
        spatial=spatial,
        dispersion=dispersion,
        spatial_window=spatial_window,
    )
    if show_residuals:
        residual, residual_error = _raw_residual(spectrum, prediction)
    else:
        residual = None
        residual_error = None
    common = {
        "error": spectrum.error,
        "residual": residual,
        "residual_error": residual_error,
        "residual_color": data_color,
        "models": models,
        "drawstyle": "steps",
        "error_style": "band",
        "xlim": xlim,
        "ylim": ylim,
        "ylog": ylog,
        "x_label": f"Observed wavelength [{spectrum.unit}]",
        "y_label": "Flux",
        "size": size,
        "title": title,
    }
    if image is None:
        figure = plot_spectrum1d(
            spectrum.obs,
            spectrum.flux,
            labels="data",
            colors=data_color,
            **common,
        )
    else:
        figure = plot_spectrum2d(
            spectrum.obs,
            image.flux,
            flux_1d=spectrum.flux,
            spatial=image.spatial,
            spatial_window=image.spatial_window,
            label="data",
            color=data_color,
            cmap=cmap,
            pmin=pmin,
            pmax=pmax,
            spatial_label="Cross-dispersion [pixel]",
            **common,
        )
    _finish_fit_figure(
        figure,
        total_label=total_label,
        legend_location=legend_location,
    )
    return figure


def _resolve_2d_input(
    spectrum: NoobSpectrum,
    *,
    flux_2d: ArrayLike | None,
    spatial: ArrayLike | None,
    dispersion: DispersionAxis | None,
    spatial_window: tuple[float, float] | None,
) -> _Spectrum2DPlotInput | None:
    if flux_2d is None:
        source = spectrum.source_2d
        if source is None:
            if (
                spatial is not None
                or dispersion is not None
                or spatial_window is not None
            ):
                raise ValueError(
                    "spatial, dispersion, and spatial_window require flux_2d or "
                    "a spectrum created with NoobSpectrum.from_2d()."
                )
            return None
        if dispersion is not None:
            raise ValueError("dispersion only applies to an explicit flux_2d array.")
        image = source.flux
        if spatial is None:
            coordinates = source.spatial
            default_window = source.spatial_window
        else:
            coordinates = np.asarray(spatial, dtype=float)
            default_window = _window_from_indices(coordinates, source.collapse_window)
        window = default_window if spatial_window is None else spatial_window
        return _validated_2d_plot_input(image, coordinates, window)

    raw = np.asarray(flux_2d, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"flux_2d must be 2-D, got shape {raw.shape}.")
    n_wave = spectrum.obs.size
    if dispersion is None:
        matching_axes = tuple(
            index for index, length in enumerate(raw.shape) if length == n_wave
        )
        if len(matching_axes) != 1:
            raise ValueError(
                "cannot infer the flux_2d wavelength axis; exactly one axis must "
                "match the 1-D spectrum length, or give dispersion explicitly."
            )
        wavelength_axis = matching_axes[0]
    elif dispersion == "row":
        wavelength_axis = 1
    elif dispersion == "column":
        wavelength_axis = 0
    else:
        raise ValueError(
            f"dispersion must be 'row', 'column', or None, got {dispersion!r}."
        )
    if raw.shape[wavelength_axis] != n_wave:
        raise ValueError(
            f"flux_2d wavelength extent {raw.shape[wavelength_axis]} must match "
            f"the 1-D spectrum length {n_wave}."
        )
    image = raw if wavelength_axis == 1 else raw.T
    coordinates = (
        np.arange(image.shape[0], dtype=float)
        if spatial is None
        else np.asarray(spatial, dtype=float)
    )
    return _validated_2d_plot_input(image, coordinates, spatial_window)


def _validated_2d_plot_input(
    flux: np.ndarray,
    spatial: np.ndarray,
    spatial_window: tuple[float, float] | None,
) -> _Spectrum2DPlotInput:
    if spatial.shape != (flux.shape[0],):
        raise ValueError(
            f"spatial length {spatial.shape} must match flux_2d spatial extent "
            f"{flux.shape[0]}."
        )
    return _Spectrum2DPlotInput(
        flux=np.asarray(flux, dtype=float),
        spatial=np.asarray(spatial, dtype=float),
        spatial_window=spatial_window,
    )


def _window_from_indices(
    spatial: np.ndarray,
    collapse_window: tuple[int, int],
) -> tuple[float, float]:
    if spatial.ndim != 1 or spatial.size == 0:
        raise ValueError("spatial must be a non-empty 1-D array.")
    if spatial.size == 1:
        edges = np.array([spatial[0] - 0.5, spatial[0] + 0.5])
    else:
        midpoints = 0.5 * (spatial[:-1] + spatial[1:])
        edges = np.concatenate(
            (
                [spatial[0] - 0.5 * (spatial[1] - spatial[0])],
                midpoints,
                [spatial[-1] + 0.5 * (spatial[-1] - spatial[-2])],
            )
        )
    lo, hi = collapse_window
    return float(edges[lo]), float(edges[hi])


def _raw_residual(
    spectrum: NoobSpectrum,
    prediction: FitPrediction,
) -> tuple[np.ndarray, np.ndarray]:
    model = np.interp(
        spectrum.obs,
        prediction.wavelength,
        prediction.total.value,
    )
    residual = np.full_like(spectrum.flux, np.nan, dtype=float)
    valid = spectrum.valid_mask
    residual[valid] = spectrum.flux[valid] - model[valid]
    residual_error = np.full_like(spectrum.error, np.nan, dtype=float)
    residual_error[valid] = spectrum.error[valid]
    return residual, residual_error


def _finish_fit_figure(
    figure: Figure,
    *,
    total_label: str,
    legend_location: str,
) -> None:
    spectrum_axis = next(
        axis
        for axis in figure.axes
        if any(line.get_label() == "data" for line in axis.get_lines())
    )
    for line in spectrum_axis.get_lines():
        if line.get_label() == "data":
            line.set_linewidth(1.5)
        elif line.get_label() == total_label:
            line.set_linewidth(2.2)
        elif line.get_label() == "continuum":
            line.set_linewidth(1.4)
        elif not line.get_label().startswith("_"):
            line.set_linewidth(1.5)

    handles, labels = spectrum_axis.get_legend_handles_labels()
    spectrum_axis.legend(
        handles,
        labels,
        loc=legend_location,
        ncol=1 if len(handles) <= 5 else 2,
        frameon=True,
        framealpha=0.85,
        fontsize="small",
        title="Components shown on continuum",
        title_fontsize="small",
    )
