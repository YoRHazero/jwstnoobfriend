"""Per-frame small-multiples renderer for joint multi-frame MCMC results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.visualization.prediction import (
    FitPrediction,
    ModelView,
    build_mcmc_frame_predictions,
)
from noobfriend.inference.spectrum.visualization.style import (
    CONTINUUM_COLOR,
    DATA_COLOR,
    TOTAL_COLOR,
    resolve_component_colors,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult


def plot_mcmc_frames(
    result: MCMCFitResult,
    *,
    hdi_probability: float | None = 0.94,
    posterior_draws: int | None = 1000,
    random_seed: int = 1729,
    show_components: bool = True,
    show_chi_square: bool = True,
    show_residuals: bool = False,
    model_oversample: int = 8,
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
    """Plot the posterior model over each frame as stacked panels.

    Every frame is drawn on its own native (oversampled) wavelength grid with
    the shared line model and that frame's continuum. Frames created with
    :meth:`NoobSpectrum.from_2d` get their 2-D source image as a strip above
    the spectrum panel. The chi-square annotation and the residual panels use
    the drawn (posterior-median) total model on each frame's valid pixels.

    Parameters
    ----------
    result
        A sampled MCMC result (single- or multi-frame).
    hdi_probability
        Shade each frame's total model with this pointwise highest-density band
        (the model's parameter uncertainty). ``None`` draws only the median.
    posterior_draws
        Cap on posterior draws used for the curves; ``None`` uses every draw.
    random_seed
        Seed for the draw subsampling.
    show_components
        Draw each line's contribution on the continuum when more than one line
        is present.
    show_chi_square
        Annotate each panel with its chi-square per pixel.
    show_residuals
        Add a residual panel (data minus total model) below each frame.
    model_oversample
        Subdivisions per native wavelength interval for the model curves.
    component_colors
        Optional per-line color overrides keyed by line id.
    data_color, continuum_color, total_color
        Colors for the data steps, continuum, and total model.
    cmap, pmin, pmax
        Colormap and percentile scaling of each frame's 2-D source strip.
    xlim, ylim, ylog, size, title
        Standard axis and figure controls; ``size`` is the figure width in
        pixels at 100 dpi. ``ylim``/``ylog`` apply to the spectrum panels.
    legend_location
        Matplotlib legend location on the first spectrum panel.
    view
        Convolution layers of the per-line component curves (``"observed"`` =
        kernels + LSF, ``"emergent"`` = kernels only, ``"intrinsic"`` = bare
        profile); the total model, chi-square, and residuals stay observed.
    """
    import matplotlib.pyplot as plt

    predictions = build_mcmc_frame_predictions(
        result,
        hdi_probability=hdi_probability,
        posterior_draws=posterior_draws,
        random_seed=random_seed,
        model_oversample=model_oversample,
        view=view,
    )
    workspace = result.inputs.workspace
    frames = workspace.spectra
    line_ids = tuple(workspace.ids)
    colors = resolve_component_colors(line_ids, component_colors)

    rows: list[tuple[str, int]] = []
    ratios: list[float] = []
    for index, frame in enumerate(frames):
        if frame.source_2d is not None:
            rows.append(("image", index))
            ratios.append(1.0)
        rows.append(("spectrum", index))
        ratios.append(2.4)
        if show_residuals:
            rows.append(("residual", index))
            ratios.append(0.9)

    width = max(size / 100.0, 3.0)
    height = max(0.9 * sum(ratios), 2.4)
    figure, axes = plt.subplots(
        len(rows),
        1,
        sharex=True,
        figsize=(width, height),
        squeeze=False,
        height_ratios=ratios,
    )
    column = axes[:, 0]

    for axis, (kind, index) in zip(column, rows, strict=True):
        frame = frames[index]
        prediction = predictions[index]
        if kind == "image":
            _draw_frame_image(axis, frame, cmap=cmap, pmin=pmin, pmax=pmax)
            continue
        if kind == "residual":
            _draw_frame_residual(
                axis,
                frame,
                prediction,
                data_color=data_color,
                continuum_color=continuum_color,
            )
            continue
        axis.step(
            frame.obs,
            frame.flux,
            where="mid",
            color=data_color,
            linewidth=1.2,
            label="data",
        )
        axis.plot(
            prediction.wavelength,
            prediction.continuum.value,
            color=continuum_color,
            linestyle="--",
            linewidth=1.2,
            label="continuum",
        )
        if show_components and len(line_ids) > 1:
            for line_id in line_ids:
                label = line_id if view == "observed" else f"{line_id} ({view})"
                axis.plot(
                    prediction.wavelength,
                    prediction.components[line_id].value,
                    color=colors[line_id],
                    linewidth=1.2,
                    label=label,
                )
        if prediction.total.lower is not None and prediction.total.upper is not None:
            axis.fill_between(
                prediction.wavelength,
                prediction.total.lower,
                prediction.total.upper,
                color=total_color,
                alpha=0.18,
                linewidth=0.0,
                zorder=1,
                label=f"{hdi_probability:.0%} HDI",
            )
        axis.plot(
            prediction.wavelength,
            prediction.total.value,
            color=total_color,
            linewidth=1.9,
            label="total model",
        )
        panel_label = workspace.frame_ids[index]
        if show_chi_square:
            chi_square = _chi_square_per_pixel(frame, prediction)
            panel_label += f"   χ²/pix = {chi_square:.2f}"
        axis.text(
            0.012,
            0.94,
            panel_label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.75,
                "pad": 0.25,
            },
        )
        axis.set_ylabel("Flux")
        if ylog:
            axis.set_yscale("log")
        if ylim is not None:
            axis.set_ylim(ylim)

    if xlim is not None:
        column[-1].set_xlim(xlim)
    column[-1].set_xlabel(f"Observed wavelength [{workspace.unit}]")
    legend_axis = column[
        next(i for i, (kind, _) in enumerate(rows) if kind == "spectrum")
    ]
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc=legend_location,
        ncol=1 if len(handles) <= 4 else 2,
        frameon=True,
        framealpha=0.85,
        fontsize="small",
    )
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def _median_model_on_frame(
    frame: NoobSpectrum, prediction: FitPrediction
) -> np.ndarray:
    """Interpolate the drawn total model back onto the frame's native pixels."""
    return np.interp(frame.obs, prediction.wavelength, prediction.total.value)


def _chi_square_per_pixel(frame: NoobSpectrum, prediction: FitPrediction) -> float:
    model = _median_model_on_frame(frame, prediction)
    valid = frame.valid_mask
    count = int(valid.sum())
    if not count:
        return float("nan")
    standardized = (frame.flux[valid] - model[valid]) / frame.error[valid]
    return float(np.sum(standardized**2)) / count


def _draw_frame_residual(
    axis: Axes,
    frame: NoobSpectrum,
    prediction: FitPrediction,
    *,
    data_color: str,
    continuum_color: str,
) -> None:
    model = _median_model_on_frame(frame, prediction)
    residual = np.full_like(frame.flux, np.nan, dtype=float)
    error = np.full_like(frame.error, np.nan, dtype=float)
    valid = frame.valid_mask
    residual[valid] = frame.flux[valid] - model[valid]
    error[valid] = frame.error[valid]
    axis.axhline(0.0, color=continuum_color, linewidth=1.0, linestyle=":")
    axis.step(frame.obs, residual, where="mid", color=data_color, linewidth=1.0)
    axis.fill_between(
        frame.obs,
        residual - error,
        residual + error,
        color=data_color,
        alpha=0.2,
        linewidth=0.0,
        step="mid",
    )
    axis.set_ylabel("Residual")


def _draw_frame_image(
    axis: Axes,
    frame: NoobSpectrum,
    *,
    cmap: str,
    pmin: float,
    pmax: float,
) -> None:
    from noobfriend.core.display.plot._spectrum2d import _centers_to_edges, _draw_image

    source = frame.source_2d
    if source is None:  # pragma: no cover - guarded by the row layout
        raise RuntimeError("frame image row requires a 2-D source.")
    vmin, vmax = np.nanpercentile(source.flux, [pmin, pmax])
    _draw_image(
        axis,
        source.flux,
        _centers_to_edges(np.asarray(frame.obs, dtype=float)),
        _centers_to_edges(source.spatial),
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        window=source.spatial_window,
    )
    axis.set_ylabel("Cross-disp.")
