"""Per-frame small-multiples renderer for joint multi-frame MCMC results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.visualization.style import (
    CONTINUUM_COLOR,
    DATA_COLOR,
    TOTAL_COLOR,
    resolve_component_colors,
)
from noobfriend.inference.spectrum.workspace.mcmc.frames import build_frame_fits

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult


def plot_mcmc_frames(
    result: MCMCFitResult,
    *,
    show_components: bool = True,
    show_chi_square: bool = True,
    component_colors: Mapping[str, str] | None = None,
    data_color: str = DATA_COLOR,
    continuum_color: str = CONTINUUM_COLOR,
    total_color: str = TOTAL_COLOR,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    size: int = 1000,
    title: str | None = None,
) -> Figure:
    """Plot the posterior-median model over each frame as stacked panels.

    Every frame is drawn on its own native wavelength grid with the shared line
    model and that frame's continuum; the per-frame chi-square per pixel is
    annotated so inter-visit systematics are visible at a glance.

    Parameters
    ----------
    result
        A sampled MCMC result (single- or multi-frame).
    show_components
        Draw each line's contribution on the continuum when more than one line
        is present.
    show_chi_square
        Annotate each panel with its chi-square per pixel.
    component_colors
        Optional per-line color overrides keyed by line id.
    data_color, continuum_color, total_color
        Colors for the data steps, continuum, and total model.
    xlim, ylim, ylog, size, title
        Standard axis and figure controls; ``size`` is the figure width in
        pixels at 100 dpi.
    """
    import matplotlib.pyplot as plt

    fits = build_frame_fits(result)
    workspace = result.inputs.workspace
    line_ids = tuple(workspace.ids)
    colors = resolve_component_colors(line_ids, component_colors)

    width = max(size / 100.0, 3.0)
    height = max(2.1 * len(fits), 2.4)
    figure, axes = plt.subplots(
        len(fits), 1, sharex=True, figsize=(width, height), squeeze=False
    )
    column = axes[:, 0]
    for axis, fit in zip(column, fits, strict=True):
        axis.step(
            fit.wavelength,
            fit.data,
            where="mid",
            color=data_color,
            linewidth=1.2,
            label="data",
        )
        axis.plot(
            fit.wavelength,
            fit.continuum,
            color=continuum_color,
            linestyle="--",
            linewidth=1.2,
            label="continuum",
        )
        if show_components and len(line_ids) > 1:
            for line_id in line_ids:
                axis.plot(
                    fit.wavelength,
                    fit.components[line_id],
                    color=colors[line_id],
                    linewidth=1.2,
                    label=line_id,
                )
        axis.plot(
            fit.wavelength,
            fit.model,
            color=total_color,
            linewidth=1.9,
            label="total model",
        )
        panel_label = fit.frame_id
        if show_chi_square:
            panel_label += f"   χ²/pix = {fit.chi_square_per_pixel:.2f}"
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
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)

    column[-1].set_xlabel(f"Observed wavelength [{workspace.unit}]")
    handles, labels = column[0].get_legend_handles_labels()
    column[0].legend(
        handles,
        labels,
        loc="upper right",
        ncol=1 if len(handles) <= 4 else 2,
        frameon=True,
        framealpha=0.85,
        fontsize="small",
    )
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure
