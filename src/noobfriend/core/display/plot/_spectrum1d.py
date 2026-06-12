"""The public 1-D spectrum plotter.

:func:`plot_spectrum1d` is a thin wrapper that builds a static matplotlib figure
and delegates the drawing to
:func:`~noobfriend.core.display.plot._spectrum.draw_spectrum` (the shared
engine, also reused by the future 2-D panel). See that module for the
spectrum / model data model and :class:`~noobfriend.core.display.plot._spectrum.ModelSpec`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from numpy.typing import ArrayLike

from noobfriend.core.display.plot._spectrum import ModelSpec, draw_spectrum

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

#: Figure height as a fraction of its width (spectra are wide and short).
_DEFAULT_ASPECT: float = 0.45


def plot_spectrum1d(
    wavelength: ArrayLike | Sequence[ArrayLike],
    flux: ArrayLike | Sequence[ArrayLike],
    *,
    error: ArrayLike | Sequence[ArrayLike | None] | None = None,
    labels: str | Sequence[str] | None = None,
    colors: str | Sequence[str] | None = None,
    models: ModelSpec | Callable | tuple | Sequence | None = None,
    drawstyle: Literal["steps", "line"] = "steps",
    error_style: Literal["band", "line", "none"] = "band",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    x_label: str = "Wavelength",
    y_label: str = "Flux",
    size: int = 680,
    title: str | None = None,
    save: str | Path | None = None,
) -> Figure:
    """Plot one or several 1-D spectra with optional model overlays.

    Data spectra are drawn as histogram-style steps (the wavelength-bin-correct
    rendering) with an optional uncertainty band; model curves are overlaid on
    top as smooth lines, each optionally carrying its own band. The figure is a
    static matplotlib figure, returned so the caller can further customise or
    export it (a notebook also renders it as the cell's last expression).

    Parameters
    ----------
    wavelength : array_like or sequence of array_like
        The wavelength grid, in any consistent numeric unit. A single 1-D array
        is shared by every flux line; a sequence of arrays gives one grid per
        line (length must match ``flux``).
    flux : array_like or sequence of array_like
        One spectrum (1-D) or several (a sequence of 1-D arrays, or a 2-D array
        read row-by-row). ``NaN`` values break the line, showing gaps.
    error : array_like or sequence of array_like or None, optional
        Per-line 1-sigma uncertainty (symmetric band ``flux +/- error``). For a
        single spectrum, one array; for several, a list with one array (or
        ``None``) per line, so some lines may carry no error.
    labels : str or sequence of str, optional
        Legend labels (one per line). A bare string is allowed only for a single
        spectrum.
    colors : str or sequence of str, optional
        Line colors. ``None`` uses matplotlib's color cycle; a single string is
        broadcast; a sequence gives one per line.
    models : ModelSpec or callable or tuple or sequence, optional
        Model curves overlaid on top. A single model is a :class:`ModelSpec`
        dict, a bare ``flux_func`` callable, or a ``(wavelength, flux)`` tuple;
        wrap several in a list. See :class:`ModelSpec`.
    drawstyle : {"steps", "line"}, default "steps"
        Render data spectra as ``steps-mid`` histograms or plain lines. Models
        are always smooth lines.
    error_style : {"band", "line", "none"}, default "band"
        Render data uncertainty as a shaded band, as a separate error spectrum
        line, or not at all.
    xlim, ylim : tuple of float, optional
        Explicit ``(min, max)`` axis limits. Left unset, matplotlib autoscales
        to the data.
    ylog : bool, default False
        Use a logarithmic y-axis.
    x_label, y_label : str, default "Wavelength", "Flux"
        Axis labels. No unit is assumed; add one here if desired.
    size : int, default 680
        Figure width in pixels; the height follows a fixed aspect ratio.
    title : str, optional
        Figure title.
    save : str or pathlib.Path, optional
        If given, write the figure to this path with ``savefig``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.

    Raises
    ------
    ValueError
        On a bad ``drawstyle``/``error_style``, a length mismatch among
        ``wavelength``/``flux``/``error``/``labels``/``colors``, or an invalid
        model specification (see :meth:`_ModelCurve.from_input`).
    """
    import matplotlib.pyplot as plt

    width_in = size / 100.0
    fig, ax = plt.subplots(
        figsize=(width_in, width_in * _DEFAULT_ASPECT), layout="constrained"
    )
    draw_spectrum(
        ax,
        wavelength,
        flux,
        error=error,
        labels=labels,
        colors=colors,
        models=models,
        drawstyle=drawstyle,
        error_style=error_style,
        xlim=xlim,
        ylim=ylim,
        ylog=ylog,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
