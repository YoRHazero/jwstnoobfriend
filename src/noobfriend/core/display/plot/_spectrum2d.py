"""The public 2-D-over-1-D spectrum panel.

:func:`plot_spectrum2d` stacks a 2-D spectrum image (cross-dispersion vs
wavelength) over its 1-D collapse on a shared wavelength axis -- the classic
spectrum-viewer layout. The two panels are linked by matplotlib ``sharex``, so
the wavelength axis aligns and pans/zooms together; no Bokeh ``x_range``
machinery and no wavelength-data-coordinate gymnastics are needed.

The 1-D panel reuses :func:`~noobfriend.core.display.plot.draw_spectrum`
(the shared engine, identical to :func:`~noobfriend.core.display.plot._spectrum1d.plot_spectrum1d`),
so model overlays, the uncertainty band, and the step rendering all behave the
same. This module only adds the top image panel (:func:`_draw_image`). The image
color stretch is where the ``vmin``/``vmax``/``pmin``/``pmax`` vocabulary lives
(resolved by :func:`~noobfriend.core.display.plot._norm.resolve_limits`, exactly
like :func:`~noobfriend.core.display.plot._image.imshow`); ``cmap`` is a
matplotlib colormap name.

Like the rest of the spectrum area this is unit-agnostic: ``wavelength``,
``flux_2d``, ``spatial`` are bare numeric arrays and units are a labelling
concern only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.core.display.plot._norm import resolve_limits
from noobfriend.core.display.plot._spectrum import ModelSpec, draw_spectrum

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Total figure height as a fraction of its width (two stacked panels).
_DEFAULT_ASPECT2D: float = 0.62

#: Colour of the collapse-window marker drawn on the 2-D panel.
_WINDOW_COLOR: str = "#ff4d4d"


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Return ``n + 1`` cell edges bracketing ``n`` bin centers.

    Interior edges are midpoints between adjacent centers; the two outer edges
    are extrapolated by half the nearest spacing. Handles a non-uniform grid and
    the degenerate single-center case.

    Parameters
    ----------
    centers : numpy.ndarray
        1-D array of bin centers (length ``n >= 1``).

    Returns
    -------
    numpy.ndarray
        1-D array of edges, length ``n + 1``.
    """
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (c[1] - c[0]) / 2.0
    last = c[-1] + (c[-1] - c[-2]) / 2.0
    return np.concatenate([[first], mid, [last]])


def _collapse_2d(flux_2d: np.ndarray) -> np.ndarray:
    """Collapse a ``(spatial, wavelength)`` array along the spatial axis.

    A NaN-aware sum: empty cells are ignored, and a wavelength column that is
    entirely NaN stays NaN (a gap), mirroring how the repo's ``GrismSpectrum``
    collapses. This is a quick-look default only; pass an explicit ``flux_1d``
    (e.g. a proper optimal extraction) for science.

    Parameters
    ----------
    flux_2d : numpy.ndarray
        2-D array of shape ``(n_spatial, n_wave)``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_wave,)``.
    """
    valid = np.isfinite(flux_2d)
    summed = np.where(valid, flux_2d, 0.0).sum(axis=0)
    return np.where(valid.any(axis=0), summed, np.nan)


def _collapse_with_error(
    flux_2d: np.ndarray, error_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse flux (boxcar sum) and its error (quadrature) along the spatial axis.

    A cell contributes only where flux and error are both finite and the error
    is positive; the propagated error is ``sqrt(sum(error**2))`` over those
    cells, and a wavelength column with no valid cell is ``NaN`` in both
    outputs. Mirrors ``GrismSpectrum.collapse`` (minus its weight / contam
    terms, which the plotter does not carry); a quick-look default only.

    Parameters
    ----------
    flux_2d, error_2d : numpy.ndarray
        2-D arrays of the same shape ``(n_spatial, n_wave)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(flux_1d, error_1d)``, each of shape ``(n_wave,)``.
    """
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(flux_2d) & np.isfinite(error_2d) & (error_2d > 0)
    n_valid = valid.sum(axis=0)
    flux_1d = np.where(valid, flux_2d, 0.0).sum(axis=0)
    var_1d = np.where(valid, error_2d**2, 0.0).sum(axis=0)
    has = n_valid > 0
    return (
        np.where(has, flux_1d, np.nan),
        np.where(has, np.sqrt(var_1d), np.nan),
    )


def _parse_window(window: Sequence[float]) -> tuple[float, float]:
    """Validate a ``spatial_window`` into a sorted ``(low, high)`` pair.

    Accepts any length-2 sequence (tuple or list); the order is normalised so
    ``low <= high``.

    Raises
    ------
    ValueError
        If ``window`` does not have exactly two elements.
    """
    win = list(window)
    if len(win) != 2:
        raise ValueError(f"spatial_window must be (low, high), got {window!r}")
    low, high = sorted(float(v) for v in win)
    return low, high


def _window_row_mask(spatial: np.ndarray, window: Sequence[float]) -> np.ndarray:
    """Boolean mask of the spatial rows whose centre lies inside ``window``.

    Raises
    ------
    ValueError
        If ``window`` is malformed (see :func:`_parse_window`) or selects no row.
    """
    low, high = _parse_window(window)
    mask = (spatial >= low) & (spatial <= high)
    if not mask.any():
        raise ValueError(
            f"spatial_window {(low, high)} selects no rows of the spatial axis"
        )
    return mask


def _draw_image(
    ax: Axes,
    flux_2d: np.ndarray,
    wave_edges: np.ndarray,
    spatial_edges: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    window: tuple[float, float] | None = None,
) -> None:
    """Draw the 2-D spectrum image onto ``ax`` (NaN cells transparent).

    If ``window`` is given, a translucent band with dashed edges marks the
    ``(low, high)`` cross-dispersion extent of the collapse aperture.
    """
    masked = np.ma.masked_invalid(flux_2d)
    ax.pcolormesh(
        wave_edges,
        spatial_edges,
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    if window is not None:
        low, high = window
        ax.axhspan(low, high, color=_WINDOW_COLOR, alpha=0.12, zorder=2)
        for edge in (low, high):
            ax.axhline(edge, color=_WINDOW_COLOR, lw=1.0, ls="--", zorder=3)
    ax.set_ylim(float(spatial_edges[0]), float(spatial_edges[-1]))


def plot_spectrum2d(
    wavelength: ArrayLike,
    flux_2d: ArrayLike,
    *,
    flux_1d: ArrayLike | None = None,
    spatial: ArrayLike | None = None,
    spatial_window: Sequence[float] | None = None,
    error: ArrayLike | None = None,
    error_2d: ArrayLike | None = None,
    label: str | None = None,
    color: str | None = None,
    models: ModelSpec | Callable | tuple | Sequence | None = None,
    drawstyle: Literal["steps", "line"] = "steps",
    error_style: Literal["band", "line", "none"] = "band",
    cmap: str = "Greys",
    vmin: float | None = None,
    vmax: float | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylog: bool = False,
    x_label: str = "Wavelength",
    y_label: str = "Flux",
    spatial_label: str = "Spatial",
    height_ratios: tuple[float, float] = (1.0, 2.0),
    size: int = 680,
    title: str | None = None,
    save: str | Path | None = None,
) -> Figure:
    """Plot a 2-D spectrum over its 1-D collapse on a shared wavelength axis.

    The top panel shows ``flux_2d`` as an image (cross-dispersion vs wavelength,
    empty NaN cells transparent); the bottom panel shows the 1-D spectrum with
    the same model-overlay / uncertainty-band rendering as
    :func:`~noobfriend.core.display.plot._spectrum1d.plot_spectrum1d`. The
    wavelength axis is shared, so the panels stay aligned and pan/zoom together.

    Parameters
    ----------
    wavelength : array_like
        1-D wavelength grid (the bin centers), length ``n_wave``; shared by both
        panels. Any consistent numeric unit.
    flux_2d : array_like
        2-D spectrum of shape ``(n_spatial, n_wave)``. ``NaN`` cells render
        transparent.
    flux_1d : array_like, optional
        The 1-D spectrum, length ``n_wave``. Defaults to a NaN-aware spatial sum
        of ``flux_2d`` (quick-look only; pass a proper extraction for science).
    spatial : array_like, optional
        1-D cross-dispersion coordinate (centers), length ``n_spatial``.
        Defaults to the pixel index.
    spatial_window : sequence of float, optional
        ``(low, high)`` collapse aperture in spatial coordinates (any length-2
        tuple or list; order-insensitive). Marked on the 2-D panel as a dashed
        band. When ``flux_1d`` is omitted it also restricts the quick-look
        collapse to the rows inside it; with an explicit ``flux_1d`` it is an
        annotation only (the caller already chose the extraction).
    error : array_like, optional
        1-D uncertainty for the bottom panel, length ``n_wave`` (see
        ``error_style``). Mutually exclusive with ``error_2d``.
    error_2d : array_like, optional
        2-D uncertainty of the same shape as ``flux_2d``. When ``flux_1d`` is
        omitted, it is collapsed in quadrature over the same rows as the
        flux (the ``spatial_window`` if given) to produce the 1-D error, so the
        band matches the collapsed line. Valid only with ``flux_1d`` omitted and
        ``error`` not given.
    label : str, optional
        Legend label for the 1-D line.
    color : str, optional
        Color of the 1-D line; defaults to the matplotlib cycle.
    models : ModelSpec or callable or tuple or sequence, optional
        Model curves overlaid on the 1-D panel; see
        :class:`~noobfriend.core.display.plot._spectrum.ModelSpec`.
    drawstyle : {"steps", "line"}, default "steps"
        Rendering of the 1-D data spectrum.
    error_style : {"band", "line", "none"}, default "band"
        Rendering of the 1-D uncertainty.
    cmap : str, default "Greys"
        Matplotlib colormap name for the 2-D image.
    vmin, vmax : float, optional
        Explicit image color limits; a side left unset falls back to the
        ``pmin``/``pmax`` percentile of ``flux_2d``.
    pmin, pmax : float, default 1.0, 99.0
        Percentile cuts for the image stretch, used for whichever of
        ``vmin``/``vmax`` is unset.
    xlim : tuple of float, optional
        Wavelength limits (shared by both panels). Defaults to the image extent.
    ylim : tuple of float, optional
        Flux limits for the 1-D panel. Left unset, matplotlib autoscales.
    ylog : bool, default False
        Use a logarithmic flux axis on the 1-D panel.
    x_label, y_label : str, default "Wavelength", "Flux"
        Wavelength-axis and 1-D-flux-axis labels. No unit is assumed.
    spatial_label : str, default "Spatial"
        Cross-dispersion axis label of the 2-D panel.
    height_ratios : tuple of float, default (1.0, 2.0)
        Height ratio of the (top 2-D, bottom 1-D) panels.
    size : int, default 680
        Figure width in pixels; the height follows a fixed aspect ratio.
    title : str, optional
        Figure title (placed above the 2-D panel).
    save : str or pathlib.Path, optional
        If given, write the figure to this path with ``savefig``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled two-panel figure.

    Raises
    ------
    ValueError
        If ``flux_2d`` is not 2-D; if ``wavelength``/``flux_1d``/``spatial``/
        ``error_2d`` do not match its shape; if both ``error`` and ``error_2d``
        are given, or ``error_2d`` is given with an explicit ``flux_1d``; or if
        ``spatial_window`` is malformed or selects no rows.
    """
    import matplotlib.pyplot as plt

    wave = np.asarray(wavelength, dtype=float)
    image = np.asarray(flux_2d, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"flux_2d must be 2-D, got shape {image.shape}")
    n_spatial, n_wave = image.shape
    if wave.shape != (n_wave,):
        raise ValueError(
            f"wavelength length {wave.shape} != flux_2d wavelength axis {n_wave}"
        )
    spat = (
        np.arange(n_spatial, dtype=float)
        if spatial is None
        else np.asarray(spatial, dtype=float)
    )
    if spat.shape != (n_spatial,):
        raise ValueError(
            f"spatial length {spat.shape} != flux_2d spatial axis {n_spatial}"
        )
    win_bounds = None if spatial_window is None else _parse_window(spatial_window)

    if error is not None and error_2d is not None:
        raise ValueError("give either error (1-D) or error_2d, not both")

    err_1d = error
    if flux_1d is not None:
        if error_2d is not None:
            raise ValueError(
                "error_2d only applies when flux_1d is omitted (the quick-look "
                "collapse path); pass an explicit 1-D error with flux_1d"
            )
        line = np.asarray(flux_1d, dtype=float)
        if line.shape != (n_wave,):
            raise ValueError(f"flux_1d length {line.shape} != n_wave {n_wave}")
    else:
        mask = (
            None if spatial_window is None else _window_row_mask(spat, spatial_window)
        )
        img_rows = image if mask is None else image[mask]
        if error_2d is None:
            line = _collapse_2d(img_rows)
        else:
            err2d = np.asarray(error_2d, dtype=float)
            if err2d.shape != image.shape:
                raise ValueError(
                    f"error_2d shape {err2d.shape} != flux_2d shape {image.shape}"
                )
            err2d_rows = err2d if mask is None else err2d[mask]
            line, err_1d = _collapse_with_error(img_rows, err2d_rows)

    wave_edges = _centers_to_edges(wave)
    spatial_edges = _centers_to_edges(spat)
    low, high = resolve_limits(image, vmin, vmax, pmin, pmax)
    if xlim is None:
        xlim = (float(wave_edges[0]), float(wave_edges[-1]))

    width_in = size / 100.0
    fig, (ax2d, ax1d) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(width_in, width_in * _DEFAULT_ASPECT2D),
        height_ratios=list(height_ratios),
        layout="constrained",
    )
    fig.get_layout_engine().set(hspace=0)

    _draw_image(
        ax2d,
        image,
        wave_edges,
        spatial_edges,
        cmap=cmap,
        vmin=low,
        vmax=high,
        window=win_bounds,
    )
    ax2d.set_ylabel(spatial_label)
    if title:
        ax2d.set_title(title)

    draw_spectrum(
        ax1d,
        wave,
        line,
        error=err_1d,
        labels=label,
        colors=color,
        models=models,
        drawstyle=drawstyle,
        error_style=error_style,
        xlim=xlim,
        ylim=ylim,
        ylog=ylog,
        x_label=x_label,
        y_label=y_label,
    )
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
