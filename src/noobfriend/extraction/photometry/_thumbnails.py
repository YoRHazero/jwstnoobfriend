"""Matplotlib aperture thumbnails for an :class:`ApertureSEDResult`.

Isolated from :mod:`noobfriend.extraction.photometry._result` so the result
object stays about data and the (heavier) matplotlib import stays local to the
functions that draw -- the same split as the SED plot in
:mod:`noobfriend.extraction.photometry._plot`.

Each band is drawn on its *own* native grid, so the two overlays need no
reprojection and line up with the science thumbnail pixel-for-pixel:

- the science thumbnail is ``band_images[name]`` (NaN pixels transparent);
- the band's own grown aperture is ``source_apertures[name].mask``, drawn as a
  crisp contour line -- "this band's mask";
- the union aperture as it was actually measured on this band is
  ``band_coverage[name]`` (fractional coverage in ``[0, 1]``), drawn as a
  translucent fill whose opacity tracks the coverage fraction so the soft,
  partially-weighted boundary pixels of the native-grid measurement are visible
  -- "the total mask".

The reference band's panel shows ``band_coverage == union_mask`` (binary), i.e.
the total mask at its finest native resolution, so no separate union panel is
needed.

Like the spectrum plotters this is a static matplotlib figure (multi-panel
montage with overlays, trivial ``savefig`` export); the interactive Bokeh
``imshow`` is for single-frame inspection, not diagnostic montages.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from noobfriend.core.display.plot._norm import resolve_limits

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from noobfriend.extraction.photometry._result import ApertureSEDResult

#: Supported intensity stretches for the thumbnail background.
_STRETCHES: tuple[str, ...] = ("linear", "log")

#: Default thumbnail colormap. A dark-background greyscale (low -> black) so the
#: bright, saturated overlay colours below stay legible; "Greys" would put them
#: on white and wash them out.
_DEFAULT_CMAP: str = "gray"

#: Default overlay / marker colours.
_APERTURE_COLOR: str = "#28c8ff"
_COVERAGE_COLOR: str = "#ff7a33"
_SEED_COLOR: str = "#ffd400"


def _log_low(low: float, high: float) -> float:
    """Floor a lower limit to a positive value usable by a log stretch."""
    if low > 0.0:
        return low
    return high * 1e-4 if high > 0.0 else 1e-6


def _make_norm(stretch: str, low: float, high: float) -> Any:
    """Build the matplotlib normalize for ``stretch`` over ``[low, high]``."""
    from matplotlib.colors import LogNorm, Normalize

    if stretch == "linear":
        return Normalize(vmin=low, vmax=high)
    if stretch == "log":
        lo = _log_low(low, high)
        return LogNorm(vmin=lo, vmax=max(high, lo * 10.0))
    raise ValueError(f"stretch must be one of {list(_STRETCHES)}, got {stretch!r}")


def _aperture_bbox(footprint: np.ndarray, pad: float) -> tuple[slice, slice] | None:
    """Return ``(row_slice, col_slice)`` around ``footprint`` padded by ``pad``.

    ``footprint`` is the boolean union of the band's own aperture and the
    covered union pixels. The tight bounding box is grown on every side by
    ``pad`` times its own extent (so the source fills the thumbnail with a
    margin) and clipped to the array. Returns ``None`` if ``footprint`` is
    empty.
    """
    rows = np.any(footprint, axis=1)
    cols = np.any(footprint, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = (int(i) for i in np.where(rows)[0][[0, -1]])
    c0, c1 = (int(i) for i in np.where(cols)[0][[0, -1]])
    dr = int(round((r1 - r0 + 1) * pad))
    dc = int(round((c1 - c0 + 1) * pad))
    n_rows, n_cols = footprint.shape
    r0, r1 = max(0, r0 - dr), min(n_rows - 1, r1 + dr)
    c0, c1 = max(0, c0 - dc), min(n_cols - 1, c1 + dc)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _draw_band_panel(
    ax: Axes,
    data: np.ndarray,
    own_mask: np.ndarray,
    coverage: np.ndarray,
    seed_xy: tuple[int, int] | None,
    *,
    cmap: str,
    stretch: str,
    vmin: float | None,
    vmax: float | None,
    pmin: float,
    pmax: float,
    crop: bool,
    crop_pad: float,
    aperture_color: str,
    coverage_color: str,
    coverage_alpha: float,
    show_coverage: bool,
    show_aperture: bool,
    show_seed: bool,
) -> None:
    """Draw one band's thumbnail with its aperture and union overlays onto ``ax``.

    The science image, the binary own-aperture contour, the fractional union
    fill, and the seed marker all share one absolute pixel-index frame (origin
    at lower-left, FITS convention), so they align without any coordinate
    bookkeeping; cropping is applied as axis limits, not by slicing, to keep
    that frame intact. The colour stretch is computed over the cropped window so
    a faint source is not flattened by bright pixels elsewhere in the cutout.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    data = np.asarray(data, dtype=float)
    own_mask = np.asarray(own_mask, dtype=bool)
    coverage = np.clip(np.asarray(coverage, dtype=float), 0.0, 1.0)

    bbox = _aperture_bbox(own_mask | (coverage > 0.0), crop_pad) if crop else None
    window = data if bbox is None else data[bbox]
    low, high = resolve_limits(window, vmin, vmax, pmin, pmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    ax.imshow(
        np.ma.masked_invalid(data),
        origin="lower",
        cmap=cmap_obj,
        norm=_make_norm(stretch, low, high),
        interpolation="nearest",
    )

    if show_coverage and coverage.any():
        rgba = np.zeros((*coverage.shape, 4), dtype=float)
        rgba[..., :3] = to_rgb(coverage_color)
        rgba[..., 3] = coverage * coverage_alpha
        ax.imshow(rgba, origin="lower", interpolation="nearest")

    if show_aperture and own_mask.any():
        ax.contour(
            own_mask.astype(float),
            levels=[0.5],
            colors=[aperture_color],
            linewidths=1.3,
        )

    if show_seed and seed_xy is not None:
        ax.plot(
            seed_xy[0],
            seed_xy[1],
            marker="x",
            color=_SEED_COLOR,
            markersize=7,
            markeredgewidth=1.6,
        )

    if bbox is not None:
        rsl, csl = bbox
        ax.set_xlim(csl.start - 0.5, csl.stop - 0.5)
        ax.set_ylim(rsl.start - 0.5, rsl.stop - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _validate_bands(result: ApertureSEDResult, bands: Any) -> list[str]:
    """Resolve the band list to draw, defaulting to wavelength-sorted order."""
    if bands is None:
        return [m.band for m in result.measurements]
    names = [str(b) for b in bands]
    if not names:
        raise ValueError("bands must name at least one band.")
    unknown = [n for n in names if n not in result.band_images]
    if unknown:
        raise ValueError(
            f"unknown band(s) {unknown}; available: {sorted(result.band_images)}."
        )
    return names


def _panel_title(result: ApertureSEDResult, name: str) -> str:
    """Return ``"NAME  λ µm"`` when the band carries a wavelength, else ``NAME``."""
    for m in result.measurements:
        if m.band == name and m.wavelength is not None:
            return f"{name}  {m.wavelength:g} µm"
    return name


def _draw_badges(ax: Axes, result: ApertureSEDResult, name: str) -> None:
    """Mark the reference band (top-left) and a flagged band (top-right)."""
    if name == result.reference_band:
        ax.text(
            0.04,
            0.96,
            "ref",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="white",
            bbox={"facecolor": "#1f6f8b", "edgecolor": "none", "pad": 1.5},
        )
    flagged = any(m.band == name and m.flagged for m in result.measurements)
    if flagged:
        ax.text(
            0.96,
            0.96,
            "flagged",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=7,
            color="white",
            bbox={"facecolor": "#c43b3b", "edgecolor": "none", "pad": 1.5},
        )


def _draw_panel_for_band(
    ax: Axes,
    result: ApertureSEDResult,
    name: str,
    **panel_kwargs: Any,
) -> None:
    """Pull one band's arrays off the result and draw its panel onto ``ax``."""
    aperture = result.source_apertures[name]
    _draw_band_panel(
        ax,
        result.band_images[name],
        aperture.mask,
        result.band_coverage[name],
        aperture.seed_xy,
        **panel_kwargs,
    )


def aperture_montage(
    result: ApertureSEDResult,
    *,
    bands: Any = None,
    ncols: int | None = None,
    crop: bool = True,
    crop_pad: float = 0.3,
    cmap: str = _DEFAULT_CMAP,
    stretch: Literal["linear", "log"] = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    aperture_color: str = _APERTURE_COLOR,
    coverage_color: str = _COVERAGE_COLOR,
    coverage_alpha: float = 0.5,
    show_coverage: bool = True,
    show_aperture: bool = True,
    show_seed: bool = True,
    panel_size: float = 2.4,
    title: str | None = None,
    save: str | Path | None = None,
) -> Figure:
    """Montage every band's thumbnail with its aperture and union overlays.

    One panel per band, each on the band's own native grid: the greyscale
    science thumbnail, the band's grown aperture as a contour, and the union
    aperture as it was measured on this band (``band_coverage``) as a
    translucent fill whose opacity tracks the coverage fraction. The seed pixel
    is marked, the reference band and any flagged band are badged, and each
    panel is cropped to its source with a margin.

    Parameters
    ----------
    result : ApertureSEDResult
        The measured result carrying ``band_images``, ``band_coverage``, and
        ``source_apertures``.
    bands : sequence of str, optional
        Bands to draw, in this order. Defaults to all bands in the result's
        (wavelength-sorted) measurement order.
    ncols : int, optional
        Number of columns. Defaults to roughly the square root of the band
        count.
    crop : bool, default True
        Crop each panel to its aperture footprint (own aperture plus covered
        union pixels) with a ``crop_pad`` margin.
    crop_pad : float, default 0.3
        Margin added on each side as a fraction of the footprint extent.
    cmap : str, default "gray"
        Matplotlib colormap name for the science thumbnail.
    stretch : {"linear", "log"}, default "linear"
        Intensity stretch for the thumbnail background.
    vmin, vmax : float, optional
        Explicit colour limits; a side left unset falls back to the
        ``pmin``/``pmax`` percentile of the cropped window.
    pmin, pmax : float, default 1.0, 99.0
        Percentile cuts used for whichever of ``vmin``/``vmax`` is unset.
    aperture_color : str, default cyan
        Contour colour of the band's own grown aperture.
    coverage_color : str, default orange
        Fill colour of the union coverage; opacity tracks the coverage
        fraction.
    coverage_alpha : float, default 0.5
        Opacity of a fully covered (coverage = 1) pixel.
    show_coverage, show_aperture, show_seed : bool, default True
        Toggle the union fill, the own-aperture contour, and the seed marker.
    panel_size : float, default 2.4
        Side length of each panel in inches.
    title : str, optional
        Figure title.
    save : str or pathlib.Path, optional
        If given, write the figure to this path with ``savefig``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled montage.

    Raises
    ------
    ValueError
        If ``bands`` names an unknown band, ``stretch`` is invalid, or the
        colour limits are inconsistent (see
        :func:`~noobfriend.core.display.plot._norm.resolve_limits`).
    """
    import matplotlib.pyplot as plt

    names = _validate_bands(result, bands)
    n = len(names)
    if ncols is None:
        ncols = max(1, round(math.sqrt(n)))
    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_size, nrows * panel_size),
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())

    panel_kwargs = {
        "cmap": cmap,
        "stretch": stretch,
        "vmin": vmin,
        "vmax": vmax,
        "pmin": pmin,
        "pmax": pmax,
        "crop": crop,
        "crop_pad": crop_pad,
        "aperture_color": aperture_color,
        "coverage_color": coverage_color,
        "coverage_alpha": coverage_alpha,
        "show_coverage": show_coverage,
        "show_aperture": show_aperture,
        "show_seed": show_seed,
    }
    for ax, name in zip(flat, names):
        _draw_panel_for_band(ax, result, name, **panel_kwargs)
        ax.set_title(_panel_title(result, name), fontsize=9)
        _draw_badges(ax, result, name)
    for ax in flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def aperture_thumbnail(
    result: ApertureSEDResult,
    band: str,
    *,
    crop: bool = True,
    crop_pad: float = 0.3,
    cmap: str = _DEFAULT_CMAP,
    stretch: Literal["linear", "log"] = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    aperture_color: str = _APERTURE_COLOR,
    coverage_color: str = _COVERAGE_COLOR,
    coverage_alpha: float = 0.5,
    show_coverage: bool = True,
    show_aperture: bool = True,
    show_seed: bool = True,
    size: float = 4.0,
    title: str | None = None,
    save: str | Path | None = None,
) -> Figure:
    """Draw one band's thumbnail with its aperture and union overlays, enlarged.

    The single-band counterpart of :func:`aperture_montage` (identical overlays
    and cropping) for inspecting one band's aperture closely.

    Parameters
    ----------
    result : ApertureSEDResult
        The measured result.
    band : str
        Band to draw.
    crop, crop_pad, cmap, stretch, vmin, vmax, pmin, pmax : see :func:`aperture_montage`
        Identical meaning.
    aperture_color, coverage_color, coverage_alpha : see :func:`aperture_montage`
        Identical meaning.
    show_coverage, show_aperture, show_seed : see :func:`aperture_montage`
        Identical meaning.
    size : float, default 4.0
        Side length of the figure in inches.
    title : str, optional
        Figure title. Defaults to the band name and wavelength.
    save : str or pathlib.Path, optional
        If given, write the figure to this path with ``savefig``.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.

    Raises
    ------
    ValueError
        If ``band`` is not in the result, or ``stretch`` / colour limits are
        invalid.
    """
    import matplotlib.pyplot as plt

    _validate_bands(result, [band])
    fig, ax = plt.subplots(figsize=(size, size), layout="constrained")
    _draw_panel_for_band(
        ax,
        result,
        band,
        cmap=cmap,
        stretch=stretch,
        vmin=vmin,
        vmax=vmax,
        pmin=pmin,
        pmax=pmax,
        crop=crop,
        crop_pad=crop_pad,
        aperture_color=aperture_color,
        coverage_color=coverage_color,
        coverage_alpha=coverage_alpha,
        show_coverage=show_coverage,
        show_aperture=show_aperture,
        show_seed=show_seed,
    )
    _draw_badges(ax, result, band)
    ax.set_title(title or _panel_title(result, band))
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
