"""Interactive Bokeh image viewer for large 2-D detector frames.

The single public entry point :func:`imshow` renders a 2-D array as one Bokeh
``image`` glyph (not a per-pixel heatmap), so pan/zoom and the hover value
read-out stay client-side and cheap regardless of frame size. The figure is
sized to the data's aspect ratio with square pixels, and hovering reports the
real pixel value (every stretch is a native Bokeh color mapper over the
unmodified data).
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from bokeh.models import (
    ColorMapper,
    CustomJSHover,
    DataRange1d,
    EqHistColorMapper,
    HoverTool,
    LinearColorMapper,
    LogColorMapper,
)
from bokeh.palettes import (
    Cividis256,
    Greys256,
    Inferno256,
    Magma256,
    Plasma256,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure

from noobfriend.core.display.plot._bokeh import display
from noobfriend.core.display.plot._norm import resolve_limits

#: Selectable colormaps, ``name -> Bokeh palette``.
CMAPS: dict[str, Sequence[str]] = {
    "Greys": Greys256,
    "Greys (inv)": tuple(reversed(Greys256)),
    "Viridis": Viridis256,
    "Cividis": Cividis256,
    "Magma": Magma256,
    "Inferno": Inferno256,
    "Plasma": Plasma256,
    "Turbo": Turbo256,
}

#: Selectable stretches; each maps to a native Bokeh color mapper.
STRETCHES: tuple[str, ...] = ("linear", "log", "eqhist")

_NAN_COLOR = "rgba(0, 0, 0, 0)"


def _log_low(low: float, high: float) -> float:
    """Clamp a lower limit to a positive value usable by :class:`LogColorMapper`.

    Background-subtracted detector frames routinely contain non-positive pixels,
    for which a log stretch is undefined; this floors the lower bound just above
    zero (a small fraction of the upper bound) so the log mapper stays valid.
    """
    if low > 0:
        return low
    return high * 1e-4 if high > 0 else 1e-6


def _make_mapper(
    stretch: str, palette: Sequence[str], low: float, high: float
) -> ColorMapper:
    """Build the native Bokeh color mapper for ``stretch`` over ``[low, high]``."""
    if stretch == "linear":
        return LinearColorMapper(
            palette=palette, low=low, high=high, nan_color=_NAN_COLOR
        )
    if stretch == "log":
        return LogColorMapper(
            palette=palette, low=_log_low(low, high), high=high, nan_color=_NAN_COLOR
        )
    return EqHistColorMapper(palette=palette, low=low, high=high, nan_color=_NAN_COLOR)


#: Target spacing, in pixels, between RA/Dec grid nodes (see :func:`_grid_axis`).
_GRID_STEP: int = 32


def _grid_axis(n: int, step: int = _GRID_STEP, max_pts: int = 129) -> int:
    """Return the number of RA/Dec grid samples along an axis of ``n`` pixels.

    The count scales with ``n`` at roughly one node per ``step`` pixels (so the
    grid adapts to the data shape and keeps cells near-square in pixel space),
    clamped to ``[2, max_pts]``. A WCS varies smoothly over a detector, so this
    coarse grid bilinearly interpolates to sub-milliarcsecond accuracy.
    """
    return int(np.clip(round(n / step) + 1, 2, max_pts))


#: Client-side RA/Dec read-out: bilinearly interpolate the coarse grid at the
#: hovered integer pixel, then format as decimal degrees or sexagesimal.
_RADEC_JS = r"""
const x = special_vars.x, y = special_vars.y;
if (x == null || y == null || x < 0 || x > w1 + 1 || y < 0 || y > h1 + 1) return "";
const i = Math.floor(x), j = Math.floor(y);
const fx = w1 > 0 ? (i / w1) * (Gx - 1) : 0;
const fy = h1 > 0 ? (j / h1) * (Gy - 1) : 0;
const kx = Math.min(Math.max(Math.floor(fx), 0), Gx - 2);
const ky = Math.min(Math.max(Math.floor(fy), 0), Gy - 2);
const tx = fx - kx, ty = fy - ky;
function bilerp(a) {
    return a[ky * Gx + kx] * (1 - tx) * (1 - ty) + a[ky * Gx + kx + 1] * tx * (1 - ty)
         + a[(ky + 1) * Gx + kx] * (1 - tx) * ty + a[(ky + 1) * Gx + kx + 1] * tx * ty;
}
let ra = (((ra0 + bilerp(ra_off)) % 360) + 360) % 360;
const dec = bilerp(dec_grid);
if (fmt === "deg") return ra.toFixed(6) + ", " + dec.toFixed(6);
function pad(n, w) { let s = String(n); while (s.length < w) s = "0" + s; return s; }
function sec(s, d) { return (s < 10 ? "0" : "") + s.toFixed(d); }
const tsec = Math.round((ra / 15) * 3600 * 1000) / 1000;
const H = Math.floor(tsec / 3600), r1 = tsec - H * 3600;
const M = Math.floor(r1 / 60), S = r1 - M * 60;
const tas = Math.round(Math.abs(dec) * 3600 * 100) / 100;
const sign = dec < 0 ? "-" : "+", D = Math.floor(tas / 3600), r2 = tas - D * 3600;
const Md = Math.floor(r2 / 60), Sd = r2 - Md * 60;
return pad(H % 24, 2) + ":" + pad(M, 2) + ":" + sec(S, 3)
     + "  " + sign + pad(D, 2) + ":" + pad(Md, 2) + ":" + sec(Sd, 2);
"""


def _radec_formatter(
    transform: Callable[[Any, Any], tuple[Any, Any]],
    n_rows: int,
    n_cols: int,
    coord_format: str,
) -> CustomJSHover:
    """Build a hover formatter that reports the hovered pixel's RA/Dec.

    A coarse grid of RA/Dec is evaluated once through ``transform`` (sized by
    :func:`_grid_axis`) and shipped to the browser, where :data:`_RADEC_JS`
    bilinearly interpolates it at the hovered pixel. RA is carried as an offset
    from the field centre (wrapped to ``[-180, 180)``) so interpolation is
    seam-safe across RA = 0/360 and keeps full precision.

    Parameters
    ----------
    transform : callable
        ``f(x, y) -> (ra, dec)`` mapping two equal-length 1-D arrays of pixel
        coordinates to RA/Dec arrays in degrees. Called once over the whole
        grid (e.g. a GWCS/astropy ``pixel_to_world_values`` for a direct image,
        or the grism ``detector_to_world`` from
        :func:`noobfriend.extraction._wcs.world_detector_transforms`).
    n_rows, n_cols : int
        Image shape.
    coord_format : str
        ``"deg"`` (decimal degrees) or ``"hms"`` (sexagesimal).

    Returns
    -------
    bokeh.models.CustomJSHover
        Formatter wired with the precomputed grid.

    Raises
    ------
    ValueError
        If ``transform`` fails on two pixel-coordinate arrays.
    """
    gx = np.linspace(0.0, n_cols - 1, _grid_axis(n_cols))
    gy = np.linspace(0.0, n_rows - 1, _grid_axis(n_rows))
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    try:
        ra, dec = transform(mesh_x.ravel(), mesh_y.ravel())
    except Exception as e:
        raise ValueError(
            "transform_pixel_to_world(x, y) failed; it must accept two "
            "equal-length 1-D pixel-coordinate arrays and return (ra, dec) "
            "in degrees."
        ) from e
    ra = np.asarray(ra, dtype=float).reshape(mesh_x.shape)
    dec = np.asarray(dec, dtype=float).reshape(mesh_x.shape)
    ra0 = float(ra[ra.shape[0] // 2, ra.shape[1] // 2])
    ra_off = ((ra - ra0 + 180.0) % 360.0) - 180.0
    return CustomJSHover(
        args={
            "ra_off": ra_off.ravel().tolist(),
            "dec_grid": dec.ravel().tolist(),
            "ra0": ra0,
            "Gx": int(gx.size),
            "Gy": int(gy.size),
            "w1": float(n_cols - 1),
            "h1": float(n_rows - 1),
            "fmt": coord_format,
        },
        code=_RADEC_JS,
    )


def imshow(
    data: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    cmap: str = "Greys",
    stretch: str = "linear",
    size: int = 620,
    title: str | None = None,
    transform_pixel_to_world: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    coord_format: Literal["deg", "hms"] = "deg",
) -> Any:
    """Display a 2-D array as a zoomable, hoverable Bokeh image.

    The figure is sized to the data's aspect ratio (long edge = ``size``) with
    square pixels, so the image fills the plot area without distortion. The
    array is drawn origin-at-lower-left (FITS convention), so ``data[0, 0]`` is
    the bottom-left pixel and data coordinates equal pixel indices. Hovering
    reports the cursor ``(x, y)`` and the real pixel value; pan, wheel-zoom,
    box-zoom, reset, and save tools are attached.

    Parameters
    ----------
    data : numpy.ndarray
        A 2-D image array. It is cast to ``float32`` before upload to halve the
        transferred payload (a 2048x2048 frame is ~17 MB at float32).
    vmin, vmax : float, optional
        Explicit lower/upper limits for the color stretch. Each takes
        precedence over the corresponding percentile cut; a side left as
        ``None`` falls back to ``pmin``/``pmax`` (so the two can be mixed,
        e.g. lock ``vmin`` while the upper bound still tracks ``pmax``).
    pmin, pmax : float, default 1.0, 99.0
        Percentile cuts used for whichever of ``vmin``/``vmax`` is not given.
    cmap : str, default ``"Greys"``
        Colormap; one of the keys of :data:`CMAPS`.
    stretch : str, default ``"linear"``
        One of :data:`STRETCHES` (``"linear"``, ``"log"``, ``"eqhist"``).
        ``"eqhist"`` (histogram equalization) is often more informative for
        high-dynamic-range astronomical frames.
    size : int, default 620
        Length of the longer figure edge, in pixels; the other edge follows the
        data's aspect ratio.
    title : str, optional
        Figure title.
    transform_pixel_to_world : callable, optional
        ``f(x, y) -> (ra, dec)`` mapping two equal-length 1-D pixel-coordinate
        arrays to RA/Dec arrays in degrees -- e.g. ``wcs.pixel_to_world_values``
        for a direct image, or ``world_detector_transforms(wcs)[1]`` for grism.
        When given, the hover gains an ``RA, Dec`` row for the hovered pixel,
        resolved client-side from a coarse precomputed grid (no kernel
        round-trip). Keeping this a plain callable lets the caller handle any
        WCS flavour; ``imshow`` does not.
    coord_format : {"deg", "hms"}, default ``"deg"``
        RA/Dec hover format: ``"deg"`` (decimal degrees, 6 dp) or ``"hms"``
        (``HH:MM:SS.sss +DD:MM:SS.ss``).

    Returns
    -------
    Any
        A display handle (see :func:`~noobfriend.core.display.plot.set_render_mode`):
        an HTML iframe by default, or a ``jupyter_bokeh`` widget. Renders
        automatically as a notebook cell's last expression.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, ``pmin``/``pmax`` are invalid, ``cmap`` /
        ``stretch`` / ``coord_format`` is not a known option, or
        ``transform_pixel_to_world`` fails on two pixel-coordinate arrays.
    """
    if cmap not in CMAPS:
        raise ValueError(f"cmap must be one of {list(CMAPS)}, got {cmap!r}")
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {list(STRETCHES)}, got {stretch!r}")
    if coord_format not in ("deg", "hms"):
        raise ValueError(f"coord_format must be 'deg' or 'hms', got {coord_format!r}")

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"imshow expects a 2-D array, got shape {data.shape}")
    arr = data.astype(np.float32, copy=False)
    n_rows, n_cols = arr.shape

    # Size the figure to the data's aspect ratio so the image fills it squarely.
    if n_cols >= n_rows:
        fig_w, fig_h = size, max(1, round(size * n_rows / n_cols))
    else:
        fig_w, fig_h = max(1, round(size * n_cols / n_rows)), size

    low, high = resolve_limits(arr, vmin, vmax, pmin, pmax)
    mapper = _make_mapper(stretch, CMAPS[cmap], low, high)

    fig = figure(
        width=fig_w,
        height=fig_h,
        title=title or "",
        x_range=DataRange1d(range_padding=0),
        y_range=DataRange1d(range_padding=0),
        match_aspect=True,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    image = fig.image(image=[arr], x=0, y=0, dw=n_cols, dh=n_rows, color_mapper=mapper)
    tooltips = [("(x, y)", "$x{0.0}, $y{0.0}"), ("value", "@image")]
    formatters: dict[str, Any] = {}
    if transform_pixel_to_world is not None:
        tooltips.append(("RA, Dec", "$x{custom}"))
        formatters["$x"] = _radec_formatter(
            transform_pixel_to_world, n_rows, n_cols, coord_format
        )
    fig.add_tools(
        HoverTool(renderers=[image], tooltips=tooltips, formatters=formatters)
    )
    return display(fig, height=fig_h + 80)
