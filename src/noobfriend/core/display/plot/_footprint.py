"""Sky-footprint plotting on a drag-to-rotate orthographic globe.

Footprints are polygons in ``(RA, Dec)`` degrees, drawn as outlines on an
orthographic projection -- the sky as seen looking at a globe. **Dragging the
plot rotates the globe** (the view centre's RA/Dec), reprojecting all
coordinates client-side by updating the data sources (no kernel round-trip); the
globe stays centred and the far hemisphere is hidden, so relative positions are
never confused by back-side footprints. JWST footprints are small, so the
projection's mild distortion is irrelevant; what matters is that points and
footprints sit at their correct relative positions, which orthographic preserves
exactly.

Footprint identity and per-corner coordinates are read by hovering the vertex
markers (footprint label, vertex index, RA/Dec); optional source points carry
their own labels and RA/Dec on hover.

No 3-D sphere is rendered and no dependency beyond Bokeh is used.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from bokeh.events import Pan, PanStart
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Range1d
from bokeh.plotting import figure

from noobfriend.core.display.plot._bokeh import display

DEFAULT_FOOTPRINT_COLOR = "#1f77b4"


def reference_center(polygons: Sequence[np.ndarray]) -> tuple[float, float]:
    """Return the ``(ra0, dec0)`` centre for a set of footprints, in degrees.

    The RA centre is a circular mean (via the mean unit vector) so it is correct
    across the 0/360 seam; the Dec centre is the plain mean. Used as the default
    initial view direction.

    Parameters
    ----------
    polygons : sequence of numpy.ndarray
        Each polygon is an ``(M, 2)`` array of ``(RA, Dec)`` degrees.

    Returns
    -------
    ra0, dec0 : float
        Centre coordinates in degrees, with ``ra0`` in ``[0, 360)``.
    """
    ra = np.concatenate([np.asarray(p, float)[:, 0] for p in polygons])
    dec = np.concatenate([np.asarray(p, float)[:, 1] for p in polygons])
    ra_rad = np.radians(ra)
    ra0 = np.degrees(np.arctan2(np.sin(ra_rad).mean(), np.cos(ra_rad).mean())) % 360.0
    return float(ra0), float(dec.mean())


def orthographic(
    ra: np.ndarray, dec: np.ndarray, center_ra: float, center_dec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic-project ``(RA, Dec)`` degrees onto the unit disk.

    The projection is the view of the celestial sphere from far away, looking
    toward ``(center_ra, center_dec)``. The near hemisphere maps inside the unit
    circle; points on the far hemisphere are returned as ``NaN`` so they are not
    drawn (orthographic is 2-to-1, so the far side must be hidden).

    Parameters
    ----------
    ra, dec : numpy.ndarray
        Coordinates in degrees.
    center_ra, center_dec : float
        View-centre direction in degrees.

    Returns
    -------
    x, y : numpy.ndarray
        Disk coordinates in ``[-1, 1]`` (``NaN`` on the far hemisphere).
    """
    a = np.radians(np.asarray(ra, float) - center_ra)
    d = np.radians(np.asarray(dec, float))
    d0 = np.radians(center_dec)
    cos_c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a)
    x = np.cos(d) * np.sin(a)
    y = np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a)
    visible = cos_c >= 0
    return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


def _graticule_raw() -> tuple[list[list[float]], list[list[float]]]:
    """Return ``(ras, decs)`` polylines for a 30-degree RA/Dec graticule."""
    ras: list[list[float]] = []
    decs: list[list[float]] = []
    dec_line = np.linspace(-90.0, 90.0, 91)
    for meridian in range(0, 360, 30):
        ras.append(np.full_like(dec_line, float(meridian)).tolist())
        decs.append(dec_line.tolist())
    ra_line = np.linspace(0.0, 360.0, 121)
    for parallel in range(-60, 90, 30):
        ras.append(ra_line.tolist())
        decs.append(np.full_like(ra_line, float(parallel)).tolist())
    return ras, decs


# Pan start: record the view centre and cursor position (in data coordinates, so
# the sensitivity is independent of zoom) to measure the drag from.
_PAN_START_JS = r"""
const s = state.data;
s.base_ra[0] = s.ra0[0];
s.base_dec[0] = s.dec0[0];
s.x0[0] = cb_obj.x;
s.y0[0] = cb_obj.y;
"""

# Pan: turn the cursor displacement (in unit-disk data coordinates) into a new
# view centre (RA, Dec) and reproject every source onto the unit disk, hiding the
# far hemisphere. Using data coords means a zoomed-in drag rotates less, so the
# control stays fine at any zoom. ``k`` is degrees per data unit (disk radius 1
# spans ~90 deg). Data y is up-positive, so the Dec term is subtracted.
_PAN_JS = r"""
const s = state.data;
const ra0 = s.base_ra[0] - (cb_obj.x - s.x0[0]) * k;
let dec0 = s.base_dec[0] - (cb_obj.y - s.y0[0]) * k;
dec0 = Math.max(-90.0, Math.min(90.0, dec0));
s.ra0[0] = ra0;
s.dec0[0] = dec0;
const d2r = Math.PI / 180.0;
const d0 = dec0 * d2r, sd0 = Math.sin(d0), cd0 = Math.cos(d0);
function pxy(ra, dec) {
    const a = (ra - ra0) * d2r, d = dec * d2r;
    const sd = Math.sin(d), cd = Math.cos(d), ca = Math.cos(a);
    if (sd0 * sd + cd0 * cd * ca < 0) return [NaN, NaN];
    return [cd * Math.sin(a), cd0 * sd - sd0 * cd * ca];
}
function flat(src) {
    const ra = src.data.ra, dec = src.data.dec, X = src.data.x, Y = src.data.y;
    for (let i = 0; i < ra.length; i++) { const p = pxy(ra[i], dec[i]); X[i] = p[0]; Y[i] = p[1]; }
    src.change.emit();
}
function nested(src) {
    const R = src.data.ras, D = src.data.decs, X = src.data.xs, Y = src.data.ys;
    for (let j = 0; j < R.length; j++) {
        const rj = R[j], dj = D[j], xj = X[j], yj = Y[j];
        for (let i = 0; i < rj.length; i++) { const p = pxy(rj[i], dj[i]); xj[i] = p[0]; yj[i] = p[1]; }
    }
    src.change.emit();
}
nested(grat_src);
nested(foot_src);
flat(vertex_src);
flat(point_src);
"""

# Keep the globe centred: after any range change (e.g. a wheel zoom, which zooms
# toward the cursor and so off-centre), re-centre each axis symmetrically about
# the origin. The lock flag prevents the re-entrant range writes from looping.
_RECENTER_JS = r"""
const s = state.data;
if (s.zoom_lock[0]) return;
s.zoom_lock[0] = 1;
const hx = (xr.end - xr.start) / 2.0;
const hy = (yr.end - yr.start) / 2.0;
xr.start = -hx; xr.end = hx;
yr.start = -hy; yr.end = hy;
s.zoom_lock[0] = 0;
"""


def _resolve_colors(
    colors: str | Sequence[str] | None, n: int, default: Sequence[str], name: str
) -> list[str]:
    """Resolve a colour spec into a length-``n`` list of colours.

    ``None`` yields ``default``; a single string is broadcast to all ``n``
    items; a sequence is used verbatim and must have length ``n``.

    Raises
    ------
    ValueError
        If a sequence is given whose length is not ``n``.
    """
    if colors is None:
        return list(default)
    if isinstance(colors, str):
        return [colors] * n
    resolved = list(colors)
    if len(resolved) != n:
        raise ValueError(f"{name} length {len(resolved)} != {n}")
    return resolved


def plot_footprints(
    polygons: Sequence[np.ndarray],
    *,
    points: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    colors: str | Sequence[str] | None = None,
    point_labels: Sequence[str] | None = None,
    point_colors: str | Sequence[str] | None = None,
    center_ra: float | None = None,
    center_dec: float | None = None,
    size: int = 620,
    title: str = "Footprints",
) -> Any:
    """Draw sky footprints (and optional sources) on a drag-to-rotate globe.

    Footprints are outlined on the unit disk with a hoverable marker at each
    vertex; the hover shows the footprint label, the vertex index, and the
    vertex RA/Dec. Source positions, if given, are over-plotted with their RA/Dec
    (and optional label) on hover. **Drag the plot to rotate** the globe to any
    direction (the view centre's RA/Dec); rotation is client-side, so it works
    without a kernel. The far hemisphere is hidden, so only footprints facing the
    viewer are shown -- drag to bring others into view. The globe stays centred.

    Parameters
    ----------
    polygons : sequence of numpy.ndarray
        Each polygon is an ``(M, 2)`` array of ``(RA, Dec)`` degrees, vertices
        in order. Vertex indices in the hover are ``0 .. M-1`` in this order.
    points : numpy.ndarray, optional
        ``(N, 2)`` array of ``(RA, Dec)`` degrees to over-plot.
    labels : sequence of str, optional
        Per-footprint labels shown in the vertex hover. Defaults to ``#0, #1,
        ...``. Must match ``len(polygons)`` if given.
    colors : str or sequence of str, optional
        Footprint colours: a single colour applied to all, or one per footprint
        (length ``len(polygons)``). Defaults to one blue colour for every
        footprint.
    point_labels : sequence of str, optional
        Per-point labels shown in the point hover. Requires ``points`` and must
        match ``len(points)`` if given.
    point_colors : str or sequence of str, optional
        Source-point colours: a single colour applied to all, or one per point
        (length ``len(points)``). Requires ``points``. Defaults to red.
    center_ra, center_dec : float, optional
        Initial view-centre direction in degrees. Defaults to the circular-mean
        RA and mean Dec of all footprint vertices.
    size : int, default 620
        Figure size in pixels (square).
    title : str, default ``"Footprints"``
        Figure title.

    Returns
    -------
    Any
        A display handle (see :func:`~noobfriend.core.display.plot.set_render_mode`).
        Renders automatically as a notebook cell's last expression.

    Raises
    ------
    ValueError
        If ``polygons`` is empty; ``labels``/``colors``/``point_labels``/
        ``point_colors`` lengths mismatch; ``point_labels`` or ``point_colors``
        is given without ``points``; or ``points`` is not ``(N, 2)``.
    """
    polys = [np.asarray(p, float) for p in polygons]
    if not polys:
        raise ValueError("polygons is empty")
    if labels is not None and len(labels) != len(polys):
        raise ValueError(
            f"labels length {len(labels)} != number of polygons {len(polys)}"
        )
    default_ra, default_dec = reference_center(polys)
    center_ra = default_ra if center_ra is None else float(center_ra)
    center_dec = default_dec if center_dec is None else float(center_dec)
    names = list(labels) if labels is not None else [f"#{i}" for i in range(len(polys))]
    colors = _resolve_colors(
        colors,
        len(polys),
        [DEFAULT_FOOTPRINT_COLOR] * len(polys),
        "colors",
    )

    # Footprint outlines (closed loops) and a hoverable marker at each original
    # vertex (no closing duplicate; indices 0 .. M-1).
    foot_ras, foot_decs, foot_xs, foot_ys = [], [], [], []
    v_x, v_y, v_ra, v_dec, v_label, v_idx, v_color = [], [], [], [], [], [], []
    for poly, name, color in zip(polys, names, colors):
        ra_c = np.append(poly[:, 0], poly[0, 0])
        dec_c = np.append(poly[:, 1], poly[0, 1])
        x, y = orthographic(ra_c, dec_c, center_ra, center_dec)
        foot_ras.append(ra_c.tolist())
        foot_decs.append(dec_c.tolist())
        foot_xs.append(x.tolist())
        foot_ys.append(y.tolist())
        vx, vy = orthographic(poly[:, 0], poly[:, 1], center_ra, center_dec)
        for i in range(len(poly)):
            v_x.append(float(vx[i]))
            v_y.append(float(vy[i]))
            v_ra.append(float(poly[i, 0]))
            v_dec.append(float(poly[i, 1]))
            v_label.append(name)
            v_idx.append(i)
            v_color.append(color)
    foot_src = ColumnDataSource(
        {
            "xs": foot_xs,
            "ys": foot_ys,
            "ras": foot_ras,
            "decs": foot_decs,
            "color": colors,
        }
    )
    vertex_src = ColumnDataSource(
        {
            "x": v_x,
            "y": v_y,
            "ra": v_ra,
            "dec": v_dec,
            "label": v_label,
            "idx": v_idx,
            "color": v_color,
        }
    )

    grat_ras, grat_decs = _graticule_raw()
    grat_xs, grat_ys = [], []
    for r, d in zip(grat_ras, grat_decs):
        x, y = orthographic(np.array(r), np.array(d), center_ra, center_dec)
        grat_xs.append(x.tolist())
        grat_ys.append(y.tolist())
    grat_src = ColumnDataSource(
        {"xs": grat_xs, "ys": grat_ys, "ras": grat_ras, "decs": grat_decs}
    )

    if points is not None:
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
        if point_labels is not None and len(point_labels) != len(pts):
            raise ValueError(
                f"point_labels length {len(point_labels)} != number of points {len(pts)}"
            )
        px, py = orthographic(pts[:, 0], pts[:, 1], center_ra, center_dec)
        point_data: dict[str, Any] = {
            "x": px.tolist(),
            "y": py.tolist(),
            "ra": pts[:, 0].tolist(),
            "dec": pts[:, 1].tolist(),
            "color": _resolve_colors(
                point_colors, len(pts), ["#d62728"] * len(pts), "point_colors"
            ),
        }
        if point_labels is not None:
            point_data["label"] = list(point_labels)
    else:
        if point_labels is not None:
            raise ValueError("point_labels given but points is None")
        if point_colors is not None:
            raise ValueError("point_colors given but points is None")
        point_data = {"x": [], "y": [], "ra": [], "dec": [], "color": []}
    point_src = ColumnDataSource(point_data)

    # Explicit Range1d objects so a zoom-recentre callback can be attached. The
    # square figure plus equal symmetric ranges keeps the globe round and
    # centred without match_aspect.
    x_range = Range1d(-1.15, 1.15)
    y_range = Range1d(-1.15, 1.15)
    fig = figure(
        width=size,
        height=size,
        title=title,
        x_range=x_range,
        y_range=y_range,
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    # Drag rotates the globe (via the Pan events below) rather than panning the
    # range, so the globe stays centred.
    fig.toolbar.active_drag = None
    fig.axis.visible = False
    fig.grid.visible = False
    fig.outline_line_color = None

    limb = np.linspace(0.0, 2.0 * np.pi, 241)
    fig.line(
        np.cos(limb).tolist(), np.sin(limb).tolist(), line_color="#bbbbbb", line_width=1
    )
    fig.multi_line("xs", "ys", source=grat_src, line_color="#e3e3e3", line_width=1)
    fig.multi_line("xs", "ys", source=foot_src, line_color="color", line_width=2)
    vertices = fig.scatter(
        "x",
        "y",
        source=vertex_src,
        size=5,
        marker="circle",
        fill_color="color",
        fill_alpha=0.7,
        line_color="color",
    )
    fig.add_tools(
        HoverTool(
            renderers=[vertices],
            tooltips=[
                ("footprint", "@label"),
                ("vertex", "@idx"),
                ("RA, Dec", "@ra{0.000000}, @dec{0.000000}"),
            ],
        )
    )
    markers = fig.scatter(
        "x",
        "y",
        source=point_src,
        size=7,
        marker="circle",
        fill_color="color",
        line_color="black",
        line_width=0.5,
    )
    point_tooltips = [("RA, Dec", "@ra{0.000000}, @dec{0.000000}")]
    if point_labels is not None:
        point_tooltips.insert(0, ("label", "@label"))
    fig.add_tools(HoverTool(renderers=[markers], tooltips=point_tooltips))

    state_src = ColumnDataSource(
        {
            "ra0": [center_ra],
            "dec0": [center_dec],
            "base_ra": [center_ra],
            "base_dec": [center_dec],
            "x0": [0.0],
            "y0": [0.0],
            "zoom_lock": [0],
        }
    )
    fig.js_on_event(PanStart, CustomJS(args={"state": state_src}, code=_PAN_START_JS))
    fig.js_on_event(
        Pan,
        CustomJS(
            args={
                "state": state_src,
                "k": 90.0,
                "foot_src": foot_src,
                "grat_src": grat_src,
                "vertex_src": vertex_src,
                "point_src": point_src,
            },
            code=_PAN_JS,
        ),
    )

    recenter = CustomJS(
        args={"state": state_src, "xr": x_range, "yr": y_range}, code=_RECENTER_JS
    )
    for prop in ("start", "end"):
        x_range.js_on_change(prop, recenter)
        y_range.js_on_change(prop, recenter)

    return display(fig, height=size + 80)
