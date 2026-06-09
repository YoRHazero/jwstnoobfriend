"""Switchable multi-image viewer: a Bokeh "blink comparator" for notebooks.

The public entry point :func:`imshow_blink` overlays several 2-D arrays at their
own pixel offsets and renders one segmented control to flip between them, plus an
optional auto-blink. Every frame is drawn up front as its own Bokeh ``image``
glyph (own color mapper, own placement); switching only toggles glyph
``visible`` on the client, which sidesteps the BokehJS pixel-cache pitfalls of
swapping a glyph's ``color_mapper`` or mutating its image data, and keeps the
flip instant with no kernel round-trip.

The figure's ranges are fixed to the union of all placed frames, so blinking
never reframes the view and pan/zoom carry across frames -- exactly what a blink
comparator wants.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import CustomJS, HoverTool, RadioButtonGroup, Range1d, Select, Toggle
from bokeh.plotting import figure

from noobfriend.core.display.plot._bokeh import display
from noobfriend.core.display.plot._image import CMAPS, STRETCHES, _make_mapper
from noobfriend.core.display.plot._norm import resolve_limits

#: Auto-blink speeds, in milliseconds per frame, offered by the speed selector.
_BLINK_SPEEDS_MS: tuple[str, ...] = ("250", "500", "1000", "2000")
_BLINK_DEFAULT_MS: str = "500"

#: Shared pixel height for the blink button and speed selector so they align in
#: their row (the ``Select`` carries no title, so both are bare boxes).
_CONTROL_HEIGHT: int = 32


def _broadcast_limits(
    v: float | Sequence[float] | None, n: int, name: str
) -> list[float | None]:
    """Broadcast a ``vmin``/``vmax`` spec to one value per image.

    Parameters
    ----------
    v : float, sequence of float, or None
        A scalar (applied to every image), a sequence (one limit per image), or
        ``None`` (no explicit limit for any image -- each falls back to its
        percentile in :func:`~noobfriend.core.display.plot._norm.resolve_limits`).
    n : int
        Number of images.
    name : str
        ``"vmin"`` or ``"vmax"``, used only in the error message.

    Returns
    -------
    list
        Length-``n`` list of ``float`` (or ``None``) limits.

    Raises
    ------
    ValueError
        If ``v`` is a sequence whose length is not ``n``.
    """
    if v is None:
        return [None] * n
    if np.ndim(v) == 0:
        return [float(v)] * n  # type: ignore[arg-type]
    seq = list(v)  # type: ignore[arg-type]
    if len(seq) != n:
        raise ValueError(f"{name} has length {len(seq)}, expected {n} (one per image)")
    return [float(x) for x in seq]


def _union_bounds(
    images: Sequence[np.ndarray], offsets: Sequence[tuple[float, float]]
) -> tuple[float, float, float, float]:
    """Return ``(x0, x1, y0, y1)`` covering every placed image.

    Each image is drawn origin-at-lower-left at its offset, spanning
    ``[ox, ox + n_cols] x [oy, oy + n_rows]`` in data (= pixel) coordinates.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        The 2-D frames.
    offsets : sequence of (float, float)
        Per-image ``(dx, dy)`` lower-left placement, same length as ``images``.

    Returns
    -------
    x0, x1, y0, y1 : float
        Axis-aligned bounds of the union of all frames.
    """
    x0 = min(ox for ox, _ in offsets)
    y0 = min(oy for _, oy in offsets)
    x1 = max(ox + img.shape[1] for (ox, _), img in zip(offsets, images))
    y1 = max(oy + img.shape[0] for (_, oy), img in zip(offsets, images))
    return float(x0), float(x1), float(y0), float(y1)


#: Toggle the active frame's visibility and steer the hover to it. Bound to the
#: RadioButtonGroup ``active`` change, so both a manual click and an auto-blink
#: tick (which advances ``active``) flow through the same path.
_SWITCH_JS = r"""
const active = cb_obj.active;
for (let k = 0; k < renderers.length; k++) {
    renderers[k].visible = (k === active);
}
hover.renderers = [renderers[active]];
"""

#: Start/stop the auto-blink interval. Timers are keyed by the toggle's id in a
#: per-window registry so multiple stacks (widget mode shares one window) and
#: speed changes don't cross-talk.
_BLINK_JS = r"""
const reg = (window.__nbf_blink = window.__nbf_blink || new Map());
const key = play.id;
if (reg.has(key)) { clearInterval(reg.get(key)); reg.delete(key); }
if (play.active) {
    play.label = "⏸ Blink";
    const id = setInterval(() => { group.active = (group.active + 1) % n; },
                           parseInt(speed.value));
    reg.set(key, id);
} else {
    play.label = "▶ Blink";
}
"""


def imshow_blink(
    images: Sequence[np.ndarray],
    *,
    offsets: Sequence[tuple[float, float]] | None = None,
    labels: Sequence[str] | None = None,
    vmin: float | Sequence[float] | None = None,
    vmax: float | Sequence[float] | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    cmap: str = "Greys",
    stretch: str = "linear",
    size: int = 620,
    title: str | None = None,
    blink: bool = True,
) -> Any:
    """Display several 2-D arrays with a control to flip between them.

    Each frame is drawn as its own Bokeh ``image`` glyph at its pixel offset and
    with its own color stretch; a segmented control (one button per frame) flips
    which one is visible, entirely client-side. Frames keep the FITS
    origin-at-lower-left convention, so data coordinates equal pixel indices and
    ``offsets`` shift a frame by ``(dx, dy)`` pixels. The view is fixed to the
    union of all frames so flipping never reframes it.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        Two or more 2-D frames. They need not share a shape. Each is cast to
        ``float32`` before upload; note all frames are embedded, so the payload
        scales with the number of frames.
    offsets : sequence of (float, float), optional
        Per-frame ``(dx, dy)`` lower-left placement in pixels. Defaults to all
        ``(0, 0)`` (frames stacked at a common origin). Length must match
        ``images``.
    labels : sequence of str, optional
        Button label for each frame. Defaults to ``"0", "1", ...``. Length must
        match ``images``.
    vmin, vmax : float or sequence of float, optional
        Color-stretch limits. A scalar applies to every frame; a sequence gives
        one limit per frame (length must match ``images``). A side left unset
        falls back to the ``pmin``/``pmax`` percentile of that frame.
    pmin, pmax : float, default 1.0, 99.0
        Percentile cuts used per frame for whichever of ``vmin``/``vmax`` is not
        given.
    cmap : str, default ``"Greys"``
        Colormap shared by all frames; one of the keys of
        :data:`~noobfriend.core.display.plot._image.CMAPS`.
    stretch : str, default ``"linear"``
        Stretch shared by all frames; one of
        :data:`~noobfriend.core.display.plot._image.STRETCHES`.
    size : int, default 620
        Length of the longer figure edge, in pixels; the other edge follows the
        aspect ratio of the union of all frames.
    title : str, optional
        Figure title.
    blink : bool, default True
        Add a play/pause button (and a speed selector) that auto-cycles through
        the frames client-side. Ignored when only one frame is given.

    Returns
    -------
    Any
        A display handle (see
        :func:`~noobfriend.core.display.plot.set_render_mode`), rendering
        automatically as a notebook cell's last expression.

    Raises
    ------
    ValueError
        If ``images`` is empty, any frame is not 2-D, ``cmap``/``stretch`` is
        unknown, a resolved ``vmin > vmax``, or ``offsets``/``labels``/a
        sequence ``vmin``/``vmax`` has a length other than ``len(images)``.
    """
    if cmap not in CMAPS:
        raise ValueError(f"cmap must be one of {list(CMAPS)}, got {cmap!r}")
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {list(STRETCHES)}, got {stretch!r}")

    arrs = [np.asarray(im) for im in images]
    n = len(arrs)
    if n == 0:
        raise ValueError("imshow_blink requires at least one image")
    for k, a in enumerate(arrs):
        if a.ndim != 2:
            raise ValueError(f"images[{k}] must be 2-D, got shape {a.shape}")
    arrs = [a.astype(np.float32, copy=False) for a in arrs]

    if offsets is None:
        offs: list[tuple[float, float]] = [(0.0, 0.0)] * n
    else:
        offs = [(float(ox), float(oy)) for ox, oy in offsets]
        if len(offs) != n:
            raise ValueError(f"offsets has length {len(offs)}, expected {n}")
    if labels is None:
        labels_ = [str(k) for k in range(n)]
    else:
        labels_ = [str(s) for s in labels]
        if len(labels_) != n:
            raise ValueError(f"labels has length {len(labels_)}, expected {n}")

    vmins = _broadcast_limits(vmin, n, "vmin")
    vmaxs = _broadcast_limits(vmax, n, "vmax")
    limits = [resolve_limits(arrs[k], vmins[k], vmaxs[k], pmin, pmax) for k in range(n)]

    # Size the figure to the union's aspect so frames fill it with square pixels.
    x0, x1, y0, y1 = _union_bounds(arrs, offs)
    span_x, span_y = x1 - x0, y1 - y0
    if span_x >= span_y:
        fig_w, fig_h = size, max(1, round(size * span_y / span_x))
    else:
        fig_w, fig_h = max(1, round(size * span_x / span_y)), size

    fig = figure(
        width=fig_w,
        height=fig_h,
        title=title or "",
        x_range=Range1d(x0, x1),
        y_range=Range1d(y0, y1),
        match_aspect=True,
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    renderers = []
    for k, a in enumerate(arrs):
        n_rows, n_cols = a.shape
        ox, oy = offs[k]
        low, high = limits[k]
        mapper = _make_mapper(stretch, CMAPS[cmap], low, high)
        renderers.append(
            fig.image(
                image=[a],
                x=ox,
                y=oy,
                dw=n_cols,
                dh=n_rows,
                color_mapper=mapper,
                visible=(k == 0),
            )
        )

    hover = HoverTool(
        renderers=[renderers[0]],
        tooltips=[("(x, y)", "$x{0.0}, $y{0.0}"), ("value", "@image")],
    )
    fig.add_tools(hover)

    if n == 1:
        return display(fig, height=fig_h + 80)

    group = RadioButtonGroup(labels=labels_, active=0)
    group.js_on_change(
        "active",
        CustomJS(args={"renderers": renderers, "hover": hover}, code=_SWITCH_JS),
    )
    if not blink:
        # Segmented control on its own row above the figure.
        return display(column(group, fig), height=fig_h + 110)

    # Second row: play/pause toggle + speed selector, equal-height so they align.
    play = Toggle(label="▶ Blink", active=False, width=90, height=_CONTROL_HEIGHT)
    speed = Select(
        value=_BLINK_DEFAULT_MS,
        options=[(ms, f"{ms} ms") for ms in _BLINK_SPEEDS_MS],
        width=110,
        height=_CONTROL_HEIGHT,
    )
    blink_js = CustomJS(
        args={"group": group, "play": play, "speed": speed, "n": n}, code=_BLINK_JS
    )
    play.js_on_change("active", blink_js)
    speed.js_on_change("value", blink_js)

    layout = column(group, row(play, speed, spacing=8), fig)
    return display(layout, height=fig_h + 170)
