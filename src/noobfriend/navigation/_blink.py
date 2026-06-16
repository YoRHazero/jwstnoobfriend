"""Shared frame adapter bridging NooBooks to the core blink comparator.

Both :attr:`NooBook.viz <noobfriend.navigation.noobook._core.NooBook.viz>` and
:attr:`NooBox.viz <noobfriend.navigation.noobox._core.NooBox.viz>` expose an
``imshow_blink`` that overlays several frames and flips between them. The one
thing they share -- turning a sequence of :class:`~noobfriend.navigation.NooBook`
and/or raw arrays into the ``(images, labels)`` that
:func:`noobfriend.core.display.plot.imshow_blink` consumes -- lives here, so
neither accessor duplicates it.

The core function draws every frame on a single shared pixel grid: same pixel
scale, placement by translation only (per-frame ``offsets``), no rotation (see
its docstring). The frames handed in are therefore expected to be co-gridded;
``offsets`` and all styling options pass straight through. Passing
``align="wcs"`` derives those offsets from the books' GWCS so co-gridded imaging
frames (e.g. a pointing's dithers) line up on the sky.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from noobfriend.navigation.noobook import NooBook

#: A 2-D coordinate transform ``f(a, b) -> (c, d)`` -- e.g. a GWCS
#: world<->detector leg from :meth:`gwcs.WCS.get_transform`.
_Transform = Callable[[Any, Any], tuple[Any, Any]]


def _resolve_frames(
    frames: Sequence[NooBook | np.ndarray], labels: Sequence[str] | None
) -> tuple[list[np.ndarray], list[str]]:
    """Reduce mixed book/array frames to ``(images, labels)``.

    Parameters
    ----------
    frames : sequence of NooBook or numpy.ndarray
        Frames to overlay, in draw order. A :class:`~noobfriend.navigation.NooBook`
        contributes its ``SCI`` array (:attr:`~noobfriend.navigation.NooBook.data`);
        a :class:`numpy.ndarray` is taken as-is.
    labels : sequence of str or None
        Explicit per-frame labels. When ``None``, each frame is labelled by its
        book id, or by its index for a raw array.

    Returns
    -------
    images : list of numpy.ndarray
        One array per frame, in input order.
    labels : list of str
        One label per frame, same length as ``images``.

    Raises
    ------
    ValueError
        If ``labels`` is given but its length does not match ``frames``.
    """
    images = [f.data if isinstance(f, NooBook) else np.asarray(f) for f in frames]
    if labels is None:
        out_labels = [
            f.id if isinstance(f, NooBook) else str(k) for k, f in enumerate(frames)
        ]
    else:
        out_labels = [str(s) for s in labels]
        if len(out_labels) != len(images):
            raise ValueError(
                f"labels has length {len(out_labels)}, expected {len(images)}"
            )
    return images, out_labels


def _translation_offset(
    anchor_world_to_detector: _Transform,
    frame_detector_to_world: _Transform,
    shape: tuple[int, int],
    atol: float,
) -> tuple[float, float]:
    """Pixel ``(dx, dy)`` placing a frame onto the anchor's grid by translation.

    Maps each of the frame's four corner pixels to world coordinates and back to
    anchor pixels; the per-corner implied shift ``(x_anchor - x, y_anchor - y)``
    is constant only when the two grids differ by a pure translation. The corners
    must agree within ``atol`` -- so a relative rotation or scale is rejected
    rather than silently mis-placed -- and their mean is returned. The half-pixel
    between a Bokeh cell's lower-left placement and a pixel-centre coordinate
    cancels in the difference, so none is applied.

    Parameters
    ----------
    anchor_world_to_detector : callable
        Anchor frame's ``(ra, dec) -> (x, y)`` transform.
    frame_detector_to_world : callable
        This frame's ``(x, y) -> (ra, dec)`` transform.
    shape : tuple of int
        The frame's array shape ``(n_rows, n_cols)``.
    atol : float
        Maximum allowed spread, in pixels, of the four corners' implied offsets.

    Returns
    -------
    tuple of float
        The ``(dx, dy)`` lower-left placement for this frame.

    Raises
    ------
    ValueError
        If the corners disagree by more than ``atol`` (the frames differ by more
        than a translation).
    """
    n_rows, n_cols = shape
    corners = [(0, 0), (n_cols - 1, 0), (0, n_rows - 1), (n_cols - 1, n_rows - 1)]
    implied = []
    for x, y in corners:
        ra, dec = frame_detector_to_world(x, y)
        xa, ya = anchor_world_to_detector(ra, dec)
        implied.append((float(xa) - x, float(ya) - y))
    arr = np.asarray(implied, dtype=float)
    offset = arr.mean(axis=0)
    residual = float(np.abs(arr - offset).max())
    if residual > atol:
        raise ValueError(
            f"frames differ by more than a translation (corner residual "
            f"{residual:.3g} px > atol {atol}); blink is translation-only, so "
            "reproject to a common grid or pass explicit offsets"
        )
    return float(offset[0]), float(offset[1])


def _wcs_offsets(
    frames: Sequence[NooBook | np.ndarray],
    images: Sequence[np.ndarray],
    atol: float,
) -> list[tuple[float, float]]:
    """Per-frame WCS-derived translation offsets, anchored on the first frame.

    Builds the world<->detector transforms straight from each book's GWCS (the
    plain :meth:`gwcs.WCS.get_transform` legs -- imaging products only) and
    returns the ``(dx, dy)`` placing every frame onto the first frame's pixel
    grid (see :func:`_translation_offset`).

    Parameters
    ----------
    frames : sequence of NooBook or numpy.ndarray
        The frames being blinked; every one must be a :class:`NooBook` carrying
        an assigned, non-grism WCS.
    images : sequence of numpy.ndarray
        The frames' resolved arrays (for their shapes), parallel to ``frames``.
    atol : float
        Corner-agreement tolerance forwarded to :func:`_translation_offset`.

    Returns
    -------
    list of tuple of float
        One ``(dx, dy)`` per frame; the first (anchor) is ``(0.0, 0.0)``.

    Raises
    ------
    ValueError
        If any frame is not a NooBook, lacks a WCS, or is a WFSS grism product
        (blink is imaging-only -- align grism by passing explicit offsets).
    """
    wcss = []
    for f in frames:
        if not isinstance(f, NooBook):
            raise ValueError("align='wcs' requires every frame to be a NooBook")
        wcs = f.wcs
        if wcs is None:
            raise ValueError(f"align='wcs': book {f.id!r} has no assigned WCS")
        if "grism_detector" in wcs.available_frames:
            raise ValueError(
                f"align='wcs': book {f.id!r} is a WFSS grism product; blink is "
                "imaging-only -- pass explicit offsets to align grism frames"
            )
        wcss.append(wcs)
    anchor_world_to_detector = wcss[0].get_transform("world", "detector")
    return [
        _translation_offset(
            anchor_world_to_detector,
            wcs.get_transform("detector", "world"),
            images[k].shape,
            atol,
        )
        for k, wcs in enumerate(wcss)
    ]


def blink_frames(
    frames: Sequence[NooBook | np.ndarray],
    *,
    labels: Sequence[str] | None = None,
    align: Literal["wcs"] | None = None,
    atol: float = 1.0,
    offsets: Sequence[tuple[float, float]] | None = None,
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
    """Resolve book/array frames and hand them to the core blink comparator.

    Parameters
    ----------
    frames : sequence of NooBook or numpy.ndarray
        Frames to overlay, in draw order (see :func:`_resolve_frames`).
    labels : sequence of str, optional
        Per-frame button labels; defaults to each book's id, or the frame's
        index for a raw array.
    align : {"wcs", None}, optional
        When ``"wcs"``, derive each frame's pixel ``offsets`` from its GWCS so
        co-gridded imaging frames (e.g. a pointing's dithers) line up on the sky;
        every frame must then be a NooBook with a non-grism WCS, and passing
        ``offsets`` as well is an error. When ``None`` (default), ``offsets`` is
        used directly, defaulting to zero for every frame. Grism frames are
        imaging-only's deliberate gap -- align them by passing explicit
        ``offsets``.
    atol : float, optional
        Only with ``align="wcs"``: the maximum spread, in pixels, allowed between
        a frame's four corners' implied offsets before the frames are judged to
        differ by more than a translation (and rejected). Default ``1.0``.
    offsets : sequence of (float, float), optional
        Per-frame ``(dx, dy)`` lower-left placement in pixels. Mutually
        exclusive with ``align="wcs"``. Defaults to all frames stacked at a
        common origin.
    vmin, vmax : float or sequence of float, optional
        Color-stretch limits. A scalar applies to every frame; a sequence gives
        one limit per frame. A side left unset falls back to ``pmin``/``pmax``.
    pmin, pmax : float, default 1.0, 99.0
        Percentile cuts used per frame for whichever of ``vmin``/``vmax`` is not
        given.
    cmap : str, default ``"Greys"``
        Colormap shared by all frames.
    stretch : str, default ``"linear"``
        Intensity stretch shared by all frames.
    size : int, default 620
        Length of the longer figure edge, in pixels.
    title : str, optional
        Figure title.
    blink : bool, default True
        Add a play/pause button that auto-cycles through the frames.

    Returns
    -------
    Any
        The display handle returned by the core ``imshow_blink``.

    Raises
    ------
    ValueError
        If ``align="wcs"`` but ``offsets`` is also given, a frame is not a
        WCS-carrying imaging NooBook, or the frames differ by more than a
        translation; or for an unknown ``align`` value.
    """
    images, resolved_labels = _resolve_frames(frames, labels)
    if align == "wcs":
        if offsets is not None:
            raise ValueError("pass either align='wcs' or offsets, not both")
        resolved_offsets = _wcs_offsets(frames, images, atol)
    elif align is not None:
        raise ValueError(f"align must be 'wcs' or None, got {align!r}")
    else:
        resolved_offsets = offsets

    from noobfriend.core.display.plot import imshow_blink

    return imshow_blink(
        images,
        offsets=resolved_offsets,
        labels=resolved_labels,
        vmin=vmin,
        vmax=vmax,
        pmin=pmin,
        pmax=pmax,
        cmap=cmap,
        stretch=stretch,
        size=size,
        title=title,
        blink=blink,
    )
