"""Shared frame -> field-grid reprojection for the stage-3 mosaic engine.

The outlier median, the sky matcher and the coadd all reproject one frame onto
a window of a shared :class:`~noobfriend.reduction.mosaic.FieldGrid` via
:func:`noobase.image.reproject_exact` (exact-polygon, surface-brightness
conserving). They share the same three pieces, kept here so there is one copy:

- :func:`as_native_float` -- coerce a (typically big-endian JWST) array to the
  native-endian ``float32`` / ``float64`` the Rust kernel requires;
- :func:`frame_window` -- the grid-pixel bounding window a frame covers
  (edge-sampled, so distortion-bowed edges are enclosed), keeping work and
  memory to a frame-sized sub-window rather than the whole field;
- :func:`reproject_to_window` -- reproject a source (and optionally its error)
  onto that window, returning the image, the valid-overlap weight and the
  propagated error.

This module is internal (leading underscore) and not re-exported; it is imported
by the sibling ``_outlier`` / ``_sky`` / ``_coadd`` modules only.
"""

import math
from typing import Any, Callable

import numpy as np
from noobase.image import make_pixel_corners, reproject_exact

from noobfriend.reduction.mosaic._tiling import FieldGrid

#: Sample points per frame edge when projecting a frame's outline onto the grid
#: (distortion bows the edges, so corners alone under-cover).
_EDGE_SAMPLES: int = 9

#: Extra grid pixels around a frame's projected bounding box.
_WINDOW_MARGIN: int = 8


def as_native_float(data: np.ndarray) -> np.ndarray:
    """Coerce to native-endian, C-contiguous float32/float64 for ``noobase``.

    JWST FITS arrays are typically big-endian (``>f4``); float width is
    preserved and non-float input is promoted to ``float64``.
    """
    array = np.asarray(data)
    if array.dtype.kind == "f" and array.dtype.itemsize in (4, 8):
        target = np.dtype(f"=f{array.dtype.itemsize}")
    else:
        target = np.dtype("=f8")
    return np.ascontiguousarray(array, dtype=target)


def _padded(length: int, step: int) -> int:
    """Round ``length`` up to a multiple of ``step``.

    ``make_pixel_corners`` requires ``coarse_step`` to divide the target shape
    exactly, so target windows are padded up and the result cropped back.
    """
    return int(math.ceil(length / step)) * step


def frame_window(
    shape: tuple[int, int],
    detector_to_world: Callable[..., tuple[Any, Any]],
    grid: FieldGrid,
) -> tuple[int, int, int, int]:
    """Return the grid-pixel window ``(x0, y0, nx, ny)`` covering a frame.

    The frame outline is sampled along each edge (not just the corners --
    distortion bows the edges outward), projected into grid pixels, and the
    bounding box padded by :data:`_WINDOW_MARGIN` and clipped to the grid.

    Parameters
    ----------
    shape : tuple of int
        The frame's ``(ny, nx)`` pixel shape.
    detector_to_world : callable
        The frame's ``(x, y) -> (ra, dec)`` transform (degrees).
    grid : FieldGrid
        The output grid whose pixels the window is expressed in.

    Returns
    -------
    tuple of int
        ``(x0, y0, nx, ny)`` -- the window origin and size in grid pixels
        (``nx``/``ny`` are ``0`` when the frame falls off the grid).
    """
    ny_f, nx_f = shape
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
    xs = np.concatenate(
        [t * (nx_f - 1), np.full_like(t, nx_f - 1), t * (nx_f - 1), np.zeros_like(t)]
    )
    ys = np.concatenate(
        [np.zeros_like(t), t * (ny_f - 1), np.full_like(t, ny_f - 1), t * (ny_f - 1)]
    )
    ra, dec = detector_to_world(xs, ys)
    gx, gy = grid.wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
    ny_g, nx_g = grid.shape
    x0 = max(0, int(np.floor(np.min(gx))) - _WINDOW_MARGIN)
    y0 = max(0, int(np.floor(np.min(gy))) - _WINDOW_MARGIN)
    x1 = min(nx_g, int(np.ceil(np.max(gx))) + _WINDOW_MARGIN + 1)
    y1 = min(ny_g, int(np.ceil(np.max(gy))) + _WINDOW_MARGIN + 1)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def reproject_to_window(
    source: np.ndarray,
    target_shape: tuple[int, int],
    target_pixel_to_world: Callable[..., tuple[Any, Any]],
    source_world_to_pixel: Callable[..., tuple[Any, Any]],
    coarse_step: tuple[int, int] | None,
    *,
    error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reproject ``source`` (and optional ``error``) onto ``target_shape``.

    The target shape is padded up to a ``coarse_step`` multiple (a
    ``make_pixel_corners`` requirement) and the result cropped back; the pad
    rows/columns fall outside the source and cost nothing.

    Parameters
    ----------
    source : numpy.ndarray
        The frame plane to reproject; NaN (e.g. masked/bad) pixels are excluded.
    target_shape : tuple of int
        ``(ny, nx)`` of the output window.
    target_pixel_to_world : callable
        Output-window ``(x, y) -> (ra, dec)``.
    source_world_to_pixel : callable
        ``(ra, dec) -> (x, y)`` of the source frame.
    coarse_step : tuple of int or None
        Coarse-grid WCS evaluation stride for :func:`make_pixel_corners`
        (interpolated corners); ``None`` evaluates every corner exactly.
    error : numpy.ndarray, optional
        The frame's error plane; when given, its reprojection is returned too.

    Returns
    -------
    image : numpy.ndarray
        The surface-brightness-conserving reprojection; NaN where nothing valid
        contributes.
    weight : numpy.ndarray
        Valid-pixel overlap fraction (NaN/masked source pixels excluded).
    error : numpy.ndarray or None
        The reprojected error, or ``None`` when no ``error`` was supplied.
    """
    ny, nx = target_shape
    if coarse_step is not None:
        shape = (_padded(ny, coarse_step[0]), _padded(nx, coarse_step[1]))
    else:
        shape = (ny, nx)
    corners = make_pixel_corners(
        shape,
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
        coarse_step=coarse_step,
    )
    result = reproject_exact(
        as_native_float(source),
        corners,
        error=None if error is None else as_native_float(error),
    )
    out_error = None if result.error is None else result.error[:ny, :nx]
    return result.image[:ny, :nx], result.weight[:ny, :nx], out_error
