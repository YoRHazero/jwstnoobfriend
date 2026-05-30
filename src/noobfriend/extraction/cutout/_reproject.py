"""Flux-conserving reprojection of a cutout region, backed by ``noobase``.

This module isolates the dependency on :mod:`noobase.image`. The actual
work — building the corner field and the surface-brightness-conserving polygon
clip — lives in ``noobase``; here we only adapt the call: turn a pair of
WCS callables and an output shape into the corner array ``reproject_exact``
expects.

WCS handling stays entirely in the caller's :mod:`gwcs` / :mod:`astropy`
objects, in keeping with ``noobase``'s design: it never touches spherical
geometry itself.
"""

from typing import Any, Callable

import numpy as np
from noobase.image import make_pixel_corners, reproject_exact


def _as_native_float(data: np.ndarray) -> np.ndarray:
    """Coerce ``data`` to a native-endian, C-contiguous float32/float64 array.

    ``reproject_exact`` requires native-endian ``float32``/``float64``, but JWST
    FITS arrays are typically big-endian (``>f4``). Float width is preserved
    (``float32`` stays ``float32``); non-float inputs are promoted to
    ``float64``. Returns the input untouched when it already satisfies the
    requirement, otherwise a converted copy.
    """
    array = np.asarray(data)
    if array.dtype.kind == "f" and array.dtype.itemsize in (4, 8):
        target = np.dtype(f"=f{array.dtype.itemsize}")
    else:
        target = np.dtype("=f8")
    return np.ascontiguousarray(array, dtype=target)


def reproject_onto(
    data: np.ndarray,
    *,
    target_pixel_to_world: Callable[[Any, Any], Any],
    source_world_to_pixel: Callable[..., tuple[Any, Any]],
    shape: tuple[int, int],
    coarse_step: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject ``data`` onto an output grid via exact polygon clipping.

    Parameters
    ----------
    data : numpy.ndarray
        The source image. Coerced to native-endian ``float32``/``float64`` as
        the kernel requires (JWST big-endian ``>f4`` is handled); non-float
        inputs are promoted to ``float64``. NaNs are treated as masked input
        pixels.
    target_pixel_to_world : callable
        ``f(x, y) -> world`` on the *output* grid, where ``x``/``y`` are arrays
        of output-pixel coordinates (integer = pixel center). For JWST gwcs this
        is the ``detector`` -> ``world`` transform (``get_transform``), not the
        APE-14 ``pixel_to_world_values`` wrapper.
    source_world_to_pixel : callable
        ``f(*world) -> (x_source, y_source)`` mapping world coordinates back to
        ``data``'s pixel grid — the ``world`` -> ``detector`` transform. (JWST
        gwcs does not provide a usable ``world_to_pixel_values``.)
    shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the desired output.
    coarse_step : tuple[int, int], optional
        Evaluate the WCS callables on a coarse subgrid and bicubic-upsample the
        corners; see :func:`noobase.image.make_pixel_corners`. Recommended for
        JWST gwcs, whose numerical inverse is expensive per call. Each component
        must divide the corresponding ``shape`` axis exactly.

    Returns
    -------
    image : numpy.ndarray
        The surface-brightness-conserving reprojection; ``float64`` of ``shape``.
        Output pixels with no valid contribution are ``NaN``.
    footprint : numpy.ndarray
        Geometric overlap fraction in ``[0, 1]`` (independent of input NaNs).
    weight : numpy.ndarray
        Overlap fraction restricted to non-NaN input pixels, ``<= footprint``.
    """
    pixel_corners = make_pixel_corners(
        shape,
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
        coarse_step=coarse_step,
    )
    result = reproject_exact(_as_native_float(data), pixel_corners)
    return result.image, result.footprint, result.weight
