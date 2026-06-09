"""Exact fractional coverage of a binary mask reprojected onto another grid.

This is the one geometric primitive the native-grid measurement relies on. The
photometry pipeline never reprojects science *data* — that would resample
surface-brightness values and corrupt both summed flux and per-pixel error
independence. It only ever reprojects binary *masks*, and what it needs from
each output pixel is the exact fraction of its area that the mask covers.

:func:`noobase.image.reproject_exact` is surface-brightness conserving: for a
``0``/``1`` mask it returns, per output pixel, ``image = masked_area /
in_bounds_area`` and ``footprint = in_bounds_area / output_area``. Their product
is therefore ``masked_area / output_area`` — the exact covered-area fraction we
want, robust at field edges and for any relative pixel size or rotation (the
backend is exact polygon clipping, not interpolation). Computing coverage this
way is what lets the measurement integrate the *same sky region* in every band
while each band stays on its own native grid.
"""

from typing import Any

import numpy as np
from noobase.image import make_pixel_corners, reproject_exact

from noobfriend.extraction._wcs import world_detector_transforms
from noobfriend.extraction.photometry._array import native_float64


def reproject_coverage(
    mask: np.ndarray,
    *,
    source_wcs: Any,
    target_wcs: Any,
    target_shape: tuple[int, int],
    coarse_step: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reproject a binary mask onto a target grid as fractional coverage.

    Parameters
    ----------
    mask : numpy.ndarray
        Source boolean (or ``0``/``1``) mask on the source grid.
    source_wcs : Any
        Source mask WCS.
    target_wcs : Any
        Target grid WCS.
    target_shape : tuple[int, int]
        Target ``(rows, cols)`` shape.
    coarse_step : tuple[int, int], optional
        Passed through to :func:`noobase.image.make_pixel_corners` to speed up
        the (numerically expensive) gwcs inverse.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``target_shape`` giving, per target pixel,
        the exact fraction of its area covered by the masked source region, in
        ``[0, 1]``.
    """
    _, target_detector_to_world = world_detector_transforms(target_wcs)
    source_world_to_detector, _ = world_detector_transforms(source_wcs)
    corners = make_pixel_corners(
        target_shape,
        target_pixel_to_world=target_detector_to_world,
        source_world_to_pixel=source_world_to_detector,
        coarse_step=coarse_step,
    )
    result = reproject_exact(
        native_float64(np.asarray(mask, dtype=float)), corners, error=None
    )
    image = np.nan_to_num(result.image, nan=0.0, posinf=0.0, neginf=0.0)
    footprint = np.nan_to_num(result.footprint, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(image * footprint, 0.0, 1.0)
