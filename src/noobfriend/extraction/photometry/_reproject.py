"""Photometry-local reprojection helpers.

This module intentionally does not import cutout's private reprojection helper.
Photometry aligns whole band images and aperture masks onto a reference band
grid, which is a separate workflow from extracting a rectangular cutout.
"""

from typing import Any

import numpy as np
from noobase.image import make_pixel_corners, reproject_exact

from noobfriend.extraction._wcs import world_detector_transforms
from noobfriend.extraction.photometry._array import native_float64


def reproject_image(
    data: np.ndarray,
    *,
    source_wcs: Any,
    target_wcs: Any,
    target_shape: tuple[int, int],
    error: np.ndarray | None = None,
    coarse_step: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Reproject an image onto a target band grid.

    Parameters
    ----------
    data : numpy.ndarray
        Source image. Must represent surface brightness or another quantity for
        which area-weighted reprojection is meaningful.
    source_wcs : Any
        Source image WCS.
    target_wcs : Any
        Target reference WCS.
    target_shape : tuple[int, int]
        Target ``(rows, cols)`` shape.
    error : numpy.ndarray, optional
        Source 1-sigma error image.
    coarse_step : tuple[int, int], optional
        Passed through to :func:`noobase.image.make_pixel_corners`.

    Returns
    -------
    image, error, footprint, weight : tuple
        Reprojected image, propagated error (or ``None``), geometric footprint,
        and valid-data weight.
    """
    _, target_detector_to_world = world_detector_transforms(target_wcs)
    source_world_to_detector, _ = world_detector_transforms(source_wcs)
    corners = make_pixel_corners(
        target_shape,
        target_pixel_to_world=target_detector_to_world,
        source_world_to_pixel=source_world_to_detector,
        coarse_step=coarse_step,
    )
    error_in = None if error is None else native_float64(error)
    result = reproject_exact(native_float64(data), corners, error=error_in)
    return result.image, result.error, result.footprint, result.weight


def reproject_mask(
    mask: np.ndarray,
    *,
    source_wcs: Any,
    target_wcs: Any,
    target_shape: tuple[int, int],
    threshold: float = 0.5,
    coarse_step: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a boolean mask and threshold its fractional coverage.

    Parameters
    ----------
    mask : numpy.ndarray
        Source boolean mask.
    source_wcs : Any
        Source mask WCS.
    target_wcs : Any
        Target reference WCS.
    target_shape : tuple[int, int]
        Target ``(rows, cols)`` shape.
    threshold : float, default 0.5
        Target pixels with reprojected mask fraction greater than or equal to
        this value are considered inside the mask.
    coarse_step : tuple[int, int], optional
        Passed through to :func:`noobase.image.make_pixel_corners`.

    Returns
    -------
    mask, fraction : tuple[numpy.ndarray, numpy.ndarray]
        Thresholded target-grid mask and the fractional coverage image.
    """
    fraction, _, _, _ = reproject_image(
        np.asarray(mask, dtype=np.float64),
        source_wcs=source_wcs,
        target_wcs=target_wcs,
        target_shape=target_shape,
        error=None,
        coarse_step=coarse_step,
    )
    fraction = np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)
    return fraction >= threshold, fraction
