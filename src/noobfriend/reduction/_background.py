"""Custom per-exposure 2-D background subtraction for reduced frames.

Removes the smooth sky + scattered-light background (e.g. the large-scale glow
left on NIRCam frames) that the official ``jwst`` pipeline does not handle for a
program without dedicated background exposures: ``BackgroundStep`` subtracts
*background exposures* (none here) and ``SkyMatchStep`` only matches a *scalar*
sky level across a mosaic. This estimates a 2-D background from the source-masked
pixels -- a grid of box medians, smoothed and resized back to full resolution
(the photutils ``Background2D`` mechanism, built here on scikit-image) -- and
subtracts it.

Pure ``(data, err, dq)`` arrays in and out, with no ``jwst`` / datamodel /
navigation dependency. ``err`` / ``dq`` pass through: the smooth background
model's own uncertainty is negligible and no pixels are reflagged.
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.transform import resize

from noobfriend.reduction._masking import source_exclusion

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]


def subtract_background(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    *,
    mask: NDArray[np.bool_] | None = None,
    box_size: int = 128,
    filter_size: int = 3,
    nsigma: float = 3.0,
    dilate: int = 5,
    dq_bad_bits: int = 1,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract a source-masked 2-D background model from a frame.

    The image is tiled into ``box_size`` boxes; each box's sigma-robust median of
    the background (non-source, non-flagged, finite) pixels forms a low-resolution
    grid, which is median-smoothed, then bilinearly resized back to full
    resolution and subtracted. Source flux is preserved because source pixels are
    excluded from the grid but the smooth model is subtracted everywhere.

    Parameters
    ----------
    data : numpy.ndarray
        2-D image to background-subtract (e.g. a flux-calibrated ``_cal`` ``SCI``
        array). Not modified in place; a corrected copy is returned.
    err, dq : numpy.ndarray
        Matching error and data-quality arrays, returned unchanged; ``dq`` is read
        to mask bad pixels.
    mask : numpy.ndarray of bool, optional
        Pixels to exclude from the background estimate (``True`` = exclude). When
        ``None`` (default) a source mask is derived from ``data`` and grown by
        ``dilate``.
    box_size : int, default 128
        Side length (pixels) of the background mesh boxes. Smaller follows finer
        structure but risks absorbing extended sources; larger is smoother.
    filter_size : int, default 3
        Side length (in boxes) of the median filter smoothing the low-resolution
        grid; ``1`` disables smoothing.
    nsigma : float, default 3.0
        Source-detection level for the automatic mask (used only when ``mask`` is
        ``None``).
    dilate : int, default 5
        Binary-dilation iterations for the automatic source mask; more generous
        than the 1/f step so source wings do not bias the background.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = ``DO_NOT_USE``).

    Returns
    -------
    data : numpy.ndarray
        The background-subtracted image (a new array).
    err, dq : numpy.ndarray
        The inputs ``err`` and ``dq``, unchanged.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or ``box_size`` is not positive.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1; got {box_size}.")

    excluded = source_exclusion(
        image, dq, mask=mask, nsigma=nsigma, dilate=dilate, dq_bad_bits=dq_bad_bits
    )
    background = _mesh_background(image, excluded, box_size, filter_size)
    return image - background, err, dq


def _mesh_background(
    image: _Floats, excluded: NDArray[np.bool_], box_size: int, filter_size: int
) -> _Floats:
    """Return a smooth 2-D background from a grid of source-masked box medians."""
    masked = np.where(excluded, np.nan, image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN boxes
        grid = block_reduce(
            masked, (box_size, box_size), func=np.nanmedian, cval=np.nan
        )
    grid = _fill_nan_nearest(grid)
    if filter_size > 1:
        grid = ndi.median_filter(grid, size=filter_size, mode="nearest")
    return resize(
        grid,
        image.shape,
        order=1,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )


def _fill_nan_nearest(grid: _Floats) -> _Floats:
    """Replace NaN grid cells (fully-masked boxes) with their nearest finite cell."""
    nan = ~np.isfinite(grid)
    if not nan.any():
        return grid
    if nan.all():
        return np.zeros_like(grid)
    indices = ndi.distance_transform_edt(
        nan, return_distances=False, return_indices=True
    )
    return grid[tuple(indices)]
