"""Trace-aware source mask for NIRCam WFSS (grism) frames.

A plain ``nsigma`` threshold under-masks the dispersed spectral traces (their
continuum sits near the noise once the sky gradient is removed), so the 1/f /
background steps that estimate a level from *background* pixels would be biased
by trace flux. This builds a mask that follows the traces by detecting on a
**sky-flattened** frame and growing detections *along the dispersion*.

All-scikit-image: a coarse block-median sky model (:func:`skimage.measure.block_reduce`
+ :func:`skimage.transform.resize`) flattens the dispersed-sky gradient so the
threshold is uniform; detections are grown with a horizontal
:func:`skimage.morphology.footprint_rectangle`; specks are dropped by
connected-component size (:func:`skimage.measure.label`). Ridge filters
(``sato``) and Otsu/Li thresholds were tried and over-mask badly -- they latch
onto the textured sky gradient, not the traces -- so flatten-then-detect is used.
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from skimage.measure import block_reduce, label
from skimage.morphology import dilation, footprint_rectangle
from skimage.transform import resize

_Floats = NDArray[np.floating]
_Bool = NDArray[np.bool_]


def grism_trace_mask(
    data: _Floats,
    dq: NDArray[np.integer],
    *,
    nsigma: float = 4.0,
    block: int = 64,
    iters: int = 2,
    min_size: int = 6,
    grow_x: int = 9,
    grow_y: int = 2,
    dq_bad_bits: int = 1,
) -> _Bool:
    """Return the boolean exclusion mask of dispersed traces, sources and bad pixels.

    The dispersed-sky gradient is removed with a source-excluded coarse block
    median (refined for ``iters`` passes), detections are made at ``nsigma`` over
    a MAD noise on the flat residual, then grown ``grow_x`` along the dispersion
    (horizontal, GRISMR) and ``grow_y`` across it before small components are
    dropped. Non-finite and ``dq``-flagged pixels are always excluded.

    Parameters
    ----------
    data : numpy.ndarray
        2-D grism frame (a ``_rate`` or post-flat ``SCI`` array).
    dq : numpy.ndarray
        Matching data-quality array.
    nsigma : float, default 4.0
        Detection level (MAD sigmas) on the sky-flattened residual. ``4`` tracks
        the real traces at ~8% coverage; lower starts catching noise speckle.
    block : int, default 64
        Side length (pixels) of the coarse-sky mesh boxes used to flatten the
        dispersed-sky gradient before detection.
    iters : int, default 2
        Number of flatten/detect passes; each refines the sky model by excluding
        the current detections.
    min_size : int, default 6
        Minimum connected-component size (pixels) kept after growing.
    grow_x, grow_y : int, default 9, 2
        Half-widths of the rectangular growth footprint along (``x``, dispersion)
        and across (``y``) the traces.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = ``DO_NOT_USE``).

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where a pixel must be excluded (trace, source or bad).

    Raises
    ------
    ValueError
        If ``data`` is not 2-D.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")

    bad = ~np.isfinite(image)
    if dq_bad_bits:
        bad |= (np.asarray(dq) & dq_bad_bits) != 0

    detected = np.zeros(image.shape, dtype=bool)
    flattened = image
    for _ in range(max(1, iters)):
        flattened = image - _coarse_sky(image, bad | detected, block)
        sigma = _robust_sigma(np.where(bad | detected, np.nan, flattened))
        detected = np.isfinite(flattened) & (flattened > nsigma * sigma)

    grown = dilation(detected, footprint_rectangle((2 * grow_y + 1, 2 * grow_x + 1)))
    return _drop_small(grown, min_size) | bad


def _robust_sigma(values: _Floats) -> float:
    """Return the MAD-based standard-deviation estimate over the finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(1.4826 * mad)


def _coarse_sky(data: _Floats, exclude: _Bool, block: int) -> _Floats:
    """Return a smooth dispersed-sky model: block medians resized to full res."""
    masked = np.where(exclude, np.nan, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN blocks
        grid = block_reduce(masked, (block, block), func=np.nanmedian)
    finite = grid[np.isfinite(grid)]
    grid = np.nan_to_num(grid, nan=float(np.median(finite)) if finite.size else 0.0)
    return resize(
        grid, data.shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True
    )


def _drop_small(mask: _Bool, min_size: int) -> _Bool:
    """Remove connected components (8-connectivity) smaller than ``min_size``."""
    labels = label(mask, connectivity=2)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]
