"""Shared source / bad-pixel exclusion mask for reduction steps.

Several steps (1/f destriping, 2-D background) estimate a level from the
*background* pixels and so must first exclude sources, flagged pixels and
non-finite values. They share this one helper so the masking convention stays
identical across steps; it is internal (no public re-export).
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from noobfriend.core.imgutils import segment


def source_exclusion(
    data: NDArray[np.floating],
    dq: NDArray[np.integer],
    *,
    mask: NDArray[np.bool_] | None = None,
    nsigma: float = 3.0,
    dilate: int = 3,
    dq_bad_bits: int = 1,
    big_px: int = 300,
    wing_scale: float = 2.0,
    wing_max: float = 100.0,
) -> NDArray[np.bool_]:
    """Return the boolean mask of pixels to exclude from a background estimate.

    Combines non-finite pixels and ``dq`` bits in ``dq_bad_bits`` with either the
    caller's ``mask`` or an automatic source mask
    (:func:`~noobfriend.core.imgutils.segment` thresholded at ``nsigma`` and grown
    by ``dilate`` binary dilations). Segments larger than ``big_px`` additionally
    get a circular *wing exclusion* of radius ``wing_scale`` times the segment's
    equivalent radius (capped at ``wing_max``): bright/extended sources carry
    faint wings well below the detection threshold that would otherwise bias a
    row/column or box median at the few-1e-3 level -- too faint for sigma
    clipping (they sit within 1 sigma of the median) yet bright enough to leave
    dark streaks through the source after 1/f subtraction.

    Parameters
    ----------
    data : numpy.ndarray
        2-D image the mask is for.
    dq : numpy.ndarray
        Matching data-quality array.
    mask : numpy.ndarray of bool, optional
        Caller-supplied pixels to exclude (``True`` = exclude). When given, the
        automatic source detection (including the wing tier) is skipped.
    nsigma : float, default 3.0
        Source-detection level for the automatic mask.
    dilate : int, default 3
        Binary-dilation iterations grown around the automatic source mask.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = JWST
        ``DO_NOT_USE``); pass ``0`` to ignore ``dq``.
    big_px : int, default 300
        Area gate for the wing tier: only segments with more than ``big_px``
        pixels get the extra wing exclusion.
    wing_scale : float, default 2.0
        Wing-exclusion radius in units of the segment's equivalent radius
        ``sqrt(area / pi)``, measured from the segment boundary. ``0`` disables
        the wing tier.
    wing_max : float, default 100.0
        Cap on the wing-exclusion radius in pixels.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where a pixel must be excluded from the estimate.
    """
    image = np.asarray(data, dtype=float)
    excluded = ~np.isfinite(image)
    if dq_bad_bits:
        excluded |= (np.asarray(dq) & dq_bad_bits) != 0
    if mask is not None:
        return excluded | np.asarray(mask, dtype=bool)

    sources = segment(np.where(excluded, np.nan, image), nsigma=nsigma, deblend=False)
    source_mask = sources > 0
    if dilate > 0 and source_mask.any():
        source_mask = ndi.binary_dilation(source_mask, iterations=dilate)
    return (
        excluded
        | source_mask
        | _wing_exclusion(
            sources, big_px=big_px, wing_scale=wing_scale, wing_max=wing_max
        )
    )


def _wing_exclusion(
    labels: NDArray[np.integer],
    *,
    big_px: int,
    wing_scale: float,
    wing_max: float,
) -> NDArray[np.bool_]:
    """Circular exclusion around segments larger than ``big_px``.

    Each qualifying segment is grown by ``min(wing_scale * sqrt(area / pi),
    wing_max)`` pixels beyond its boundary. A single Euclidean distance
    transform handles all per-segment radii at once (each pixel is compared
    against the radius of its *nearest* big segment), so the cost is one pass
    regardless of how large the radii are.
    """
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background is not a segment
    big = np.nonzero(counts > big_px)[0]
    if wing_scale <= 0 or big.size == 0:
        return np.zeros(labels.shape, dtype=bool)

    radius = np.zeros(counts.size)
    radius[big] = np.minimum(wing_scale * np.sqrt(counts[big] / np.pi), wing_max)
    distance, (iy, ix) = ndi.distance_transform_edt(
        radius[labels] <= 0, return_indices=True
    )
    return distance <= radius[labels[iy, ix]]
