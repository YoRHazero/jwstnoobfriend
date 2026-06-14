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
) -> NDArray[np.bool_]:
    """Return the boolean mask of pixels to exclude from a background estimate.

    Combines non-finite pixels and ``dq`` bits in ``dq_bad_bits`` with either the
    caller's ``mask`` or an automatic source mask
    (:func:`~noobfriend.core.imgutils.segment` thresholded at ``nsigma`` and grown
    by ``dilate`` binary dilations).

    Parameters
    ----------
    data : numpy.ndarray
        2-D image the mask is for.
    dq : numpy.ndarray
        Matching data-quality array.
    mask : numpy.ndarray of bool, optional
        Caller-supplied pixels to exclude (``True`` = exclude). When given, the
        automatic source detection is skipped.
    nsigma : float, default 3.0
        Source-detection level for the automatic mask.
    dilate : int, default 3
        Binary-dilation iterations grown around the automatic source mask.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = JWST
        ``DO_NOT_USE``); pass ``0`` to ignore ``dq``.

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
    return excluded | source_mask
