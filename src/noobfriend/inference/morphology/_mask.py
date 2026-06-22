"""Auto-derive a per-band fit mask that excludes contaminating neighbours.

The contamination model is a clean binary: sources you care about are full target
components, and everything else is masked. This builds the "everything else" mask
for one band by segmenting it on its own native grid, keeping the segment(s) at the
target position (plus the sky), and excluding -- dilated, to catch their wings --
every other source's segment. A target that is undetected yields an all-clear mask
rather than risking masking it.

(A union across bands, projecting each band's neighbour detections into the others
to avoid shallow-band under-detection, is the intended refinement; today this is
per-band.) This module is internal.
"""

from __future__ import annotations

import numpy as np


def auto_mask_band(
    data: np.ndarray,
    target_xy: tuple[float, float],
    *,
    nsigma: float = 2.0,
    dilate: int = 2,
) -> np.ndarray:
    """Return a boolean exclude-mask (``True`` = not part of the target) for a band.

    Parameters
    ----------
    data : numpy.ndarray
        2-D science image on the band's native grid.
    target_xy : tuple of float
        Target position in native ``(x, y)`` pixel coordinates; the segment there
        is kept.
    nsigma : float, default 2.0
        Detection threshold for :func:`noobfriend.core.imgutils.segment`.
    dilate : int, default 2
        Binary-dilation iterations grown around neighbour segments (to mask wings).

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where a pixel belongs to a *non-target* source.
    """
    from noobfriend.core.imgutils import segment

    labels = segment(data, nsigma=nsigma)
    rows, cols = data.shape
    tx, ty = int(round(target_xy[0])), int(round(target_xy[1]))
    if not (0 <= tx < cols and 0 <= ty < rows):
        return np.zeros(data.shape, dtype=bool)

    target_label = int(labels[ty, tx])
    if target_label == 0:
        # Target undetected; do not risk masking it (and neighbours are unlikely).
        return np.zeros(data.shape, dtype=bool)

    other = (labels != 0) & (labels != target_label)
    if dilate > 0 and other.any():
        from scipy import ndimage as ndi

        other = ndi.binary_dilation(other, iterations=int(dilate))
    return other
