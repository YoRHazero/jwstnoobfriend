"""Turn an image into an integer label map (thin scikit-image wrapper).

Mechanism only: a robust detection threshold, connected-component labeling, and
optional distance-transform watershed deblending, all from scikit-image /
scipy. Everything here is **pixel space** and carries no astronomical
interpretation — mapping labels to sky positions or calibrated fluxes is the
caller's job. That domain-neutrality is exactly why this lives in ``core``
rather than a science module; if it ever grows WCS- or photometry-aware
behaviour, that version graduates out of ``core``.
"""

from typing import Literal

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label as _connected_components
from skimage.segmentation import watershed

_MAD_TO_SIGMA = 1.4826
#: Map user-facing 4-/8-connectivity to scikit-image's neighbourhood rank.
_CONNECTIVITY_RANK: dict[int, int] = {4: 1, 8: 2}


def segment(
    data: np.ndarray,
    *,
    threshold: float | None = None,
    nsigma: float = 2.0,
    min_size: int = 5,
    deblend: bool = True,
    connectivity: Literal[4, 8] = 8,
) -> np.ndarray:
    """Return an integer label map of sources in ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        2-D image. ``NaN`` pixels are treated as background.
    threshold : float, optional
        Absolute detection threshold. When ``None`` (default) it is derived from
        the image noise as ``median + nsigma * sigma``, with ``sigma`` estimated
        robustly from the median absolute deviation of the finite pixels.
    nsigma : float, default 2.0
        Detection level in robust sigmas above the median, used only when
        ``threshold`` is ``None``.
    min_size : int, default 5
        Connected regions smaller than this many pixels are discarded.
    deblend : bool, default True
        When ``True``, split touching regions with a distance-transform
        watershed; when ``False``, each connected region becomes one label.
    connectivity : {4, 8}, default 8
        Pixel connectivity: ``4`` joins only face-adjacent pixels, ``8`` also
        joins diagonally-adjacent ones.

    Returns
    -------
    numpy.ndarray
        Integer label map the same shape as ``data``: ``0`` is background and
        each source has a distinct positive label.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or ``connectivity`` is not 4 or 8.

    Notes
    -----
    Watershed deblending is driven by local maxima of the distance transform, so
    a single but irregular source can occasionally be split. Pass
    ``deblend=False`` when only connected-component labelling is wanted.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity!r}.")
    rank = _CONNECTIVITY_RANK[connectivity]

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.intp)

    level = (
        _noise_threshold(image[finite], nsigma)
        if threshold is None
        else float(threshold)
    )
    mask = finite & (image > level)
    if min_size > 1:
        mask = _drop_small(mask, min_size, rank)
    if not mask.any():
        return np.zeros(image.shape, dtype=np.intp)

    if not deblend:
        return _connected_components(mask, connectivity=rank).astype(np.intp)
    return _watershed_deblend(mask, rank)


def _noise_threshold(values: np.ndarray, nsigma: float) -> float:
    """Robust ``median + nsigma * sigma`` threshold from finite pixels."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + nsigma * _MAD_TO_SIGMA * mad


def _drop_small(mask: np.ndarray, min_size: int, rank: int) -> np.ndarray:
    """Drop connected regions smaller than ``min_size`` pixels.

    ``rank`` is scikit-image's neighbourhood rank (1 or 2), not the user-facing
    4-/8-connectivity.
    """
    labels = _connected_components(mask, connectivity=rank)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False  # background
    return keep[labels]


def _watershed_deblend(mask: np.ndarray, rank: int) -> np.ndarray:
    """Split touching regions in ``mask`` with a distance-transform watershed.

    ``rank`` is scikit-image's neighbourhood rank (1 or 2), not the user-facing
    4-/8-connectivity.
    """
    distance = ndi.distance_transform_edt(mask)
    coordinates = peak_local_max(
        distance, labels=mask, min_distance=1, exclude_border=False
    )
    seeds = np.zeros(mask.shape, dtype=bool)
    seeds[tuple(coordinates.T)] = True
    markers, n_markers = ndi.label(seeds, structure=np.ones((3, 3), dtype=int))
    if n_markers <= 1:
        return _connected_components(mask, connectivity=rank).astype(np.intp)
    return watershed(-distance, markers, mask=mask).astype(np.intp)
