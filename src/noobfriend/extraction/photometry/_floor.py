"""Area-scaled grow stop-check floor shared by the grow and re-grow paths.

noobase's grow stop is a cumulative inner-annulus SNR that stays disabled until
``min_pixels_before_stop_check`` pixels have been admitted, so *every* aperture
is at least that many pixels regardless of signal. That floor is a raw pixel
count, so to subtend a consistent sky *area* across bands of different pixel
scale it is scaled down on coarser grids, and -- where a segmentation map is
present -- capped at the seed's own segment size.

These helpers are used both by :meth:`ApertureSED.draft` (the all-band grow) and
by :meth:`ApertureSEDDraft.regrow` (re-growing one band), so they live in their
own module to keep those two callers free of an import cycle.
"""

from typing import Any

import numpy as np

from noobfriend.extraction.photometry._array import valid_data_mask

#: Lower bound on the scale-adjusted stop-check floor, so the annulus ring keeps
#: enough pixels for a stable SNR estimate even on the coarsest bands.
_LO_FLOOR = 8


def _generic_floor(scale: float, finest_scale: float, base: int) -> int:
    """Scale the grow stop-check warm-up floor by sky area across bands.

    ``min_pixels_before_stop_check`` is a pixel count, so for the minimum
    aperture to subtend a consistent sky *area* on grids of different pixel
    scale the base floor is multiplied by ``(scale / finest_scale) ** 2``. The
    scales are in pixels-per-degree, so a coarser band has a *smaller* value and
    thus a smaller floor (the finest band keeps ``base``); the result is clamped
    below at :data:`_LO_FLOOR` so the annulus ring stays measurable.

    Parameters
    ----------
    scale, finest_scale : float
        Local pixel scale (pixels-per-degree) of this band and of the finest
        band.
    base : int
        Floor at the finest band (the user value or the noobase default of 30).

    Returns
    -------
    int
    """
    if finest_scale <= 0.0:
        return max(_LO_FLOOR, base)
    scaled = base * (scale / finest_scale) ** 2
    return max(_LO_FLOOR, int(round(scaled)))


def _label_floor(
    generic: int,
    label_map: np.ndarray,
    data: np.ndarray,
    seed_xy: tuple[float, float],
) -> int:
    """Cap the warm-up floor at the seed's own segment, ``min(segment, generic)``.

    The segment size is the count of growable (finite, non-zero) pixels sharing
    the seed pixel's label. A seed on the background label ``0`` has no segment,
    so its size is ``1`` -- there is no stable signal to grow, and the stop check
    is left free to fire immediately.

    Parameters
    ----------
    generic : int
        The scale-adjusted floor from :func:`_generic_floor`.
    label_map : numpy.ndarray
        Segmentation labels on this band's grid.
    data : numpy.ndarray
        Band image, used to count only growable (valid) segment pixels.
    seed_xy : tuple[float, float]
        Seed ``(x, y)`` pixel.

    Returns
    -------
    int
    """
    seed_x, seed_y = int(round(seed_xy[0])), int(round(seed_xy[1]))
    n_rows, n_cols = data.shape
    if not (0 <= seed_y < n_rows and 0 <= seed_x < n_cols):
        return generic  # out of bounds: grow_aperture_mask raises the real error
    seed_label = int(label_map[seed_y, seed_x])
    if seed_label == 0:
        segment_size = 1
    else:
        segment_size = int(
            np.count_nonzero((label_map == seed_label) & valid_data_mask(data))
        )
    return min(segment_size, generic)


def resolve_floor(
    base: int,
    scale: float,
    finest_scale: float,
    label_map: Any | None,
    data: np.ndarray,
    seed_xy: tuple[float, float],
) -> int:
    """Resolve a band's stop-check floor: area-scaled, then capped by its segment.

    Combines :func:`_generic_floor` (area scaling vs the finest band) with
    :func:`_label_floor` (cap at the seed's segment) when a ``label_map`` is
    present.

    Parameters
    ----------
    base : int
        Floor at the finest band.
    scale, finest_scale : float
        Local pixel scale (pixels-per-degree) of this band and of the finest.
    label_map : array-like or None
        The segmentation map actually driving this band's growth, or ``None``.
    data : numpy.ndarray
        Band image (for the segment pixel count).
    seed_xy : tuple[float, float]
        Seed ``(x, y)`` pixel.

    Returns
    -------
    int
    """
    generic = _generic_floor(scale, finest_scale, base)
    if label_map is None:
        return generic
    return _label_floor(generic, np.asarray(label_map), data, seed_xy)
