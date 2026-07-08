"""DIY scalar cross-frame sky matching (replaces jwst ``SkyMatchStep``).

jwst ``SkyMatchStep(skymethod='match')`` is slow because it finds overlapping
frame pairs with spherical polygons -- the same O(N^2) spherical-geometry cost
as ``TweakRegStep`` -- while the sky statistics themselves are cheap. This does
the same scalar leveling without the spherical polygons: each frame's
source-masked sky is reprojected onto a *coarse* shared field grid (sky needs no
resolution), overlaps are read straight off the reprojected coverage windows
(bounding-box, not polygon intersection), and one small least-squares gives the
per-frame offsets.

Two levels:

- :func:`frame_sky` -- one frame's own robust scalar sky (median + a single MAD
  clip; fast and parallel). Subtracting it is the ``'global'`` method, adequate
  when the sky is spatially uniform.
- :class:`SkyMatcher` -- the ``'match'`` method: accumulate frames onto the
  coarse grid, then solve the overlap-consistent per-frame offsets (match-down
  gauge, so the lowest frame keeps zero), robust to a field-scale gradient.

On a chain that already zeroes each frame's level (a 2-D background subtraction,
or even the 1/f column/row median) the offsets come out ~0 -- a cheap, logged
no-op -- while a chain that skips those gets real leveling. This module is pure
``numpy`` plus the shared reprojection helpers; it takes arrays, not files.
"""

from itertools import combinations
from typing import Any

import numpy as np

from noobfriend.reduction.mosaic._reproject import frame_window, reproject_to_window
from noobfriend.reduction.mosaic._tiling import FieldGrid

__all__ = ["SkyMatcher", "frame_sky"]

#: JWST "do not use" pixel DQ bit (stable published value; kept jwst-import-free).
DO_NOT_USE: int = 1

#: Minimum finite overlap pixels for a frame pair to contribute a match equation.
_MIN_OVERLAP_PIXELS: int = 20


def frame_sky(
    data: np.ndarray, dq: np.ndarray, *, mask_bit: int = DO_NOT_USE
) -> tuple[float, float]:
    """Return a frame's robust scalar sky level and its MAD.

    The good (non-``mask_bit``, finite) pixels are reduced to a median and a
    median-absolute-deviation scale; pixels above ``median + 3 * mad`` (sources)
    are dropped and the median re-taken. One clip, no iteration -- fast enough
    to run per frame in the detection pass.

    Parameters
    ----------
    data : numpy.ndarray
        The frame's science plane.
    dq : numpy.ndarray
        The frame's DQ plane; pixels with ``mask_bit`` set are excluded.
    mask_bit : int, optional
        DQ bit marking pixels to exclude, by default :data:`DO_NOT_USE`.

    Returns
    -------
    level : float
        The robust sky level (``nan`` if no good pixels).
    mad : float
        The (Gaussian-scaled) median absolute deviation of the good pixels.
    """
    src, level, mad = _masked_sky_plane(data, dq, mask_bit)
    return level, mad


def _masked_sky_plane(
    data: np.ndarray, dq: np.ndarray, mask_bit: int
) -> tuple[np.ndarray, float, float]:
    """Return ``(source_masked_plane, level, mad)`` for a frame.

    ``source_masked_plane`` is ``data`` with bad-DQ and above-threshold (source)
    pixels set to ``nan`` -- the sky-only plane to reproject.
    """
    values = np.asarray(data)
    good = ((np.asarray(dq) & mask_bit) == 0) & np.isfinite(values)
    vals = values[good]
    if vals.size == 0:
        return np.full(values.shape, np.nan), float("nan"), float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) * 1.4826)
    # keep sky (<= threshold), drop only brighter sources; ``<=`` so a noiseless
    # constant frame (mad == 0) is kept whole rather than fully masked.
    sky = vals <= med + 3.0 * mad
    src = np.where(good & (values <= med + 3.0 * mad), values, np.nan)
    return src, float(np.median(vals[sky])), mad


class SkyMatcher:
    """Cross-frame scalar sky leveling on a coarse field grid (streaming).

    Frames are added one at a time -- each frame's source-masked sky is
    reprojected onto its window of the (coarse) shared grid and kept, so peak
    memory is the sum of the small windowed maps, not the frames. Once all
    frames are in, :meth:`match` reads the overlaps off the stored maps and
    solves the per-frame offsets.

    Parameters
    ----------
    field : FieldGrid
        The shared (coarse, e.g. ~0.5 arcsec/pix) grid the sky maps live on.
    coarse_step : tuple of int, optional
        Coarse-grid WCS-evaluation stride for the reprojection; ``None``
        (default) evaluates every corner exactly (fine on a small coarse grid).
    mask_bit : int, optional
        DQ bit marking pixels to exclude, by default :data:`DO_NOT_USE`.
    min_overlap : int, optional
        Minimum finite overlap pixels for a frame pair to contribute a match
        equation, by default :data:`_MIN_OVERLAP_PIXELS`.
    """

    def __init__(
        self,
        field: FieldGrid,
        *,
        coarse_step: tuple[int, int] | None = None,
        mask_bit: int = DO_NOT_USE,
        min_overlap: int = _MIN_OVERLAP_PIXELS,
    ) -> None:
        """See the class docstring for parameters."""
        self._field = field
        self._coarse_step = coarse_step
        self._mask_bit = mask_bit
        self._min_overlap = min_overlap
        self._maps: dict[str, np.ndarray | None] = {}
        self._windows: dict[str, tuple[int, int, int, int]] = {}
        self._levels: dict[str, float] = {}

    def add(self, frame_id: str, data: np.ndarray, dq: np.ndarray, wcs: Any) -> None:
        """Reproject one frame's source-masked sky onto its grid window.

        Parameters
        ----------
        frame_id : str
            Identifier the returned offset is keyed by.
        data, dq : numpy.ndarray
            The frame's science and DQ planes.
        wcs : object
            The frame's WCS exposing ``get_transform("world", "detector")`` /
            ``("detector", "world")`` (a JWST gwcs, a NoobWCS, or any adapter).
        """
        src, level, _ = _masked_sky_plane(data, dq, self._mask_bit)
        self._levels[frame_id] = level
        detector_to_world = wcs.get_transform("detector", "world")
        world_to_detector = wcs.get_transform("world", "detector")
        x0, y0, nx, ny = frame_window(
            np.asarray(data).shape, detector_to_world, self._field
        )
        self._windows[frame_id] = (x0, y0, nx, ny)
        if nx == 0 or ny == 0:
            self._maps[frame_id] = None
            return

        def window_pixel_to_world(
            x: Any, y: Any, x0: int = x0, y0: int = y0
        ) -> tuple[Any, Any]:
            return self._field.wcs.pixel_to_world_values(
                np.asarray(x) + x0, np.asarray(y) + y0
            )

        image, weight, _ = reproject_to_window(
            src, (ny, nx), window_pixel_to_world, world_to_detector, self._coarse_step
        )
        self._maps[frame_id] = np.where(weight > 0.5, image, np.nan)

    def global_levels(self) -> dict[str, float]:
        """Return each frame's own robust sky level (the ``'global'`` method)."""
        return dict(self._levels)

    def match(self) -> dict[str, float]:
        """Return the match-consistent per-frame sky level to subtract.

        Pairs of frames whose windows overlap contribute one equation
        ``offset_i - offset_j = median(map_i - map_j)`` over the shared pixels;
        the least-squares solution is gauge-fixed match-down (the lowest frame
        gets zero). Frames with no usable overlap get zero.

        Returns
        -------
        dict of str to float
            The scalar sky offset to subtract from each frame, keyed by
            ``frame_id``.
        """
        ids = list(self._maps)
        index = {frame_id: i for i, frame_id in enumerate(ids)}
        pairs: list[tuple[int, int, float]] = []
        for a, b in combinations(ids, 2):
            diff = self._overlap_diff(a, b)
            if diff is not None:
                pairs.append((index[a], index[b], diff))
        offsets = _solve_offsets(pairs, len(ids))
        return {frame_id: float(offsets[index[frame_id]]) for frame_id in ids}

    def _overlap_diff(self, a: str, b: str) -> float | None:
        """Median sky difference of frames ``a`` and ``b`` over their overlap."""
        map_a, map_b = self._maps[a], self._maps[b]
        if map_a is None or map_b is None:
            return None
        xa, ya, na, ma = self._windows[a]
        xb, yb, nb, mb = self._windows[b]
        ox0, oy0 = max(xa, xb), max(ya, yb)
        ox1, oy1 = min(xa + na, xb + nb), min(ya + ma, yb + mb)
        if ox1 <= ox0 or oy1 <= oy0:
            return None
        sub_a = map_a[oy0 - ya : oy1 - ya, ox0 - xa : ox1 - xa]
        sub_b = map_b[oy0 - yb : oy1 - yb, ox0 - xb : ox1 - xb]
        diff = sub_a - sub_b
        if np.isfinite(diff).sum() < self._min_overlap:
            return None
        return float(np.nanmedian(diff))


def _solve_offsets(pairs: list[tuple[int, int, float]], n: int) -> np.ndarray:
    """Least-squares per-frame offsets from pairwise diffs, gauge-fixed match-down.

    Each pair adds ``offset_i - offset_j = diff``; a final ``sum(offset) = 0``
    row fixes the gauge, and the result is shifted so the minimum is zero.
    """
    if not pairs:
        return np.zeros(n)
    rows, rhs = [], []
    for i, j, diff in pairs:
        row = np.zeros(n)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(diff)
    rows.append(np.ones(n))  # gauge: sum of offsets = 0
    rhs.append(0.0)
    offsets, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return offsets - offsets.min()
