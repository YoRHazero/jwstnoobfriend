"""Combine N same-field dither exposures into one union-footprint heatmap.

The line-likelihood product is the band-pass SNR map (hot = likely emission
line). This module builds the two heatmap products:

* per exposure -- :func:`exposure_heatmap`, the SNR map 1:1 with the frame;
* the union -- :func:`combine_heatmap` reprojects every exposure onto a common
  grid covering the *union* of all footprints (nothing clipped), robustly
  median-combines them (rejecting cosmic rays and detector-fixed artifacts,
  which land at scattered sky positions across dithers), and band-passes the
  deep result.

Median (not a clipped mean) is used for the cross-exposure combine: with few
frames a single extreme outlier inflates the clip width and survives, whereas
the median ignores it outright -- at the cost of a ~1.25x noise penalty over an
inverse-variance mean, which the depth gain absorbs.

This module is internal to ``noobfriend.extraction.grism.linefind``.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from gwcs import WCS
from noobase.image import make_pixel_corners, reproject_exact

from noobfriend.extraction._wcs import world_detector_transforms
from noobfriend.extraction.grism._array import _native
from noobfriend.extraction.grism._coverage import FrameMeta
from noobfriend.extraction.grism.linefind._filter import band_pass_snr

#: Standard error of the median relative to the mean (asymptotic, Gaussian).
_MEDIAN_NOISE = 1.2533


@dataclass(frozen=True)
class UnionGrid:
    """The common output grid covering the union of all exposures' footprints.

    Attributes
    ----------
    shape : tuple[int, int]
        Output grid shape ``(H, W)``.
    x_off, y_off : int
        Offset of the union grid origin in the reference exposure's detector
        pixels: union pixel ``(ux, uy)`` is reference pixel
        ``(ux + x_off, uy + y_off)``. Either may be negative.
    reference_wcs : gwcs.WCS
        WCS of the reference exposure (the grid is its detector frame,
        translated to cover the union), for mapping heatmap pixels to the sky.
    """

    shape: tuple[int, int]
    x_off: int
    y_off: int
    reference_wcs: WCS


@dataclass(frozen=True)
class CombinedHeatmap:
    """The union-footprint line-likelihood heatmap and the layers behind it.

    Attributes
    ----------
    heatmap : numpy.ndarray
        Band-pass SNR on the union grid (the product); ``NaN`` outside coverage.
    data, error : numpy.ndarray
        The robust-combined data and propagated 1-sigma error on the union grid.
    count : numpy.ndarray
        Per-pixel number of exposures covering it (the coverage / depth map).
    grid : UnionGrid
        The output grid geometry.
    """

    heatmap: np.ndarray
    data: np.ndarray
    error: np.ndarray
    count: np.ndarray
    grid: UnionGrid


def exposure_heatmap(
    data: np.ndarray,
    error: np.ndarray,
    *,
    dispersion_axis: int,
    line_scales: Sequence[tuple[float, float]],
    continuum_sigma_disp: float,
) -> np.ndarray:
    """Return one exposure's line-likelihood heatmap (band-pass SNR, 1:1).

    Parameters
    ----------
    data, error : numpy.ndarray
        The 2-D frame and its 1-sigma error.
    dispersion_axis : int
        Array axis the grism disperses along (``1`` row / ``0`` column).
    line_scales : Sequence[tuple[float, float]]
        ``(sigma_cross, sigma_disp)`` pixel sigma pairs to scan.
    continuum_sigma_disp : float
        Along-dispersion continuum Gaussian sigma (pixels).

    Returns
    -------
    numpy.ndarray
        The per-pixel maximum band-pass SNR over scales, same shape as ``data``;
        ``NaN`` where masked / low-coverage.
    """
    bp = band_pass_snr(
        data,
        error,
        dispersion_axis=dispersion_axis,
        line_scales=line_scales,
        continuum_sigma_disp=continuum_sigma_disp,
    )
    filled = np.where(np.isfinite(bp.snr), bp.snr, -np.inf).max(axis=0)
    return np.where(np.isfinite(filled), filled, np.nan)


def _union_bbox(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    """Integer bounding box ``(x_off, y_off, (H, W))`` of scattered points.

    Parameters
    ----------
    xs, ys : numpy.ndarray
        Point coordinates (e.g. exposure footprint corners in reference pixels).

    Returns
    -------
    tuple[int, int, tuple[int, int]]
        ``(x_off, y_off, (height, width))`` of the floor/ceil bounding box that
        contains every point, with origin ``(x_off, y_off)``.
    """
    x_off = int(np.floor(xs.min()))
    y_off = int(np.floor(ys.min()))
    width = int(np.ceil(xs.max())) - x_off + 1
    height = int(np.ceil(ys.max())) - y_off + 1
    return x_off, y_off, (height, width)


def _union_grid(frames: Sequence[FrameMeta], reference_index: int) -> UnionGrid:
    """Build the union grid by mapping every footprint into reference pixels."""
    ref = frames[reference_index]
    ref_w2d, _ = world_detector_transforms(ref.wcs)
    xs: list[float] = []
    ys: list[float] = []
    for f in frames:
        _, f_d2w = world_detector_transforms(f.wcs)
        h, w = f.shape
        for cx, cy in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)):
            ra, dec = f_d2w(cx, cy)
            rx, ry = ref_w2d(ra, dec)
            xs.append(float(rx))
            ys.append(float(ry))
    x_off, y_off, shape = _union_bbox(np.array(xs), np.array(ys))
    return UnionGrid(shape, x_off, y_off, ref.wcs)


def _combine_stack(
    images: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-combine an aligned ``(N, H, W)`` stack with NaN as missing.

    Parameters
    ----------
    images, errors : numpy.ndarray
        Reprojected data and 1-sigma error stacks, ``NaN`` outside each
        exposure's coverage.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(data, error, count)``: the per-pixel median, its propagated 1-sigma
        (``1.2533 * rms(error) / sqrt(count)``), and the integer coverage count.
        Uncovered pixels (``count == 0``) are ``NaN`` in ``data`` / ``error``.
    """
    covered = np.isfinite(images)
    count = covered.sum(axis=0).astype(np.int32)
    safe = np.where(count > 0, count, 1)
    errors_covered = np.where(covered, errors, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.nanmedian(images, axis=0)
        per_var = np.nanmean(errors_covered**2, axis=0)
    error = _MEDIAN_NOISE * np.sqrt(per_var) / np.sqrt(safe)
    covered = count > 0
    data = np.where(covered, data, np.nan)
    error = np.where(covered, error, np.nan)
    return data, error, count


def combine_heatmap(
    frames: Sequence[FrameMeta],
    load: Callable[[str], tuple[np.ndarray, np.ndarray]],
    *,
    dispersion_axis: int,
    line_scales: Sequence[tuple[float, float]],
    continuum_sigma_disp: float,
    reference_index: int = 0,
    coarse_step: tuple[int, int] | None = (64, 64),
) -> CombinedHeatmap:
    """Combine same-field dither exposures into one union-footprint heatmap.

    Reprojects every exposure onto a common grid spanning the union of all
    footprints, median-combines (rejecting cosmic rays and detector-fixed
    artifacts), and band-passes the deep result into a line-likelihood heatmap.

    Parameters
    ----------
    frames : Sequence[FrameMeta]
        The exposures (``wcs`` + ``shape`` + ``id``); must share field/grism/PA
        (same-pointing dithers). The undispersed astrometry of each ``wcs`` sets
        the alignment.
    load : collections.abc.Callable
        ``load(id) -> (data, error)``, called once per exposure to fetch pixels.
    dispersion_axis : int
        Array axis the grism disperses along (``1`` row / ``0`` column); the
        union grid is reference-detector-aligned so this is the reference's axis.
    line_scales : Sequence[tuple[float, float]]
        ``(sigma_cross, sigma_disp)`` pixel sigma pairs to scan.
    continuum_sigma_disp : float
        Along-dispersion continuum Gaussian sigma (pixels).
    reference_index : int, optional
        Index of the exposure whose detector frame defines the grid, by default
        ``0``.
    coarse_step : tuple[int, int] or None, optional
        ``make_pixel_corners`` speedup for the expensive GWCS inverse; ``(64,
        64)`` by default. Safe here because the alignment uses the smooth
        undispersed astrometry, not the curved trace.

    Returns
    -------
    CombinedHeatmap
        The union heatmap plus the combined data / error / coverage and grid.
    """
    grid = _union_grid(frames, reference_index)
    if coarse_step is not None:
        # make_pixel_corners requires coarse_step to divide the grid; pad up
        # (the extra rows/cols fall outside coverage -> NaN).
        sy, sx = coarse_step
        h, w = grid.shape
        grid = UnionGrid(
            (-(-h // sy) * sy, -(-w // sx) * sx),
            grid.x_off,
            grid.y_off,
            grid.reference_wcs,
        )
    _, ref_d2w = world_detector_transforms(grid.reference_wcs)

    def union_p2w(ux: np.ndarray, uy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ref_d2w(np.asarray(ux) + grid.x_off, np.asarray(uy) + grid.y_off)

    images: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for f in frames:
        data, error = load(f.id)
        f_w2d, _ = world_detector_transforms(f.wcs)
        corners = make_pixel_corners(
            grid.shape,
            target_pixel_to_world=union_p2w,
            source_world_to_pixel=f_w2d,
            coarse_step=coarse_step,
        )
        res = reproject_exact(_native(data), corners, error=_native(error))
        covered = res.weight > 0
        images.append(np.where(covered, res.image, np.nan).astype(np.float32))
        errors.append(np.where(covered, res.error, np.nan).astype(np.float32))

    data, error, count = _combine_stack(np.stack(images), np.stack(errors))
    heatmap = exposure_heatmap(
        data,
        error,
        dispersion_axis=dispersion_axis,
        line_scales=line_scales,
        continuum_sigma_disp=continuum_sigma_disp,
    )
    return CombinedHeatmap(heatmap, data, error, count, grid)
