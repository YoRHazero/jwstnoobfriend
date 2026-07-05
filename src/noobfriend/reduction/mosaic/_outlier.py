"""Cross-exposure outlier (cosmic-ray) flagging on a shared field grid.

A ``noobase``-backed replacement for the imaging mode of jwst's
``OutlierDetectionStep``, whose only persistent product is DQ bits: each
exposure is reprojected onto one field grid, a pixel-wise median over the
exposure layers builds a CR-free reference, the median is "blotted" back into
every frame's detector plane, and pixels that stand off the blot by more than a
noise + interpolation-slope threshold are flagged. The flagging math
(:func:`flag_outliers`, :func:`_abs_deriv`) replicates
``stcal.outlier_detection.utils.flag_resampled_crs`` exactly so the two engines
are comparable; the resampling differs deliberately:

- both directions use :func:`noobase.image.reproject_exact` -- the exact-polygon
  (surface-brightness-conserving) equivalent of drizzle's ``pixfrac=1`` square
  kernel -- instead of drizzle forward / spline blot;
- input NaNs and DQ-masked pixels are *missing* (renormalised away), not
  zero-weighted, and the low-coverage edge mask uses the valid-overlap weight
  instead of an ivm drizzle weight (equivalent for homogeneous-depth groups);
- the median stack streams through a disk-backed layer file
  (:class:`FieldMedian`), so peak memory is one frame plus the median image
  regardless of group size.

Like the rest of :mod:`noobfriend.reduction.mosaic` this module is jwst-free:
frames come in as ``(data, err, dq)`` arrays plus a WCS exposing the gwcs
``get_transform("detector", "world")`` protocol, and the grid is a
:class:`~noobfriend.reduction.mosaic.FieldGrid`.
"""

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from noobase.image import make_pixel_corners, reproject_exact

from noobfriend.reduction.mosaic._tiling import FieldGrid

#: JWST pixel DQ flags this module reads/writes (stable published values;
#: defined here so the reduction layer stays jwst-import-free).
DO_NOT_USE: int = 1
OUTLIER: int = 16
#: The bits :func:`flag_outliers` hits are meant to set: ``DO_NOT_USE | OUTLIER``.
OUTLIER_DQ: int = DO_NOT_USE | OUTLIER

#: Number of sample points per frame edge when projecting a frame's outline
#: onto the grid (distortion bows the edges, so corners alone under-cover).
_EDGE_SAMPLES: int = 9

#: Extra grid pixels around a frame's projected bounding box.
_WINDOW_MARGIN: int = 8


def _as_native_float(data: np.ndarray) -> np.ndarray:
    """Coerce to native-endian, C-contiguous float32/float64 for ``noobase``.

    JWST FITS arrays are typically big-endian (``>f4``); float width is
    preserved and non-float input is promoted to ``float64``.
    """
    array = np.asarray(data)
    if array.dtype.kind == "f" and array.dtype.itemsize in (4, 8):
        target = np.dtype(f"=f{array.dtype.itemsize}")
    else:
        target = np.dtype("=f8")
    return np.ascontiguousarray(array, dtype=target)


def _padded(length: int, step: int) -> int:
    """Round ``length`` up to a multiple of ``step``.

    ``make_pixel_corners`` requires ``coarse_step`` to divide the target shape
    exactly, so target windows are padded up and the result cropped back.
    """
    return int(math.ceil(length / step)) * step


def _frame_window(
    shape: tuple[int, int],
    detector_to_world: Callable[..., tuple[Any, Any]],
    grid: FieldGrid,
) -> tuple[int, int, int, int]:
    """Return the grid-pixel window ``(x0, y0, nx, ny)`` covering a frame.

    The frame outline is sampled along each edge (not just the corners --
    distortion bows the edges outward), projected into grid pixels, and the
    bounding box padded by :data:`_WINDOW_MARGIN` and clipped to the grid.
    """
    ny_f, nx_f = shape
    t = np.linspace(0.0, 1.0, _EDGE_SAMPLES)
    xs = np.concatenate(
        [t * (nx_f - 1), np.full_like(t, nx_f - 1), t * (nx_f - 1), np.zeros_like(t)]
    )
    ys = np.concatenate(
        [np.zeros_like(t), t * (ny_f - 1), np.full_like(t, ny_f - 1), t * (ny_f - 1)]
    )
    ra, dec = detector_to_world(xs, ys)
    gx, gy = grid.wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
    ny_g, nx_g = grid.shape
    x0 = max(0, int(np.floor(np.min(gx))) - _WINDOW_MARGIN)
    y0 = max(0, int(np.floor(np.min(gy))) - _WINDOW_MARGIN)
    x1 = min(nx_g, int(np.ceil(np.max(gx))) + _WINDOW_MARGIN + 1)
    y1 = min(ny_g, int(np.ceil(np.max(gy))) + _WINDOW_MARGIN + 1)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _reproject(
    source: np.ndarray,
    target_shape: tuple[int, int],
    target_pixel_to_world: Callable[..., tuple[Any, Any]],
    source_world_to_pixel: Callable[..., tuple[Any, Any]],
    coarse_step: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject ``source`` onto ``target_shape``; return ``(image, weight)``.

    The target shape is padded up to a ``coarse_step`` multiple (a
    ``make_pixel_corners`` requirement) and the result cropped back; the pad
    rows/columns fall outside the source and cost nothing. ``image`` is NaN
    where nothing valid contributes; ``weight`` is the valid-pixel overlap
    fraction (NaN/masked source pixels excluded).
    """
    ny, nx = target_shape
    if coarse_step is not None:
        shape = (_padded(ny, coarse_step[0]), _padded(nx, coarse_step[1]))
    else:
        shape = (ny, nx)
    corners = make_pixel_corners(
        shape,
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
        coarse_step=coarse_step,
    )
    result = reproject_exact(_as_native_float(source), corners)
    return result.image[:ny, :nx], result.weight[:ny, :nx]


class FieldMedian:
    """Streaming median stack over the exposure layers of one mosaic group.

    Frames are added one at a time: each frame's good pixels are reprojected
    onto the grid sub-window under its footprint and pasted into the layer of
    the exposure it belongs to, so all detectors of one dither share a layer
    (mirroring jwst's ``group_id`` layering; where sibling detectors overlap by
    a sliver the later paste wins). The stack lives in a disk-backed memmap
    when ``work_dir`` is given, so peak memory stays at one frame plus one
    strip regardless of how many exposures the group holds.

    Parameters
    ----------
    grid : FieldGrid
        The shared field grid the layers live on (native-ish pixel scale --
        CR detection needs no supersampling).
    layers : Sequence of str
        The exposure keys, one per layer, in any stable order.
    work_dir : str or Path, optional
        Directory for the backing memmap file (created if missing). ``None``
        keeps the stack in memory (small groups / tests).
    maskpt : float, optional
        Low-coverage mask threshold: window pixels whose valid-overlap weight
        falls below ``maskpt`` times the window's mean positive weight are
        dropped before entering the median (jwst's edge mask, with the
        valid-overlap weight standing in for the ivm drizzle weight).

    Raises
    ------
    ValueError
        If ``layers`` is empty.
    """

    def __init__(
        self,
        grid: FieldGrid,
        layers: Sequence[str],
        *,
        work_dir: str | Path | None = None,
        maskpt: float = 0.7,
    ) -> None:
        """See the class docstring for parameters."""
        if not layers:
            raise ValueError("FieldMedian needs at least one layer.")
        self._grid = grid
        self._index = {key: i for i, key in enumerate(layers)}
        self._maskpt = float(maskpt)
        shape = (len(self._index), *grid.shape)
        self._path: Path | None = None
        if work_dir is None:
            self._stack: np.ndarray = np.full(shape, np.nan, dtype=np.float32)
        else:
            directory = Path(work_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / "median_stack.dat"
            self._stack = np.memmap(
                self._path, dtype=np.float32, mode="w+", shape=shape
            )
            self._stack[:] = np.nan

    def add(
        self,
        layer: str,
        data: np.ndarray,
        dq: np.ndarray,
        wcs: Any,
        *,
        coarse_step: tuple[int, int] | None = (64, 64),
    ) -> None:
        """Reproject one frame into its exposure's layer.

        Parameters
        ----------
        layer : str
            The exposure key of the layer this frame belongs to.
        data, dq : numpy.ndarray
            The frame's science and DQ planes; ``DO_NOT_USE`` pixels are
            treated as missing.
        wcs : object
            The frame's WCS, exposing ``get_transform("world", "detector")`` /
            ``("detector", "world")`` (a JWST imaging gwcs, or any adapter).
        coarse_step : tuple of int, optional
            Coarse-grid WCS evaluation for :func:`noobase.image.make_pixel_corners`
            (the expensive gwcs inverse is interpolated), by default ``(64, 64)``.
            ``None`` evaluates every corner exactly.

        Raises
        ------
        KeyError
            ``layer`` was not declared at construction.
        """
        index = self._index[layer]
        world_to_detector = wcs.get_transform("world", "detector")
        detector_to_world = wcs.get_transform("detector", "world")
        x0, y0, nx, ny = _frame_window(data.shape, detector_to_world, self._grid)
        if nx == 0 or ny == 0:
            return

        def window_pixel_to_world(x: Any, y: Any) -> tuple[Any, Any]:
            return self._grid.wcs.pixel_to_world_values(
                np.asarray(x) + x0, np.asarray(y) + y0
            )

        source = np.where((np.asarray(dq) & DO_NOT_USE) == 0, data, np.nan)
        image, weight = _reproject(
            source, (ny, nx), window_pixel_to_world, world_to_detector, coarse_step
        )
        positive = weight[weight > 0]
        if positive.size == 0:
            return
        image = np.where(weight >= self._maskpt * positive.mean(), image, np.nan)

        window = self._stack[index, y0 : y0 + ny, x0 : x0 + nx]
        np.copyto(window, image.astype(np.float32), where=np.isfinite(image))

    def median(self, *, strip: int = 512) -> np.ndarray:
        """Return the pixel-wise median over the layers, computed in strips.

        Parameters
        ----------
        strip : int, optional
            Row-strip height, by default 512. Peak memory for the reduction is
            ``n_layers x strip x width`` regardless of the stack's total size.

        Returns
        -------
        numpy.ndarray
            ``float32`` median image on the grid; NaN where no layer covers.
        """
        ny, nx = self._grid.shape
        out = np.full((ny, nx), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN", RuntimeWarning)
            for row in range(0, ny, strip):
                block = np.asarray(self._stack[:, row : row + strip, :])
                out[row : row + strip] = np.nanmedian(block, axis=0)
        return out

    def cleanup(self) -> None:
        """Release the backing memmap and delete its file (no-op in memory)."""
        if self._path is not None:
            del self._stack
            self._path.unlink(missing_ok=True)
            self._path = None


def blot_to_frame(
    median: np.ndarray,
    grid: FieldGrid,
    wcs: Any,
    shape: tuple[int, int],
    *,
    coarse_step: tuple[int, int] | None = (64, 64),
) -> np.ndarray:
    """Reproject the field median back into one frame's detector plane.

    The cheap direction: the target transform is the frame's *forward* gwcs and
    the source inverse is the grid's analytic TAN, so no numerical gwcs
    inversion is involved.

    Parameters
    ----------
    median : numpy.ndarray
        The field median image (from :meth:`FieldMedian.median`).
    grid : FieldGrid
        The grid ``median`` lives on.
    wcs : object
        The frame's WCS (``get_transform("detector", "world")`` protocol).
    shape : tuple of int
        The frame's ``(ny, nx)`` shape.
    coarse_step : tuple of int, optional
        Coarse-grid WCS evaluation, by default ``(64, 64)``; ``None`` is exact.

    Returns
    -------
    numpy.ndarray
        The blotted median in detector pixels; NaN where the median has no
        coverage (those pixels are never flagged).
    """
    detector_to_world = wcs.get_transform("detector", "world")
    image, _ = _reproject(
        median, shape, detector_to_world, grid.wcs.world_to_pixel_values, coarse_step
    )
    return image


def _abs_deriv(array: np.ndarray) -> np.ndarray:
    """Local maximum absolute neighbour difference, replicating stcal.

    Mirrors ``stcal.outlier_detection.utils._abs_deriv`` operation-for-operation
    (including its NaN handling) so :func:`flag_outliers` stays comparable to
    the jwst engine.
    """
    out = np.zeros_like(array)
    if np.issubdtype(array.dtype, np.floating):
        out[np.isnan(array)] = np.nan

    row_diff = np.abs(np.diff(array, axis=0))
    np.putmask(out[1:], np.isfinite(row_diff), row_diff)
    row_offset_view = out[:-1]
    np.putmask(row_offset_view, row_diff > row_offset_view, row_diff)
    del row_diff

    col_diff = np.abs(np.diff(array, axis=1))
    col_offset_view = out[:, 1:]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    col_offset_view = out[:, :-1]
    np.putmask(col_offset_view, col_diff > col_offset_view, col_diff)
    return out


def _dilate3(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation via shifted ORs (jwst's boxcar-smoothed mask)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    grown = out.copy()
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    return grown


def flag_outliers(
    data: np.ndarray,
    err: np.ndarray,
    blot: np.ndarray,
    *,
    snr: tuple[float, float] = (5.0, 4.0),
    scale: tuple[float, float] = (1.2, 0.7),
    backg: float = 0.0,
) -> np.ndarray:
    """Return the cosmic-ray mask of one frame against its blotted median.

    An exact replica of ``stcal.outlier_detection.utils.flag_resampled_crs``:
    a first mask at ``(snr[0], scale[0])`` is grown by one pixel (3x3) and
    intersected with a looser second mask at ``(snr[1], scale[1])``, where each
    threshold is ``scale x |local blot slope| + snr x err`` -- the slope term
    absorbs resampling/registration mismatch so bright-source edges are not
    flagged. Pixels where ``blot`` is NaN (no median coverage) are never
    flagged. The caller applies the mask as ``dq |= mask * OUTLIER_DQ``.

    Parameters
    ----------
    data, err : numpy.ndarray
        The frame's science and error planes.
    blot : numpy.ndarray
        The blotted median (from :func:`blot_to_frame`).
    snr : tuple of float, optional
        First/second-pass significance thresholds, jwst defaults ``(5.0, 4.0)``.
    scale : tuple of float, optional
        First/second-pass blot-slope scalings, jwst defaults ``(1.2, 0.7)``.
    backg : float, optional
        Scalar background offset of ``data`` relative to ``blot``, by default
        0 (the CLEAR chain background-subtracts at 2b).

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where a pixel is an outlier.
    """
    err_data = np.nan_to_num(err)
    blot_deriv = _abs_deriv(np.asarray(blot))
    diff_noise = np.abs(data - blot - backg)

    mask1 = np.greater(diff_noise, scale[0] * blot_deriv + snr[0] * err_data)
    mask2 = np.greater(diff_noise, scale[1] * blot_deriv + snr[1] * err_data)
    return _dilate3(mask1) & mask2
