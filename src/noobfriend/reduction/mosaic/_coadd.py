"""Inverse-variance coadd of a mosaic tile (replaces jwst ``ResampleStep``).

Each member frame is reprojected onto the output tile with
:func:`noobase.image.reproject_exact` -- exact-polygon, surface-brightness
conserving, i.e. a square-kernel ``pixfrac=1`` drizzle -- and the frames are
combined inverse-variance-weighted. Validated against jwst ``ResampleStep`` on
real data: source fluxes agree to <1%, positions to ~0.02 px, ~7x faster.

Two decisions differ from jwst, both deliberate:

- The per-pixel ``err`` is the *honest* per-pixel noise (it matches the measured
  background RMS), not jwst's ~1.8x-inflated value. jwst bakes a scalar
  correlated-noise fudge into ``err``; that is only right for one aperture size.
- Instead, :func:`noise_kernel` records the noise *autocovariance* of the coadd
  (the resampling correlation is strong: nearest-neighbour rho ~0.5). Combined
  with the honest ``err`` it gives the exact aperture noise at any size via
  :func:`noobfriend.core.imgutils.correlated_sum_variance` -- validated to
  reproduce a synthetic correlated field's aperture noise to ~1%.

:class:`TileCoadd` accumulates frames one at a time onto a frame-sized
sub-window of the tile (peak memory is the tile plus one frame), so it streams a
large group. Frames arrive already sky-subtracted and WCS-corrected; this module
is pure ``numpy`` plus the shared reprojection helpers and
:mod:`noobfriend.core.imgutils`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from noobfriend.core.imgutils import NoiseAutocovariance, noise_autocovariance
from noobfriend.reduction.mosaic._reproject import frame_window, reproject_to_window
from noobfriend.reduction.mosaic._tiling import FieldGrid, TileSpec

__all__ = ["CoaddTile", "TileCoadd", "noise_kernel"]

#: JWST "do not use" pixel DQ bit (stable published value; kept jwst-import-free).
DO_NOT_USE: int = 1


@dataclass(frozen=True)
class CoaddTile:
    """The coadded planes of one tile.

    Attributes
    ----------
    sci : numpy.ndarray
        The inverse-variance-weighted surface brightness; NaN where no frame
        covers.
    err : numpy.ndarray
        The honest per-pixel 1-sigma noise (``1 / sqrt(weight)``); NaN where no
        coverage. Aperture noise needs this plus a :func:`noise_kernel`.
    wht : numpy.ndarray
        The summed inverse-variance weight (the ``ivm`` weight map).
    """

    sci: np.ndarray
    err: np.ndarray
    wht: np.ndarray


class TileCoadd:
    """Streaming inverse-variance coadd of one output tile.

    Parameters
    ----------
    field : FieldGrid
        The shared output grid the tile is a window of.
    tile : TileSpec
        The tile (pixel window of ``field``) being coadded.
    coarse_step : tuple of int, optional
        Coarse-grid WCS-evaluation stride for the reprojection, by default
        ``(64, 64)`` (the WCS is interpolated between; see
        :func:`noobase.image.make_pixel_corners`).
    mask_bit : int, optional
        DQ bit marking pixels to exclude, by default :data:`DO_NOT_USE`.
    """

    def __init__(
        self,
        field: FieldGrid,
        tile: TileSpec,
        *,
        coarse_step: tuple[int, int] | None = (64, 64),
        mask_bit: int = DO_NOT_USE,
    ) -> None:
        """See the class docstring for parameters."""
        self._field = field
        self._tile = tile
        self._coarse_step = coarse_step
        self._mask_bit = mask_bit
        self._num = np.zeros((tile.ny, tile.nx))  # sum of ivm * image
        self._wht = np.zeros((tile.ny, tile.nx))  # sum of ivm

    def add(
        self,
        data: np.ndarray,
        err: np.ndarray,
        dq: np.ndarray,
        wcs: Any,
        *,
        sky: float = 0.0,
    ) -> None:
        """Reproject one member frame onto its overlap with the tile and add it.

        Parameters
        ----------
        data, err, dq : numpy.ndarray
            The frame's science, error and DQ planes. ``mask_bit`` (bad) pixels
            are excluded.
        wcs : object
            The frame's WCS -- **already alignment-corrected** by the caller --
            exposing ``get_transform("world", "detector")`` /
            ``("detector", "world")``.
        sky : float, optional
            A scalar sky level subtracted from ``data`` before coadding (e.g.
            from :class:`~noobfriend.reduction.mosaic.SkyMatcher`), by default 0.
        """
        tile = self._tile
        detector_to_world = wcs.get_transform("detector", "world")
        world_to_detector = wcs.get_transform("world", "detector")
        fx0, fy0, fnx, fny = frame_window(
            np.asarray(data).shape, detector_to_world, self._field
        )
        # intersect the frame's field window with the tile's field region
        ox0 = max(fx0, tile.x0)
        oy0 = max(fy0, tile.y0)
        ox1 = min(fx0 + fnx, tile.x0 + tile.nx)
        oy1 = min(fy0 + fny, tile.y0 + tile.ny)
        if ox1 <= ox0 or oy1 <= oy0:
            return
        wny, wnx = oy1 - oy0, ox1 - ox0

        def window_pixel_to_world(
            x: Any, y: Any, ox0: int = ox0, oy0: int = oy0
        ) -> tuple[Any, Any]:
            return self._field.wcs.pixel_to_world_values(
                np.asarray(x) + ox0, np.asarray(y) + oy0
            )

        good = (np.asarray(dq) & self._mask_bit) == 0
        source = np.where(good, np.asarray(data) - sky, np.nan)
        errors = np.where(good, np.asarray(err), np.nan)
        image, weight, rerr = reproject_to_window(
            source,
            (wny, wnx),
            window_pixel_to_world,
            world_to_detector,
            self._coarse_step,
            error=errors,
        )
        valid = np.isfinite(image) & np.isfinite(rerr) & (rerr > 0) & (weight > 0)
        ivm = np.zeros_like(weight)
        ivm[valid] = weight[valid] / rerr[valid] ** 2
        sy, sx = (
            slice(oy0 - tile.y0, oy1 - tile.y0),
            slice(ox0 - tile.x0, ox1 - tile.x0),
        )
        self._num[sy, sx] += np.where(valid, ivm * image, 0.0)
        self._wht[sy, sx] += ivm

    def result(self) -> CoaddTile:
        """Return the coadded ``(sci, err, wht)`` planes.

        Returns
        -------
        CoaddTile
            ``sci = num / wht``, ``err = 1 / sqrt(wht)`` and ``wht`` itself; NaN
            where no frame contributed.
        """
        covered = self._wht > 0
        wht = self._wht
        sci = np.where(covered, self._num / np.where(covered, wht, 1.0), np.nan)
        err = np.where(covered, 1.0 / np.sqrt(np.where(covered, wht, 1.0)), np.nan)
        return CoaddTile(sci=sci, err=err, wht=wht)


def noise_kernel(
    sci: np.ndarray, err: np.ndarray, *, max_lag: int = 16, clip_sigma: float = 3.0
) -> NoiseAutocovariance:
    """Estimate the coadd's noise autocovariance from its blank sky.

    Resampling correlates neighbouring output pixels, so the per-pixel ``err``
    alone under-predicts aperture noise. This measures the stationary noise
    correlation ``C(d)`` from a source-masked region of the coadd; paired with
    the per-pixel ``err`` it gives the exact instrumental aperture noise via
    :func:`noobfriend.core.imgutils.correlated_sum_variance` (hybrid mode).

    Parameters
    ----------
    sci, err : numpy.ndarray
        The coadd science and error planes (e.g. from :meth:`TileCoadd.result`).
    max_lag : int, optional
        Half-width of the returned lag window, by default 16 (covers apertures
        up to radius ~8 px; a lag window needs to reach ``2 * radius``).
    clip_sigma : float, optional
        Sigma threshold for masking sources out of the sky estimate, by default
        3.0.

    Returns
    -------
    NoiseAutocovariance
        The ``error``-aware autocovariance (usable in hybrid
        :func:`correlated_sum_variance`).
    """
    covered = np.isfinite(sci)
    median = float(np.nanmedian(sci))
    robust = 0.5 * float(np.nanpercentile(sci, 84) - np.nanpercentile(sci, 16))
    sky = covered & (np.abs(sci - median) < clip_sigma * robust)
    return noise_autocovariance(
        np.nan_to_num(sci - median),
        mask=sky,
        max_lag=max_lag,
        error=np.nan_to_num(err),
    )
