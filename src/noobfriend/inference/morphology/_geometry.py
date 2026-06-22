"""Per-band render geometry: the bridge from sky parameters to pixel grids.

The model's geometry parameters live in a shared sky tangent plane -- a centre
offset in arcsec east/north of the target, a half-light radius in arcsec, a
position angle east of north. To render them into one band, we linearise that
band's WCS to an affine around the target centre and precompute, once, the
oversampled grid of east/north arcsec offsets at every subpixel. The differentiable
renderer then only has to shift, rotate, and scale that static grid -- no WCS calls
inside the sampler.

This keeps the native-grid contract: each band is rendered on its own grid with
its own affine; nothing is reprojected. This module is internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astropy.cosmology import FLRW

_ARCSEC_PER_DEG: float = 3600.0


@dataclass(frozen=True)
class BandGeometry:
    """Static render geometry for one band.

    Attributes
    ----------
    e_grid, n_grid : numpy.ndarray
        Oversampled arcsec offsets east and north of the target centre, one value
        per subpixel; shape ``(rows * oversample, cols * oversample)``.
    subpixel_area : float
        Solid angle of one oversampled subpixel, in arcsec^2 (for flux scaling).
    affine : numpy.ndarray
        The ``2x2`` matrix mapping ``[east, north]`` arcsec offsets to ``[dx, dy]``
        native-pixel offsets from the centre pixel.
    pix_center : tuple[float, float]
        The target centre in native ``(x, y)`` pixel coordinates.
    oversample : int
        Linear subpixel oversampling factor.
    native_shape : tuple[int, int]
        The native ``(rows, cols)`` image shape.
    """

    e_grid: np.ndarray
    n_grid: np.ndarray
    subpixel_area: float
    affine: np.ndarray
    pix_center: tuple[float, float]
    oversample: int
    native_shape: tuple[int, int]


def build_geometry(
    wcs: Any,
    center: tuple[float, float],
    native_shape: tuple[int, int],
    oversample: int,
) -> BandGeometry:
    """Linearise a band WCS into a :class:`BandGeometry`.

    Parameters
    ----------
    wcs : Any
        A WCS exposing the noobfriend world<->detector transform protocol.
    center : tuple of float
        Target ``(ra, dec)`` in degrees (ICRS).
    native_shape : tuple of int
        Native ``(rows, cols)`` image shape.
    oversample : int
        Linear subpixel oversampling factor (``>= 1``).

    Returns
    -------
    BandGeometry

    Raises
    ------
    ValueError
        If ``oversample < 1`` or the affine is singular (degenerate WCS).
    """
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}.")
    from noobfriend.extraction._wcs import world_detector_transforms

    world_to_detector, _ = world_detector_transforms(wcs)
    ra0, dec0 = float(center[0]), float(center[1])

    def w2d(ra: float, dec: float) -> tuple[float, float]:
        x, y = world_to_detector(ra, dec)
        return float(np.asarray(x)), float(np.asarray(y))

    xc, yc = w2d(ra0, dec0)
    step = 1.0 / _ARCSEC_PER_DEG
    dra = step / np.cos(np.radians(dec0))
    x_east, y_east = w2d(ra0 + dra, dec0)
    x_north, y_north = w2d(ra0, dec0 + step)
    affine = np.array(
        [[x_east - xc, x_north - xc], [y_east - yc, y_north - yc]], dtype=float
    )
    det = float(np.linalg.det(affine))
    if det == 0.0 or not np.isfinite(det):
        raise ValueError("WCS linearisation is singular at the target centre.")
    affine_inv = np.linalg.inv(affine)
    native_area = abs(float(np.linalg.det(affine_inv)))

    rows, cols = int(native_shape[0]), int(native_shape[1])
    o = int(oversample)
    xs = (np.arange(cols * o) + 0.5) / o - 0.5
    ys = (np.arange(rows * o) + 0.5) / o - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = grid_x - xc
    dy = grid_y - yc
    e_grid = affine_inv[0, 0] * dx + affine_inv[0, 1] * dy
    n_grid = affine_inv[1, 0] * dx + affine_inv[1, 1] * dy

    return BandGeometry(
        e_grid=e_grid,
        n_grid=n_grid,
        subpixel_area=native_area / (o * o),
        affine=affine,
        pix_center=(xc, yc),
        oversample=o,
        native_shape=(rows, cols),
    )


def center_to_subpixel(
    geom: BandGeometry, east: float, north: float
) -> tuple[float, float]:
    """Map an arcsec east/north offset to a fractional oversampled ``(i, j)`` index.

    Used to deposit a point source at the model centre on the oversampled grid.

    Parameters
    ----------
    geom : BandGeometry
        The band geometry.
    east, north : float
        Centre offset in arcsec east/north of the target.

    Returns
    -------
    tuple of float
        ``(col, row)`` fractional indices into the oversampled grid.
    """
    xc, yc = geom.pix_center
    dx, dy = geom.affine @ np.array([east, north], dtype=float)
    o = geom.oversample
    col = (xc + dx + 0.5) * o - 0.5
    row = (yc + dy + 0.5) * o - 0.5
    return float(col), float(row)


def kpc_per_arcsec(z: float, cosmology: FLRW | None = None) -> float:
    """Proper kiloparsecs per arcsecond at redshift ``z``.

    Parameters
    ----------
    z : float
        Redshift (``> 0``).
    cosmology : astropy.cosmology.FLRW, optional
        Cosmology to use; defaults to Planck18.

    Returns
    -------
    float
        Proper kpc per arcsec.

    Raises
    ------
    ValueError
        If ``z <= 0``.
    """
    if z <= 0:
        raise ValueError(f"kpc/arcsec needs z > 0, got {z}.")
    from astropy import units as u

    if cosmology is None:
        from astropy.cosmology import Planck18

        cosmology = Planck18
    per_arcmin = cosmology.kpc_proper_per_arcmin(z)
    return float(per_arcmin.to(u.kpc / u.arcsec).value)
