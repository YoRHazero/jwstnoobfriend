"""Output-grid geometry for stage-3 mosaicking: one field WCS, many sky tiles.

A stage-3 mosaic is identified by metadata (``field`` + ``filter``); this module
turns the *footprints* of that group into the **output** geometry the resample
step drizzles onto. The contract is deliberately pure (corners in, geometry out)
and navigation-/jwst-agnostic, the stage-3 analogue of the ``(data, err, dq)``
step contract used by the per-frame reduction steps.

The design follows two rules that keep tiles stitchable and memory bounded:

- **One canonical field WCS per (field, filter).** Every tile is a pixel window
  of this single TAN projection, so tiles -- and, when the same ``pixel_scale``
  is used, different bands -- share one grid and stitch/stack without
  resampling. :func:`field_grid` builds it from the union of footprints.
- **Tiles fit the field, not a rigid lattice.** :func:`tile_grid` splits the
  field bounding box into *near-equal* tiles of about ``target_size`` (so the
  last row/column is never a sliver) with an ``overlap`` border, and
  :func:`tile_members` reports each tile's real footprint coverage so the caller
  can drop the near-empty corner tiles of an irregular mosaic outline.

The resample call itself is fed by :func:`tile_resample_params`, which expresses
each tile as ``crval`` / ``crpix`` / ``rotation`` / ``pixel_scale`` /
``output_shape`` -- the parameter interface of ``jwst`` ``ResampleStep`` -- with
a shared ``crval`` so the projection is identical across every tile.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.wcs import WCS

#: World ``(ra, dec)`` corner vertices of one product, shape ``(K, 2)`` in deg.
Corners = np.ndarray


@dataclass(frozen=True)
class FieldGrid:
    """The canonical output grid of one ``(field, filter)`` mosaic.

    Attributes
    ----------
    wcs : astropy.wcs.WCS
        The full-field TAN projection; tiles are pixel windows into it.
    shape : tuple of int
        Full-field array shape ``(ny, nx)`` in NumPy order.
    crval : tuple of float
        Projection centre ``(ra, dec)`` in degrees (shared by every tile).
    crpix : tuple of float
        1-based reference pixel ``(x, y)`` of ``crval`` in the full grid.
    pixel_scale : float
        Output pixel scale in arcsec/pixel.
    rotation : float
        Output rotation in degrees (0 = north up, east left).
    """

    wcs: "WCS"
    shape: tuple[int, int]
    crval: tuple[float, float]
    crpix: tuple[float, float]
    pixel_scale: float
    rotation: float


@dataclass(frozen=True)
class TileSpec:
    """One rectangular tile, a pixel window of a :class:`FieldGrid`.

    The window includes the ``overlap`` border; ``ix`` / ``iy`` index the tile in
    the grid (column / row). ``x0`` / ``y0`` are the 0-based origin of the window
    in full-field pixels and ``nx`` / ``ny`` its size.
    """

    iy: int
    ix: int
    x0: int
    y0: int
    nx: int
    ny: int


def _circmean_deg(ra: np.ndarray) -> float:
    """Circular mean of right ascensions (degrees), robust to the 0/360 seam."""
    angle = np.deg2rad(np.asarray(ra, dtype=float))
    mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    return float(np.rad2deg(mean) % 360.0)


def field_grid(
    footprints: list[Corners],
    pixel_scale: float,
    *,
    rotation: float = 0.0,
    margin: int = 8,
) -> FieldGrid:
    """Build the canonical output grid covering every footprint.

    A TAN projection is centred on the footprints' mean direction, all corners
    are projected into it, and the array is sized to the resulting pixel bounding
    box (plus a small ``margin``). The reference pixel is shifted so the whole
    field lands at non-negative pixel coordinates.

    Parameters
    ----------
    footprints : list of numpy.ndarray
        One ``(K, 2)`` array of world ``(ra, dec)`` corners per input product.
    pixel_scale : float
        Output pixel scale in arcsec/pixel (the unified mosaic scale).
    rotation : float, optional
        Output rotation in degrees; 0 (default) is north-up, east-left.
    margin : int, optional
        Extra blank pixels added on every side, by default 8.

    Returns
    -------
    FieldGrid

    Raises
    ------
    ValueError
        If ``footprints`` is empty or ``pixel_scale`` is not positive.
    """
    from astropy.wcs import WCS

    if not footprints:
        raise ValueError("field_grid needs at least one footprint.")
    if not pixel_scale > 0:
        raise ValueError(f"pixel_scale must be positive; got {pixel_scale!r}.")

    corners = np.vstack([np.asarray(fp, dtype=float) for fp in footprints])
    cra = _circmean_deg(corners[:, 0])
    cdec = float(np.mean(corners[:, 1]))

    scale_deg = pixel_scale / 3600.0
    rot = np.deg2rad(rotation)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [cra, cdec]
    wcs.wcs.crpix = [1.0, 1.0]
    # North-up, east-left: RA increases to the left (negative first CD column).
    wcs.wcs.cd = scale_deg * np.array(
        [[-np.cos(rot), np.sin(rot)], [np.sin(rot), np.cos(rot)]]
    )

    x, y = wcs.world_to_pixel_values(corners[:, 0], corners[:, 1])
    x0, y0 = float(np.min(x)), float(np.min(y))
    nx = int(np.ceil(np.max(x) - x0)) + 1 + 2 * margin
    ny = int(np.ceil(np.max(y) - y0)) + 1 + 2 * margin
    # Shift so the minimum corner sits at pixel `margin` (crpix is 1-based).
    crpix_x = 1.0 - x0 + margin
    crpix_y = 1.0 - y0 + margin
    wcs.wcs.crpix = [crpix_x, crpix_y]

    return FieldGrid(
        wcs=wcs,
        shape=(ny, nx),
        crval=(cra, cdec),
        crpix=(crpix_x, crpix_y),
        pixel_scale=pixel_scale,
        rotation=rotation,
    )


def _even_bounds(length: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Return overlapping ``(lo, hi)`` windows tiling ``[0, length)`` evenly.

    The axis is split into ``ceil(length / target)`` contiguous *cores* whose
    sizes differ by at most one pixel (the remainder is spread one-per-tile, not
    dumped on the last tile), then each core is padded by ``overlap`` and clipped
    to the axis.
    """
    n = max(1, math.ceil(length / target))
    base, extra = divmod(length, n)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((max(0, start - overlap), min(length, stop + overlap)))
        start = stop
    return bounds


def tile_grid(
    field: FieldGrid, *, target_size: int = 4096, overlap: int = 128
) -> list[TileSpec]:
    """Split a field into evenly-sized overlapping tiles of about ``target_size``.

    Each axis is divided into ``ceil(N / target_size)`` tiles whose core sizes
    differ by at most one pixel -- the remainder is distributed one pixel per
    tile rather than left on a short last tile, so there is never a thin sliver.
    An ``overlap`` border is added on every interior side (clipped at the field
    edges), so adjacent tiles share a margin and a source straddling a core
    boundary is whole in at least one tile.

    Parameters
    ----------
    field : FieldGrid
        The output grid to tile.
    target_size : int, optional
        Approximate interior tile size in pixels, by default 4096.
    overlap : int, optional
        Overlap border in pixels added around each tile core, by default 128.

    Returns
    -------
    list of TileSpec
        Row-major (``iy`` outer, ``ix`` inner) tiles covering the field.

    Raises
    ------
    ValueError
        If ``target_size`` or ``overlap`` is negative / non-positive.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive; got {target_size!r}.")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative; got {overlap!r}.")

    ny, nx = field.shape
    x_bounds = _even_bounds(nx, target_size, overlap)
    y_bounds = _even_bounds(ny, target_size, overlap)

    tiles: list[TileSpec] = []
    for iy, (y_lo, y_hi) in enumerate(y_bounds):
        for ix, (x_lo, x_hi) in enumerate(x_bounds):
            tiles.append(TileSpec(iy, ix, x_lo, y_lo, x_hi - x_lo, y_hi - y_lo))
    return tiles


def tile_members(
    field: FieldGrid, tile: TileSpec, footprints: list[Corners]
) -> tuple[list[int], float]:
    """Return the inputs overlapping ``tile`` and the tile's coverage fraction.

    Membership is a bounding-box overlap test between each footprint (projected
    into field pixels) and the tile window -- a safe superset, since a frame that
    contributes nothing is merely ignored by the resample. Coverage is the
    fraction of tile pixels inside the *union* of the member footprints
    (rasterised), so the caller can drop the near-empty corner tiles of an
    irregular mosaic outline.

    Parameters
    ----------
    field : FieldGrid
        The output grid the tile belongs to.
    tile : TileSpec
        The tile to test.
    footprints : list of numpy.ndarray
        World ``(ra, dec)`` corners per input, in the same order as the inputs.

    Returns
    -------
    members : list of int
        Indices into ``footprints`` whose bounding box overlaps the tile.
    coverage : float
        Fraction of tile area covered by the member footprints' union (0--1).
    """
    members: list[int] = []
    member_px: list[np.ndarray] = []
    tx1, ty1 = tile.x0 + tile.nx, tile.y0 + tile.ny
    for i, fp in enumerate(footprints):
        corners = np.asarray(fp, dtype=float)
        x, y = field.wcs.world_to_pixel_values(corners[:, 0], corners[:, 1])
        if x.max() < tile.x0 or x.min() > tx1 or y.max() < tile.y0 or y.min() > ty1:
            continue
        members.append(i)
        member_px.append(np.column_stack([x - tile.x0, y - tile.y0]))
    return members, _coverage(member_px, tile.ny, tile.nx)


def _coverage(member_px: list[np.ndarray], ny: int, nx: int) -> float:
    """Fraction of an ``ny``-by-``nx`` tile inside the union of polygons.

    ``member_px`` are corner arrays already shifted into tile-local pixels.
    """
    if not member_px or ny <= 0 or nx <= 0:
        return 0.0
    from skimage.draw import polygon

    mask = np.zeros((ny, nx), dtype=bool)
    for px in member_px:
        rr, cc = polygon(px[:, 1], px[:, 0], shape=(ny, nx))
        mask[rr, cc] = True
    return float(mask.mean())


def tile_resample_params(field: FieldGrid, tile: TileSpec) -> dict[str, object]:
    """Return the ``jwst`` ``ResampleStep`` output-grid kwargs for ``tile``.

    All tiles share ``crval`` / ``rotation`` / ``pixel_scale`` (one projection);
    only ``crpix`` (shifted by the tile origin) and ``output_shape`` differ, so
    the tiles are exactly pixel-aligned and stitch seamlessly.

    Parameters
    ----------
    field : FieldGrid
        The output grid.
    tile : TileSpec
        The tile to resample.

    Returns
    -------
    dict
        ``crval`` (``[ra, dec]``), ``crpix`` (``[x, y]``, tile-local),
        ``rotation``, ``pixel_scale`` and ``output_shape`` (``[nx, ny]``).

    Notes
    -----
    ``field.crpix`` is the astropy/FITS 1-based reference pixel; ``jwst``
    ``ResampleStep`` takes ``crpix`` 0-based, so the value is converted with a
    ``-1`` (verified against the resampled output WCS -- without it every tile is
    offset by exactly one pixel).
    """
    cx, cy = field.crpix
    return {
        "crval": [field.crval[0], field.crval[1]],
        "crpix": [cx - 1 - tile.x0, cy - 1 - tile.y0],
        "rotation": field.rotation,
        "pixel_scale": field.pixel_scale,
        "output_shape": [tile.nx, tile.ny],
    }
