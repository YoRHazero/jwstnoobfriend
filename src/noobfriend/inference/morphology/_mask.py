"""Auto-derive fit masks that exclude contaminating neighbours.

The contamination model is conservative: user masks and hard fit-radius cuts are
honoured first, then automatic neighbour masks are added only where they do not
overlap a target-protection region. The protection region keeps the target core,
its seed segment when detected, and a guarded noobase-grown target aperture from
being removed by neighbour segmentation.

When multiple bands are fit, neighbour detections can be projected between native
grids through the per-band affine geometry. This catches neighbours seen in a deep
band before they contaminate a shallow or dropout band. This module is internal.
"""

from __future__ import annotations

from typing import Any

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


def native_offset_grids(geom: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return native-pixel centre east/north offset grids for a band geometry."""
    rows, cols = geom.native_shape
    o = int(geom.oversample)
    e = np.asarray(geom.e_grid, dtype=float).reshape(rows, o, cols, o)
    n = np.asarray(geom.n_grid, dtype=float).reshape(rows, o, cols, o)
    return e.mean(axis=(1, 3)), n.mean(axis=(1, 3))


def radial_exclude_mask(geom: Any, radius_arcsec: float) -> np.ndarray:
    """Return ``True`` outside a sky-frame radius around the target centre."""
    if radius_arcsec <= 0:
        raise ValueError(f"fit_radius_arcsec must be > 0, got {radius_arcsec}.")
    e, n = native_offset_grids(geom)
    return e * e + n * n > radius_arcsec * radius_arcsec


def target_protection_mask(
    data: np.ndarray,
    error: np.ndarray | None,
    geom: Any,
    *,
    nsigma: float = 2.0,
    core_radius_arcsec: float,
    max_radius_arcsec: float | None = None,
) -> np.ndarray:
    """Return pixels automatic neighbour masking is not allowed to exclude.

    The mask is intentionally a protection layer, not the final fit region. It
    combines a small radius around the target centre, the segmentation label under
    the target seed, and a noobase-grown target aperture when the seed source is
    detected. Growth failures fall back to the geometric/segment protection.
    """
    from noobfriend.core.imgutils import segment

    protect = ~radial_exclude_mask(geom, core_radius_arcsec)
    rows, cols = data.shape
    tx, ty = int(round(geom.pix_center[0])), int(round(geom.pix_center[1]))
    if not (0 <= tx < cols and 0 <= ty < rows):
        return _cap_protection(protect, geom, max_radius_arcsec)

    labels = segment(data, nsigma=nsigma)
    target_label = int(labels[ty, tx])
    if target_label > 0:
        protect |= labels == target_label
        try:
            from noobfriend.extraction.photometry._aperture import grow_aperture_mask

            grown = grow_aperture_mask(
                data,
                seed_xy=geom.pix_center,
                error=error,
                label_map=labels,
                allow_background=True,
            )
        except ValueError:
            pass
        else:
            protect |= grown.mask
    return _cap_protection(protect, geom, max_radius_arcsec)


def neighbour_mask_band(
    data: np.ndarray,
    geom: Any,
    protect: np.ndarray,
    *,
    nsigma: float = 2.0,
    dilate: int = 2,
) -> np.ndarray:
    """Return source-segment pixels that are not the target/protected region."""
    from noobfriend.core.imgutils import segment

    labels = segment(data, nsigma=nsigma)
    rows, cols = data.shape
    tx, ty = int(round(geom.pix_center[0])), int(round(geom.pix_center[1]))
    other = labels != 0
    if 0 <= tx < cols and 0 <= ty < rows:
        target_label = int(labels[ty, tx])
        if target_label > 0:
            other &= labels != target_label
    other &= ~protect
    if dilate > 0 and other.any():
        from scipy import ndimage as ndi

        other = ndi.binary_dilation(other, iterations=int(dilate))
        other &= ~protect
    return other


def project_mask_to_geometry(
    mask: np.ndarray,
    source_geom: Any,
    target_geom: Any,
    *,
    dilate: int = 1,
) -> np.ndarray:
    """Project a native mask between band geometries using their affine frames."""
    mask = np.asarray(mask, dtype=bool)
    if source_geom is target_geom:
        out = mask.copy()
    else:
        e, n = native_offset_grids(source_geom)
        rows, cols = target_geom.native_shape
        out = np.zeros((rows, cols), dtype=bool)
        if mask.any():
            offsets = np.vstack([e[mask], n[mask]])
            dxdy = np.asarray(target_geom.affine, dtype=float) @ offsets
            x = np.rint(target_geom.pix_center[0] + dxdy[0]).astype(int)
            y = np.rint(target_geom.pix_center[1] + dxdy[1]).astype(int)
            inside = (0 <= x) & (x < cols) & (0 <= y) & (y < rows)
            out[y[inside], x[inside]] = True
    if dilate > 0 and out.any():
        from scipy import ndimage as ndi

        out = ndi.binary_dilation(out, iterations=int(dilate))
    return out


def _cap_protection(
    protect: np.ndarray, geom: Any, max_radius_arcsec: float | None
) -> np.ndarray:
    """Keep protection from extending beyond the hard fit radius."""
    if max_radius_arcsec is None:
        return protect
    return protect & ~radial_exclude_mask(geom, max_radius_arcsec)
