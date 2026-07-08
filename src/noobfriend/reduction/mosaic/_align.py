"""Deep-self-catalog absolute astrometric alignment for stage-3 mosaics.

This replaces jwst ``TweakRegStep`` on the FRESCO-scale groups it chokes on
(``align_wcs`` orders images by O(N^2) spherical-polygon overlap, grows an
unbounded reference catalog, and blind-searches an initial shift per frame --
all single-threaded). Here the relative and absolute ties are decoupled, which
is both faster (linear, parallel) and robust to GAIA sparsity:

1. **Deep reference catalog** -- pool every frame's detections (raw WCS),
   project into a field tangent plane, friends-of-friends cluster (dropping
   same-frame links), and keep one centroid per source seen in >= ``min_frames``
   frames. Built once and frozen (not grown incrementally), it is dense
   (thousands of galaxies + stars) where GAIA is sparse (~tens of stars).
2. **Per-frame relative fit** -- align each frame independently to the frozen
   catalog by a sigma-clipped weighted affine in the tangent plane. Every frame
   gets dozens of anchors, so none is starved (GAIA-only would leave frames
   with 1-2 GAIA stars unaligned).
3. **Absolute GAIA tie** -- tie the whole catalog to GAIA with one global affine
   (optional). Pooled across all frames the catalog has plenty of GAIA matches
   even when each frame has almost none, so the sparse-per-frame problem
   dissolves.

The per-frame result is the relative fit composed with the global tie, returned
as a :class:`~noobfriend.core.wcs.TangentCorrection` -- the single alignment
parameter set that drives both the compiled NoobWCS fast path (coadd) and the
astropy-core gwcs surgery (interop). This module is pure ``numpy`` / ``scipy``
plus :mod:`noobfriend.core.wcs`: it takes source coordinates, not files, so it
carries no jwst / navigation dependency.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from astropy.modeling.models import RotateCelestial2Native, Sky2Pix_Gnomonic
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from noobfriend.core.wcs import TangentCorrection

__all__ = ["FrameSources", "align_group"]


@dataclass(frozen=True)
class FrameSources:
    """One frame's detected point sources, the input unit of :func:`align_group`.

    Attributes
    ----------
    frame_id : str
        Identifier the returned correction is keyed by (e.g. a NooBook id).
    ra, dec : numpy.ndarray
        Source world coordinates in degrees (same length).
    weight : numpy.ndarray
        Per-source fit weight (e.g. ``snr ** 2``), same length; larger weights
        pull the affine fit harder.
    """

    frame_id: str
    ra: np.ndarray
    dec: np.ndarray
    weight: np.ndarray


def align_group(
    frames: Sequence[FrameSources],
    *,
    gaia: tuple[np.ndarray, np.ndarray] | None = None,
    cluster_tol_arcsec: float = 0.15,
    match_tol_arcsec: float = 0.4,
    gaia_tol_arcsec: float = 0.3,
    n_iter: int = 2,
    nclip: int = 3,
    clip_sigma: float = 3.0,
    min_frames: int = 2,
    min_match: int = 4,
) -> dict[str, TangentCorrection]:
    """Align a group of frames to a deep self-catalog and (optionally) GAIA.

    Parameters
    ----------
    frames : sequence of FrameSources
        The group's frames, each carrying its detected point sources. Frames
        with too few matches to the reference are returned with an identity
        correction (left on their input WCS).
    gaia : tuple of (numpy.ndarray, numpy.ndarray), optional
        Clean GAIA ``(ra, dec)`` in degrees for the absolute tie. When omitted,
        only the relative (self-catalog) alignment is applied and the group's
        absolute frame is the arbitrary zero point of the pooled catalog.
    cluster_tol_arcsec : float, optional
        Friends-of-friends linking length for grouping detections of the same
        source across frames, by default 0.15.
    match_tol_arcsec : float, optional
        Nearest-neighbour tolerance for matching a frame's sources to the
        reference centroids, by default 0.4.
    gaia_tol_arcsec : float, optional
        Nearest-neighbour tolerance for matching catalog centroids to GAIA, by
        default 0.3.
    n_iter : int, optional
        Number of align passes; each rebuilds the centroids from the corrected
        positions, by default 2 (1-2 converge).
    nclip : int, optional
        Sigma-clipping iterations in each affine fit, by default 3.
    clip_sigma : float, optional
        Sigma-clipping threshold in the affine fit, by default 3.0.
    min_frames : int, optional
        Minimum distinct frames a cluster must appear in to become a reference
        centroid, by default 2.
    min_match : int, optional
        Minimum matched sources for a frame's affine fit; below this the frame
        keeps an identity correction, by default 4.

    Returns
    -------
    dict of str to TangentCorrection
        One correction per input frame, keyed by ``frame_id``. Frames that
        could not be aligned carry an identity correction about the field
        fiducial.

    Raises
    ------
    ValueError
        If ``frames`` is empty.

    Notes
    -----
    The tangent plane uses astropy's exact gnomonic projection about the field
    fiducial (the median of the pooled source coordinates), matching
    :meth:`TangentCorrection.to_models`, so the fitted affine transfers into the
    correction with no projection-convention conversion.
    """
    if not frames:
        raise ValueError("align_group needs at least one frame.")

    ra = np.concatenate([np.asarray(f.ra, float) for f in frames])
    dec = np.concatenate([np.asarray(f.dec, float) for f in frames])
    weight = np.concatenate([np.asarray(f.weight, float) for f in frames])
    sizes = [len(f.ra) for f in frames]
    frame_of = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)])
    bounds = np.cumsum([0, *sizes])

    fiducial = (float(np.median(ra)), float(np.median(dec)))
    xy0 = _project(ra, dec, fiducial)

    identity = TangentCorrection(fiducial=fiducial)
    labels = _cluster(xy0, frame_of, cluster_tol_arcsec)
    ref_labels = [
        lab
        for lab in np.unique(labels)
        if len(np.unique(frame_of[labels == lab])) >= min_frames
    ]
    if not ref_labels:  # nothing to align against
        return {f.frame_id: identity for f in frames}

    # Stage 1 + 2: per-frame relative affine to the (iterated) frozen catalog.
    xy = xy0.copy()
    rel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for iteration in range(max(n_iter, 1)):
        centroids = np.array([xy[labels == lab].mean(axis=0) for lab in ref_labels])
        rel = {}
        for k in range(len(frames)):
            sl = slice(bounds[k], bounds[k + 1])
            im = xy0[sl]
            im_idx, ref_idx = _match(centroids, im, match_tol_arcsec)
            if len(im_idx) < min_match:
                continue
            m, s = _fit_affine(
                im[im_idx], centroids[ref_idx], weight[sl][im_idx], nclip, clip_sigma
            )
            rel[k] = (m, s)
            if iteration < n_iter - 1:
                xy[sl] = im @ m.T + s

    # Stage 3: one global affine tying the corrected catalog to GAIA.
    tie = (np.eye(2), np.zeros(2))
    if gaia is not None:
        final = np.array([xy[labels == lab].mean(axis=0) for lab in ref_labels])
        member_w = np.array(
            [len(np.unique(frame_of[labels == lab])) for lab in ref_labels], float
        )
        gaia_xy = _project(
            np.asarray(gaia[0], float), np.asarray(gaia[1], float), fiducial
        )
        cat_idx, gaia_idx = _match(gaia_xy, final, gaia_tol_arcsec)
        if len(cat_idx) >= 3:
            tie = _fit_affine(
                final[cat_idx], gaia_xy[gaia_idx], member_w[cat_idx], nclip, clip_sigma
            )

    m_g, s_g = tie
    out: dict[str, TangentCorrection] = {}
    for k, f in enumerate(frames):
        if k not in rel:
            out[f.frame_id] = identity
            continue
        m_f, s_f = rel[k]
        m = m_g @ m_f
        s = m_g @ s_f + s_g
        out[f.frame_id] = TangentCorrection(
            fiducial=fiducial,
            matrix=(tuple(m[0]), tuple(m[1])),
            offset=(float(s[0]), float(s[1])),
        )
    return out


def _project(
    ra: np.ndarray, dec: np.ndarray, fiducial: tuple[float, float]
) -> np.ndarray:
    """Gnomonic-project ``(ra, dec)`` about ``fiducial``; ``(N, 2)`` in degrees.

    Uses the same ``RotateCelestial2Native | Sky2Pix_Gnomonic`` chain as
    :meth:`TangentCorrection.to_models`, so a fit in this plane transfers to the
    correction verbatim.
    """
    proj = RotateCelestial2Native(fiducial[0], fiducial[1], 180.0) | Sky2Pix_Gnomonic()
    x, y = proj(np.asarray(ra, float), np.asarray(dec, float))
    return np.column_stack([np.asarray(x, float), np.asarray(y, float)])


def _match(
    ref_xy: np.ndarray, im_xy: np.ndarray, tol_arcsec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour match within ``tol_arcsec``.

    Returns
    -------
    im_idx, ref_idx : numpy.ndarray
        Index arrays into ``im_xy`` and ``ref_xy`` of the mutually-nearest
        pairs (each side's nearest is the other), within the tolerance.
    """
    if len(ref_xy) == 0 or len(im_xy) == 0:
        return np.empty(0, int), np.empty(0, int)
    tol = tol_arcsec / 3600.0
    tree_r, tree_i = cKDTree(ref_xy), cKDTree(im_xy)
    dist, nn_r = tree_r.query(im_xy, distance_upper_bound=tol)
    _, nn_i = tree_i.query(ref_xy)
    im_idx, ref_idx = [], []
    for i, (d, r) in enumerate(zip(dist, nn_r)):
        if np.isfinite(d) and r < len(ref_xy) and nn_i[r] == i:
            im_idx.append(i)
            ref_idx.append(r)
    return np.asarray(im_idx, int), np.asarray(ref_idx, int)


def _fit_affine(
    im_xy: np.ndarray,
    ref_xy: np.ndarray,
    weight: np.ndarray,
    nclip: int,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clipped weighted affine mapping ``im_xy`` -> ``ref_xy``.

    Solves ``ref ~= M @ im + s`` (a 2x2 matrix and a shift) by weighted least
    squares, then re-solves ``nclip`` times dropping pairs beyond ``clip_sigma``
    of the residual RMS.

    Returns
    -------
    matrix, offset : numpy.ndarray
        The ``(2, 2)`` affine matrix and length-2 offset.
    """
    keep = np.ones(len(im_xy), bool)
    matrix, offset = np.eye(2), np.zeros(2)
    for _ in range(nclip + 1):
        pts, sw = im_xy[keep], np.sqrt(weight[keep])
        design = np.zeros((2 * len(pts), 6))
        design[0::2, 0], design[0::2, 1], design[0::2, 4] = (
            pts[:, 0] * sw,
            pts[:, 1] * sw,
            sw,
        )
        design[1::2, 2], design[1::2, 3], design[1::2, 5] = (
            pts[:, 0] * sw,
            pts[:, 1] * sw,
            sw,
        )
        rhs = np.empty(2 * len(pts))
        rhs[0::2], rhs[1::2] = ref_xy[keep, 0] * sw, ref_xy[keep, 1] * sw
        p, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        matrix, offset = np.array([[p[0], p[1]], [p[2], p[3]]]), np.array([p[4], p[5]])
        resid = np.linalg.norm(ref_xy - (im_xy @ matrix.T + offset), axis=1)
        new_keep = resid < clip_sigma * np.std(resid[keep])
        if new_keep.sum() == keep.sum() or new_keep.sum() < 4:
            break
        keep = new_keep
    return matrix, offset


def _cluster(xy: np.ndarray, frame_of: np.ndarray, tol_arcsec: float) -> np.ndarray:
    """Friends-of-friends label the pooled points; same-frame links are dropped.

    A cluster gathers detections within ``tol_arcsec`` (transitively) that come
    from *different* frames -- two detections of one frame cannot be one source,
    so linking them is suppressed. Returns a connected-component label per point.
    """
    tree = cKDTree(xy)
    pairs = np.array(list(tree.query_pairs(tol_arcsec / 3600.0)), dtype=int)
    if len(pairs) == 0:
        return np.arange(len(xy))
    pairs = pairs[frame_of[pairs[:, 0]] != frame_of[pairs[:, 1]]]
    if len(pairs) == 0:
        return np.arange(len(xy))
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy))
    )
    _, labels = connected_components(graph, directed=False)
    return labels
