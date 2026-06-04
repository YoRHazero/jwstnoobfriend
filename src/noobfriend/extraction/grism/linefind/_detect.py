"""Stage 3: turn band-pass SNR maps into deblended line candidates.

Collapses the per-scale SNR stack to one detection map, finds seed peaks above
a threshold (``skimage.feature.peak_local_max``), grows and deblends their
footprints with a marker-controlled watershed, and measures each candidate's
SNR-weighted RMS extent along and across the (known) dispersion axis. It does
not reject anything: the extents are *features* for the downstream
discrimination / threshold-calibration steps, in keeping with the high-recall
design.

This module is internal to ``noobfriend.extraction.grism.linefind``.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from noobfriend.extraction.grism.linefind._filter import BandPass


@dataclass(frozen=True)
class Candidate:
    """One line candidate: a seed peak plus its deblended footprint shape.

    Attributes
    ----------
    y, x : int
        Detector pixel of the seed (the SNR peak).
    snr : float
        Peak band-pass SNR (the detection significance, not a flux).
    scale : tuple[float, float]
        The ``(sigma_cross, sigma_disp)`` scale (pixels) that peaked at the seed.
    npix : int
        Number of pixels in the deblended footprint.
    disp_extent, cross_extent : float
        SNR-weighted RMS extent of the footprint along and across the
        dispersion axis (pixels). A real line is compact along dispersion; a
        continuum residual is extended -- but this is a *feature*, not a cut.
    """

    y: int
    x: int
    snr: float
    scale: tuple[float, float]
    npix: int
    disp_extent: float
    cross_extent: float


def detect(
    bp: BandPass,
    *,
    dispersion_axis: int,
    threshold: float,
    grow_threshold: float | None = None,
    min_distance: int = 3,
) -> list[Candidate]:
    """Detect line candidates in a band-pass SNR stack.

    Parameters
    ----------
    bp : BandPass
        Per-scale band-pass SNR, from
        :func:`~noobfriend.extraction.grism.linefind._filter.band_pass_snr`.
    dispersion_axis : int
        Array axis the grism disperses along (``1`` row / ``0`` column).
    threshold : float
        Seed (peak) SNR threshold. Should come from a trials-calibrated
        false-alarm rate, not a fixed 5-sigma (see the design notes).
    grow_threshold : float or None, optional
        Lower SNR threshold to which seeds are grown and deblended. ``None``
        (default) uses ``0.5 * threshold``.
    min_distance : int, optional
        Minimum separation (pixels) between seed peaks, by default ``3``.

    Returns
    -------
    list[Candidate]
        Candidates sorted by descending peak SNR; empty if none pass.
    """
    filled = np.where(np.isfinite(bp.snr), bp.snr, -np.inf)
    snr_max = filled.max(axis=0)
    finite = np.isfinite(snr_max)

    seeds = peak_local_max(
        snr_max,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )
    if seeds.size == 0:
        return []

    gt = 0.5 * threshold if grow_threshold is None else grow_threshold
    mask = finite & (snr_max > gt)

    seed_mask = np.zeros(snr_max.shape, dtype=bool)
    seed_mask[tuple(seeds.T)] = True
    markers, n_lab = ndi.label(seed_mask)
    landscape = -np.where(finite, snr_max, 0.0)
    labels = watershed(landscape, markers, mask=mask)

    # SNR-weighted, axis-aligned footprint moments for every label, vectorized.
    weight = np.where(finite, np.clip(snr_max, 0.0, None), 0.0).ravel()
    yy, xx = np.indices(snr_max.shape)
    disp = (xx if dispersion_axis == 1 else yy).astype(np.float64).ravel()
    cross = (yy if dispersion_axis == 1 else xx).astype(np.float64).ravel()
    lab = labels.ravel()
    size = int(n_lab) + 1
    sw = np.bincount(lab, weight, size)
    swd = np.bincount(lab, weight * disp, size)
    swd2 = np.bincount(lab, weight * disp * disp, size)
    swc = np.bincount(lab, weight * cross, size)
    swc2 = np.bincount(lab, weight * cross * cross, size)
    npix = np.bincount(lab, None, size)

    safe = np.where(sw > 0, sw, 1.0)
    disp_var = swd2 / safe - (swd / safe) ** 2
    cross_var = swc2 / safe - (swc / safe) ** 2
    disp_ext = np.where(sw > 0, np.sqrt(np.clip(disp_var, 0.0, None)), 0.0)
    cross_ext = np.where(sw > 0, np.sqrt(np.clip(cross_var, 0.0, None)), 0.0)

    candidates: list[Candidate] = []
    for sy, sx in seeds:
        ll = int(labels[sy, sx])
        col = bp.snr[:, sy, sx]
        scale_idx = int(np.nanargmax(col)) if np.isfinite(col).any() else 0
        candidates.append(
            Candidate(
                y=int(sy),
                x=int(sx),
                snr=float(snr_max[sy, sx]),
                scale=bp.scales[scale_idx],
                npix=int(npix[ll]),
                disp_extent=float(disp_ext[ll]),
                cross_extent=float(cross_ext[ll]),
            )
        )
    candidates.sort(key=lambda c: c.snr, reverse=True)
    return candidates
