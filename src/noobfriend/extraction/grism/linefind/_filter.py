"""Stage 1+2: compensated band-pass matched filtering and per-pixel SNR.

An emission line is compact along the dispersion axis while the continuum it
sits on is extended along it. This module turns that contrast into a detection
statistic: it band-passes the frame with a zero-sum, background-compensated
matched filter (a Gaussian across dispersion times a Mexican-hat along it),
then divides by the noise of that same filtered quantity -- propagated
per-pixel from the error map -- to yield a band-pass SNR map per line scale.

Each line scale is a ``(sigma_cross, sigma_disp)`` pair: the line's spatial
extent across dispersion and its (spatial-plus-spectral) extent along it. The
band-pass is the difference of two unit-sum smoothings (the line scale minus an
along-dispersion continuum scale) that share the cross-dispersion profile, so
the smooth continuum cancels automatically -- no separate background step. Both
smoothings are evaluated NaN-aware (noobase's renormalized correlation), so
DQ-masked pixels are treated as missing rather than poisoning their
neighbourhood.

This module is internal to ``noobfriend.extraction.grism.linefind``; it works
purely on arrays (the dispersion axis is passed as an index) with no WCS or
source knowledge.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from noobase.convolve import conv2d_renorm, gaussian1d

from noobfriend.extraction.grism._array import _native


@dataclass(frozen=True)
class BandPass:
    """Per-scale band-pass maps and their noise-normalized SNR.

    Attributes
    ----------
    snr : numpy.ndarray
        Band-pass SNR, shape ``(n_scale, H, W)``; ``NaN`` where no valid input
        contributed. This is the detection statistic (a significance, not a
        flux): peaks above a calibrated threshold are line candidates.
    signal : numpy.ndarray
        The band-pass (continuum-subtracted) amplitude, same shape as ``snr``.
        Carried for deblending / QA only; it is *not* a calibrated flux and is
        biased by the filter's negative sidelobes -- do not photometer on it.
    scales : tuple[tuple[float, float], ...]
        The ``(sigma_cross, sigma_disp)`` pixel sigma pairs, one per plane of
        ``snr`` / ``signal``.
    """

    snr: np.ndarray
    signal: np.ndarray
    scales: tuple[tuple[float, float], ...]


def _separable_2d(axis0: np.ndarray, axis1: np.ndarray) -> np.ndarray:
    """Outer-product two 1-D kernels into a separable 2-D kernel."""
    return np.outer(axis0, axis1)


def _center_pad(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center a smaller (odd) kernel inside a zero array of ``shape``."""
    out = np.zeros(shape, dtype=kernel.dtype)
    h, w = kernel.shape
    oh, ow = (shape[0] - h) // 2, (shape[1] - w) // 2
    out[oh : oh + h, ow : ow + w] = kernel
    return out


def _band_pass_kernels(
    sigma_cross: float,
    sigma_disp: float,
    continuum_sigma_disp: float,
    dispersion_axis: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the line, continuum, and zero-sum difference kernels.

    The line kernel is a 2-D Gaussian, ``sigma_cross`` across dispersion and
    ``sigma_disp`` along it; the continuum kernel shares that same
    cross-dispersion profile but is broadened to ``continuum_sigma_disp`` along
    dispersion, so it estimates the local continuum under the line. Both are
    unit-sum, so their difference is zero-sum; sharing the cross profile keeps
    it zero-sum *along dispersion* (which is what cancels a flat continuum).

    Parameters
    ----------
    sigma_cross, sigma_disp : float
        The line's Gaussian sigmas (pixels) across and along dispersion.
    continuum_sigma_disp : float
        Along-dispersion continuum Gaussian sigma (pixels); requires
        ``continuum_sigma_disp > sigma_disp``.
    dispersion_axis : int
        Array axis along which the grism disperses (``1`` for row dispersion /
        GRISMR, ``0`` for column dispersion / GRISMC).
    dtype : numpy.dtype
        Output kernel dtype (matched to the image for noobase's correlator).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(k_line, k_continuum, g)`` all of the continuum kernel's shape, with
        ``g = k_line - k_continuum`` the zero-sum band-pass kernel.
    """
    g_cross = gaussian1d(sigma_cross).astype(dtype)
    g_disp = gaussian1d(sigma_disp).astype(dtype)
    g_cont = gaussian1d(continuum_sigma_disp).astype(dtype)
    if dispersion_axis == 1:
        k_line = _separable_2d(g_cross, g_disp)
        k_cont = _separable_2d(g_cross, g_cont)
    else:
        k_line = _separable_2d(g_disp, g_cross)
        k_cont = _separable_2d(g_cont, g_cross)
    k_line = _center_pad(k_line, k_cont.shape)
    g = k_line - k_cont
    return (
        np.ascontiguousarray(k_line),
        np.ascontiguousarray(k_cont),
        np.ascontiguousarray(g),
    )


def band_pass_snr(
    data: np.ndarray,
    error: np.ndarray,
    *,
    dispersion_axis: int,
    line_scales: Sequence[tuple[float, float]],
    continuum_sigma_disp: float,
    coverage_floor: float = 0.8,
) -> BandPass:
    """Compute per-scale band-pass SNR maps from a grism frame.

    For each ``(sigma_cross, sigma_disp)`` scale the frame is band-passed
    (continuum removed, matched to a line of that 2-D size) and divided by the
    per-pixel propagated noise of the band-pass. Invalid pixels (non-finite
    data, or non-finite / non-positive error) are treated as missing and
    excluded from every neighbourhood sum.

    Parameters
    ----------
    data, error : numpy.ndarray
        2-D frame and its 1-sigma error, same shape. Coerced to native-endian
        float; big-endian JWST cal arrays are accepted.
    dispersion_axis : int
        Array axis along which the grism disperses (``1`` row / ``0`` column).
    line_scales : Sequence[tuple[float, float]]
        ``(sigma_cross, sigma_disp)`` pixel sigma pairs to scan -- the line's
        spatial size across dispersion and its (spatial-plus-spectral) size
        along it. Must be non-empty.
    continuum_sigma_disp : float
        Along-dispersion Gaussian sigma (pixels) of the continuum estimate;
        must exceed every ``sigma_disp`` (continuum is smooth on this scale,
        the line is not).
    coverage_floor : float, optional
        Minimum fraction of the band-pass kernel weight that must land on valid
        pixels for a pixel's SNR to be trusted, by default ``0.8``. Pixels
        below it -- a frame-border margin and rings around DQ masks / snowballs,
        where the zero-sum compensation and noise estimate break down -- are set
        to ``NaN``.

    Returns
    -------
    BandPass
        Per-scale ``snr`` and ``signal`` stacks plus the ``scales`` used.

    Raises
    ------
    ValueError
        If the inputs are not 2-D of equal shape, ``line_scales`` is empty, or
        any ``sigma_disp`` is not strictly below ``continuum_sigma_disp``.

    Notes
    -----
    The noise is propagated through the nominal band-pass kernel ``g`` as
    ``sum(g**2 * variance)`` over valid taps (recovered from the renormalized
    correlation's ``value * weight``); near DQ masks this is a mild
    approximation, exact in the unmasked interior. ``coverage_floor`` masks the
    border margin (and rings around DQ masks) where the zero-sum compensation
    and the noise estimate break down -- without it the renormalized
    correlation yields spurious huge SNR at the detector edges.
    """
    if not line_scales:
        raise ValueError("`line_scales` must be non-empty.")
    d = _native(data)
    e = _native(error)
    if d.ndim != 2 or d.shape != e.shape:
        raise ValueError("`data` and `error` must be 2-D arrays of equal shape.")
    dtype = d.dtype

    valid = np.isfinite(d) & np.isfinite(e) & (e > 0)
    d = np.where(valid, d, np.nan).astype(dtype)
    variance = np.where(valid, e**2, np.inf).astype(dtype)

    snr_planes: list[np.ndarray] = []
    signal_planes: list[np.ndarray] = []
    for sigma_cross, sigma_disp in line_scales:
        if not continuum_sigma_disp > sigma_disp:
            raise ValueError(
                f"continuum_sigma_disp ({continuum_sigma_disp}) must exceed every "
                f"sigma_disp (got {sigma_disp})."
            )
        k_line, k_cont, g = _band_pass_kernels(
            float(sigma_cross),
            float(sigma_disp),
            float(continuum_sigma_disp),
            dispersion_axis,
            dtype,
        )
        line_smooth, _ = conv2d_renorm(d, k_line)
        cont_smooth, _ = conv2d_renorm(d, k_cont)
        signal = line_smooth - cont_smooth

        g_sq = np.ascontiguousarray(g**2)
        nz_val, nz_wt = conv2d_renorm(variance, g_sq)
        noise = np.sqrt(nz_val * nz_wt)
        coverage = nz_wt / float(g_sq.sum())
        reliable = np.isfinite(noise) & (noise > 0) & (coverage >= coverage_floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(reliable, signal / noise, np.nan)
        snr_planes.append(snr)
        signal_planes.append(np.where(reliable, signal, np.nan))

    return BandPass(
        snr=np.stack(snr_planes),
        signal=np.stack(signal_planes),
        scales=tuple((float(c), float(d)) for c, d in line_scales),
    )
