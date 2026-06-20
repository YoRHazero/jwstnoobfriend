"""Render built PSFs to detector-sampled images and convolve with them.

A :class:`~noobfriend.extraction.psf.CorePsf` / :class:`EffectivePsf` is held on
an *oversampled* grid (plus, for the hybrid, a native wing). To use it -- as a
convolution kernel, or to build a PSF-matching kernel -- it must first be
sampled onto a native detector-pixel grid. :func:`render_core` and
:func:`render_effective` do exactly that, the latter reproducing the noobase
sampling contract ``recon(r) = (1 - f_wing(r)) * core_sample(r) + wing_sample(r)``
with a raised-cosine ``f_wing`` reconstructed from the stitch geometry.

For PSF-matching photometry, render both bands' PSFs onto the same grid, build a
matching kernel (e.g. ``photutils.psf.matching.create_matching_kernel``), and
:func:`convolve` the sharper image with it.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.psf._build import EffectivePsf


def _decimate_core(core_plane: np.ndarray, oversample: int) -> np.ndarray:
    """Sample an oversampled core at native pixel centres (central sub-pixel phase).

    The oversampled core is centred, so the native pixel value of a perfectly
    centred source is the central phase of each ``oversample``-wide block -- i.e.
    the samples at offset ``oversample // 2``.
    """
    o = int(oversample)
    return np.ascontiguousarray(core_plane[o // 2 :: o, o // 2 :: o], dtype=float)


def _resize_centered(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre ``arr`` in a ``size`` square, zero-padding or cropping symmetrically."""
    s = arr.shape[0]
    if size == s:
        return np.array(arr, dtype=float, copy=True)
    out = np.zeros((size, size), dtype=float)
    if size > s:
        off = (size - s) // 2
        out[off : off + s, off : off + s] = arr
    else:
        off = (s - size) // 2
        out[:] = arr[off : off + size, off : off + size]
    return out


def _feather(size: int, match_radius: float, feather_width: float) -> np.ndarray:
    """Raised-cosine wing weight ``f_wing(r)`` on a ``size`` square (centre = middle).

    ``0`` inside ``match_radius - feather_width/2`` (pure core), ``1`` outside
    ``match_radius + feather_width/2`` (pure wing), a raised cosine between.
    """
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - c, xx - c)
    inner = match_radius - feather_width / 2.0
    outer = match_radius + feather_width / 2.0
    if outer > inner:
        t = np.clip((r - inner) / (outer - inner), 0.0, 1.0)
        f_wing = 0.5 - 0.5 * np.cos(np.pi * t)
    else:  # degenerate (zero feather width): a hard step at match_radius
        f_wing = (r >= match_radius).astype(float)
    return np.where(r <= inner, 0.0, np.where(r >= outer, 1.0, f_wing))


def _normalize(image: np.ndarray) -> np.ndarray:
    """Scale to unit total sum (flux-preserving convolution kernel)."""
    total = float(image.sum())
    if not np.isfinite(total) or total == 0.0:
        raise ValueError(f"cannot normalise a PSF with total sum {total!r}.")
    return image / total


def render_core(
    core: np.ndarray,
    oversample: int,
    *,
    size: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Render an oversampled core onto a native detector-pixel grid.

    Parameters
    ----------
    core : numpy.ndarray
        ``(oversample * s, oversample * s)`` oversampled core (e.g.
        :attr:`CorePsf.core`).
    oversample : int
        The core's oversampling factor.
    size : int, optional
        Edge length (native pixels) of the output, centred and zero-padded or
        cropped. Defaults to the native core size ``s``.
    normalize : bool, default True
        Scale the result to unit total sum (a flux-preserving kernel).

    Returns
    -------
    numpy.ndarray
        The native-resolution core image.
    """
    native = _decimate_core(np.asarray(core, dtype=float), oversample)
    image = native if size is None else _resize_centered(native, int(size))
    return _normalize(image) if normalize else image


def render_effective(
    psf: "EffectivePsf",
    *,
    size: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Render a hybrid effective PSF onto a native detector-pixel grid.

    Reproduces the noobase sampling contract: the oversampled core plane is
    sampled to native pixels and down-weighted by ``1 - f_wing`` (a raised
    cosine reconstructed from ``match_radius`` / ``feather_width``), and the
    native wing plane (which already carries ``f_wing``) is added.

    Parameters
    ----------
    psf : EffectivePsf
        The hybrid PSF to render.
    size : int, optional
        Edge length (native pixels) of the output, centred and zero-padded or
        cropped. Defaults to ``psf.wing_size`` (the full native support).
    normalize : bool, default True
        Scale the result to unit total sum. When ``False`` the result keeps the
        encircled-energy gauge (its integral within ``ee_aperture_radius`` is
        ~1), matching noobase's normalisation.

    Returns
    -------
    numpy.ndarray
        The native-resolution hybrid PSF image.
    """
    native_core = _decimate_core(
        np.asarray(psf.core_plane, dtype=float), psf.oversample
    )
    wing = np.asarray(psf.wing, dtype=float)
    edge = wing.shape[0] if size is None else int(size)
    core_grid = _resize_centered(native_core, edge)
    wing_grid = _resize_centered(wing, edge)
    f_wing = _feather(edge, psf.match_radius, psf.feather_width)
    recon = (1.0 - f_wing) * core_grid + wing_grid
    return _normalize(recon) if normalize else recon


def convolve(
    image: np.ndarray, kernel: np.ndarray, *, mode: str = "same"
) -> np.ndarray:
    """Convolve an image with a (PSF) kernel via an FFT.

    A thin wrapper over :func:`scipy.signal.fftconvolve`. With a unit-sum kernel
    (the :func:`render_core` / :func:`render_effective` default) the flux of each
    source is preserved, except near the image boundary where ``mode="same"``
    truncates the part of the kernel that falls off the frame.

    Parameters
    ----------
    image : numpy.ndarray
        The 2-D image to convolve.
    kernel : numpy.ndarray
        The 2-D convolution kernel (e.g. a rendered PSF or a matching kernel).
    mode : str, default "same"
        Forwarded to :func:`scipy.signal.fftconvolve`.

    Returns
    -------
    numpy.ndarray
        The convolved image.
    """
    from scipy.signal import fftconvolve

    return fftconvolve(
        np.asarray(image, dtype=float), np.asarray(kernel, dtype=float), mode=mode
    )
