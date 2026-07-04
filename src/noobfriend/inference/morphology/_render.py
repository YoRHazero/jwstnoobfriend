"""JAX rendering of unit-amplitude component images.

This is the only implementation of the model physics. Each band gets a
:class:`BandStatic` bundle prepared once per fit: float64 arrays, oversampled
evaluation grids, and the padded PSF-kernel FFT (computed once and reused by
every likelihood evaluation).

Component images are rendered at *unit amplitude*: the Sersic basis uses
the analytic (infinite-plane) flux normalization via the gamma function,
so its amplitude is a total flux independent of stamp size or
discretization, and the point basis is the effective PSF (ePSF) evaluated
directly at every native pixel centre with the source's sub-pixel phase.

Point evaluation interpolates the *oversampled* ePSF grid with a Keys
bicubic (Catmull-Rom) stencil — the standard Anderson & King ePSF
evaluation. This is a single uniform code path: providers that only hold a
native-resolution kernel are internally spline-refined by ``kernel_for``,
which reproduces roughly interpolation-quality sub-pixel placement, while a
genuine oversampled ePSF eliminates the position-dependent broadening that
native-resolution interpolation suffers on undersampled PSFs (up to ~45%
FWHM at half-pixel phase for NIRCam SW at 0.05 arcsec/px; ~3% at 4x
oversampling). A Fourier phase shift was tried and rejected: band-limited
translation of an undersampled kernel produces catastrophic Gibbs ringing.
No FFT is involved in the point basis at all; the FFT convolution is only
used for the Sersic component (with the native-decimated kernel).

Importing this module enables JAX float64 globally: the log-posterior sums
tens of thousands of squared residuals, and in float32 that sum carries
absolute error comparable to the O(1) energy differences NUTS accepts or
rejects on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

import jax.numpy as jnp
from jax.scipy.special import gammaln

from noobfriend.inference.morphology import _x64  # noqa: F401

from noobfriend.inference.morphology.data import NoobImage
from noobfriend.inference.morphology.models import ModelSpec

__all__ = ["BandStatic", "prepare_static", "source_images"]


@dataclass(frozen=True)
class BandStatic:
    """Prepared per-band arrays shared by every likelihood evaluation.

    All floating arrays are float64 JAX arrays. ``invsig`` is ``1 / error``
    with masked-out pixels set to zero, so whitening by it both scales and
    masks in one multiply.
    """

    name: str
    shape: tuple[int, int]
    pixel_scale: float
    data: Any
    invsig: Any
    y_w: Any
    x: Any
    y: Any
    x_os: Any
    y_os: Any
    oversample: int
    kernel_fft: Any
    fft_shape: tuple[int, int]
    crop: tuple[int, int]
    epsf: Any
    psf_oversample: int
    epsf_center: tuple[float, float]
    mask_idx: Any
    n_masked: int
    loglik_const: float


def prepare_static(image: NoobImage, *, oversample: int = 4) -> BandStatic:
    """Prepare one image for repeated JAX model evaluation.

    Parameters
    ----------
    image : NoobImage
        The prepared single-band image.
    oversample : int
        Subpixel sampling factor per axis for Sersic evaluation.

    Returns
    -------
    BandStatic
        Static arrays including the cached padded PSF-kernel FFT.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    data = np.asarray(image.data, dtype=np.float64)
    error = np.asarray(image.error, dtype=np.float64)
    mask = np.asarray(image.mask, dtype=bool)
    height, width = data.shape

    invsig = np.zeros_like(data)
    invsig[mask] = 1.0 / error[mask]
    n_masked = int(mask.sum())
    loglik_const = -0.5 * (
        n_masked * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(error[mask]))
    )

    x, y = image.arcsec_grid()
    scale = float(image.pixel_scale)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    row_os = ((np.arange(height * oversample) + 0.5) / oversample - 0.5 - cy) * scale
    col_os = ((np.arange(width * oversample) + 0.5) / oversample - 0.5 - cx) * scale
    y_os, x_os = np.meshgrid(row_os, col_os, indexing="ij")

    fine = np.asarray(
        image.psf.kernel_for(image, oversample=oversample), dtype=np.float64
    )
    # ePSF-function normalization: samples on any native-stride lattice sum
    # to ~1, so a gather-rendered point source has unit total flux.
    fine = fine * oversample**2 / fine.sum()
    kernel = _decimate_center(fine, oversample)
    kh, kw = kernel.shape
    fft_shape = (next_fast_len(height + kh - 1), next_fast_len(width + kw - 1))
    kernel_fft = jnp.fft.rfft2(jnp.asarray(kernel), s=fft_shape)
    pad = 2
    fine_padded = np.pad(fine, pad)
    epsf_center = (
        (fine.shape[0] - 1) / 2.0 + pad,
        (fine.shape[1] - 1) / 2.0 + pad,
    )

    return BandStatic(
        name=image.name,
        shape=(height, width),
        pixel_scale=scale,
        data=jnp.asarray(data),
        invsig=jnp.asarray(invsig),
        y_w=jnp.asarray(data * invsig),
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        x_os=jnp.asarray(x_os),
        y_os=jnp.asarray(y_os),
        oversample=oversample,
        kernel_fft=kernel_fft,
        fft_shape=fft_shape,
        crop=((kh - 1) // 2, (kw - 1) // 2),
        epsf=jnp.asarray(fine_padded),
        psf_oversample=oversample,
        epsf_center=epsf_center,
        mask_idx=jnp.asarray(np.flatnonzero(mask.ravel())),
        n_masked=n_masked,
        loglik_const=float(loglik_const),
    )


def source_images(static: BandStatic, spec: ModelSpec, phys: dict[str, Any]) -> Any:
    """Render all PSF-convolved unit-amplitude source images for one band.

    Parameters
    ----------
    static : BandStatic
        Prepared band arrays.
    spec : ModelSpec
        The model being rendered.
    phys : dict
        Physical parameters from the transform layer (JAX scalars).

    Returns
    -------
    jax.Array
        Stack of shape ``(n_points + has_host, height, width)`` in component
        order ``point0, ..., host``; each image integrates to (approximately)
        unit total flux before stamp truncation.
    """
    layers = [
        _point_epsf(static, phys[f"point{k}_x"], phys[f"point{k}_y"])
        for k in range(spec.n_points)
    ]
    if spec.has_host:
        height, width = static.shape
        row0, col0 = static.crop
        profile_fft = jnp.fft.rfft2(_sersic_profile(static, phys), s=static.fft_shape)
        convolved = jnp.fft.irfft2(profile_fft * static.kernel_fft, s=static.fft_shape)
        layers.append(convolved[row0 : row0 + height, col0 : col0 + width])
    return jnp.stack(layers)


def _cubic_weights(frac: Any) -> tuple[Any, Any, Any, Any]:
    """Return Keys (Catmull-Rom, a=-1/2) weights for taps at -1, 0, 1, 2."""
    s = frac
    s2 = s * s
    s3 = s2 * s
    return (
        -0.5 * s + s2 - 0.5 * s3,
        1.0 - 2.5 * s2 + 1.5 * s3,
        0.5 * s + 2.0 * s2 - 1.5 * s3,
        -0.5 * s2 + 0.5 * s3,
    )


def _point_epsf(static: BandStatic, x0: Any, y0: Any) -> Any:
    """Evaluate the oversampled ePSF at every native pixel centre.

    The interpolant is C1 in the source position (Keys bicubic), so the
    likelihood stays smooth for NUTS; the stencil reads the zero-padded
    ePSF with clipped indices, so sources near the stamp edge simply lose
    the off-stamp part of their flux.
    """
    fine_scale = static.pixel_scale / static.psf_oversample
    rows, cols = static.epsf.shape
    u_col = (static.x - x0) / fine_scale + static.epsf_center[1]
    u_row = (static.y - y0) / fine_scale + static.epsf_center[0]
    base_col = jnp.floor(u_col).astype(jnp.int32)
    base_row = jnp.floor(u_row).astype(jnp.int32)
    wc = _cubic_weights(u_col - base_col)
    wr = _cubic_weights(u_row - base_row)
    flat = static.epsf.ravel()
    out = jnp.zeros(static.shape)
    for a in range(4):
        row_idx = base_row + (a - 1)
        row_ok = (row_idx >= 0) & (row_idx < rows)
        row_safe = jnp.clip(row_idx, 0, rows - 1)
        for b in range(4):
            col_idx = base_col + (b - 1)
            ok = row_ok & (col_idx >= 0) & (col_idx < cols)
            col_safe = jnp.clip(col_idx, 0, cols - 1)
            values = jnp.take(flat, row_safe * cols + col_safe)
            out = out + jnp.where(ok, wr[a] * wc[b] * values, 0.0)
    return out


def _decimate_center(fine: np.ndarray, oversample: int) -> np.ndarray:
    """Sample the fine ePSF on the native-stride lattice through its centre."""
    if oversample == 1:
        return fine / fine.sum()
    rows, cols = fine.shape
    center_r = (rows - 1) // 2
    center_c = (cols - 1) // 2
    r_idx = np.arange(center_r % oversample, rows, oversample)
    c_idx = np.arange(center_c % oversample, cols, oversample)
    native = fine[np.ix_(r_idx, c_idx)]
    if native.shape[0] % 2 == 0:
        native = (
            native[:-1] if r_idx[-1] - center_r > center_r - r_idx[0] else native[1:]
        )
    if native.shape[1] % 2 == 0:
        native = (
            native[:, :-1]
            if c_idx[-1] - center_c > center_c - c_idx[0]
            else native[:, 1:]
        )
    return native / native.sum()


def _sersic_profile(static: BandStatic, phys: dict[str, Any]) -> Any:
    """Render the analytically normalized unit-flux Sersic on the native grid.

    Surface brightness is evaluated on the oversampled grid and averaged into
    native pixels; the normalization is the exact infinite-plane integral
    ``2 pi n q r_eff^2 Gamma(2n) / b^(2n)``, so amplitude means total flux.
    """
    r_eff = phys["host_r_eff"]
    n = phys["host_n"]
    q = phys["host_q"]
    theta = jnp.deg2rad(phys["host_theta_deg"])
    dx = static.x_os - phys["host_x"]
    dy = static.y_os - phys["host_y"]
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    xp = dx * cos_t + dy * sin_t
    yp = -dx * sin_t + dy * cos_t
    # Soften the central cusp by a fraction of an oversampled pixel: for
    # n > 1 the surface-brightness gradient w.r.t. the centre diverges as a
    # grid node approaches r = 0, which injects gradient spikes into HMC.
    # The softening moves ~<1% of the flux for n <= 4 hosts and keeps the
    # likelihood smooth in the centre parameters.
    r_core = 0.3 * static.pixel_scale / static.oversample
    r_ell = jnp.sqrt(xp * xp + (yp / q) ** 2 + r_core * r_core)
    b = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
    log_norm = 2.0 * n * jnp.log(b) - gammaln(2.0 * n)
    norm = jnp.exp(log_norm) / (2.0 * jnp.pi * n * q * r_eff * r_eff)
    sb = norm * jnp.exp(-b * (r_ell / r_eff) ** (1.0 / n))
    height, width = static.shape
    os = static.oversample
    native = sb.reshape(height, os, width, os).mean(axis=(1, 3))
    return native * static.pixel_scale**2
