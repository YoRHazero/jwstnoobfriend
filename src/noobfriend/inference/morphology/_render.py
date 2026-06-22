"""Differentiable (JAX) rendering of light profiles into a native band image.

The primitives here take the static per-band geometry (the oversampled arcsec
grids and the pixel index grids from :mod:`_geometry`) plus scalar parameters, and
return an oversampled image whose sum is the component's total flux. A scene is
rendered by summing components on the oversampled grid, convolving once with the
oversampled PSF, and binning down to the native grid -- all flux-conserving and
differentiable, so numpyro's NUTS can take gradients through it.

The Sersic normalisation is analytic (so the rendered total equals ``flux`` up to
the grid truncation), and ``b_n`` uses the Ciotti & Bertin expansion -- smooth in
``n``, unlike ``gammaincinv``. This module is internal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_LN_2PI: float = float(jnp.log(2.0 * jnp.pi))


def sersic_b(n: jax.Array | float) -> jax.Array:
    """Sersic ``b_n`` via the Ciotti & Bertin (1999) asymptotic expansion.

    Smooth and differentiable in ``n`` (accurate for ``n >= 0.36``), so it can sit
    inside a gradient-based sampler where ``gammaincinv`` cannot.
    """
    inv = 1.0 / n
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / 405.0 * inv
        + 46.0 / 25515.0 * inv**2
        + 131.0 / 1148175.0 * inv**3
        - 2194697.0 / 30690717750.0 * inv**4
    )


def render_sersic(
    e_grid: jax.Array,
    n_grid: jax.Array,
    subpixel_area: float,
    *,
    x0: jax.Array | float,
    y0: jax.Array | float,
    r_eff: jax.Array | float,
    n: jax.Array | float,
    q: jax.Array | float,
    theta: jax.Array | float,
    flux: jax.Array | float,
) -> jax.Array:
    """Render a Sersic profile onto the oversampled grid (sum == ``flux``).

    Parameters
    ----------
    e_grid, n_grid : jax.Array
        Oversampled arcsec offsets east/north of the target centre.
    subpixel_area : float
        Subpixel solid angle in arcsec^2.
    x0, y0 : float
        Centre offset (arcsec east/north).
    r_eff : float
        Semi-major-axis half-light radius (arcsec).
    n : float
        Sersic index.
    q : float
        Axis ratio (minor/major).
    theta : float
        Position angle (degrees, east of north).
    flux : float
        Total flux.

    Returns
    -------
    jax.Array
        Oversampled image of counts per subpixel.
    """
    th = jnp.deg2rad(theta)
    s, c = jnp.sin(th), jnp.cos(th)
    de = e_grid - x0
    dn = n_grid - y0
    u = de * s + dn * c  # along the major axis
    v = de * c - dn * s  # along the minor axis
    r = jnp.sqrt(u * u + (v / q) ** 2)

    b = sersic_b(n)
    ln_ie = jnp.log(flux) - (
        jnp.log(q)
        + _LN_2PI
        + 2.0 * jnp.log(r_eff)
        + jnp.log(n)
        + b
        - 2.0 * n * jnp.log(b)
        + jax.scipy.special.gammaln(2.0 * n)
    )
    sb = jnp.exp(ln_ie - b * ((r / r_eff) ** (1.0 / n) - 1.0))
    return sb * subpixel_area


def render_point(
    col_grid: jax.Array,
    row_grid: jax.Array,
    affine: jax.Array,
    pix_center: tuple[float, float],
    oversample: int,
    *,
    x0: jax.Array | float,
    y0: jax.Array | float,
    flux: jax.Array | float,
) -> jax.Array:
    """Deposit a point source onto the oversampled grid (sum == ``flux``).

    Uses a separable triangular (bilinear) kernel of one-subpixel width centred on
    the source, so the deposit is a partition of unity (flux-conserving) and smooth
    in ``(x0, y0)``.

    Parameters
    ----------
    col_grid, row_grid : jax.Array
        Oversampled column / row index grids (same shape as the image).
    affine : jax.Array
        The ``2x2`` arcsec-east/north -> native-pixel matrix.
    pix_center : tuple of float
        Target centre in native pixel coordinates.
    oversample : int
        Oversampling factor.
    x0, y0 : float
        Centre offset (arcsec east/north).
    flux : float
        Total flux.

    Returns
    -------
    jax.Array
        Oversampled image.
    """
    xc, yc = pix_center
    dx = affine[0, 0] * x0 + affine[0, 1] * y0
    dy = affine[1, 0] * x0 + affine[1, 1] * y0
    col_c = (xc + dx + 0.5) * oversample - 0.5
    row_c = (yc + dy + 0.5) * oversample - 0.5
    w_col = jnp.clip(1.0 - jnp.abs(col_grid - col_c), 0.0, 1.0)
    w_row = jnp.clip(1.0 - jnp.abs(row_grid - row_c), 0.0, 1.0)
    return flux * w_row * w_col


def convolve_and_bin(
    oversampled: jax.Array, psf: jax.Array, oversample: int
) -> jax.Array:
    """Convolve an oversampled image with the oversampled PSF and bin to native.

    Parameters
    ----------
    oversampled : jax.Array
        Oversampled scene (counts per subpixel).
    psf : jax.Array
        Oversampled PSF kernel (should sum to 1; normalise before calling).
    oversample : int
        Oversampling factor; both image dimensions must be divisible by it.

    Returns
    -------
    jax.Array
        Native-resolution image.
    """
    conv = jax.scipy.signal.fftconvolve(oversampled, psf, mode="same")
    rows, cols = conv.shape
    return conv.reshape(
        rows // oversample, oversample, cols // oversample, oversample
    ).sum(axis=(1, 3))
