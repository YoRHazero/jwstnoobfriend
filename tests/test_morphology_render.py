"""Tests for the JAX render primitives (flux conservation, b_n, convolution)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology._geometry import build_geometry
from noobfriend.inference.morphology._render import (
    convolve_and_bin,
    render_point,
    render_sersic,
    sersic_b,
)

CENTER = (150.0, 2.0)


def _fine_geom(npix: int = 61, oversample: int = 4):
    size = npix * 0.1  # 0.1"/pix
    wcs = tangent_plane_wcs(*CENTER, size, (npix, npix))
    return build_geometry(wcs, CENTER, (npix, npix), oversample)


def _index_grids(geom):
    o = geom.oversample
    rows, cols = geom.native_shape
    col = jnp.arange(cols * o, dtype=float)
    row = jnp.arange(rows * o, dtype=float)
    cg, rg = jnp.meshgrid(col, row)
    return cg, rg


class TestSersicB:
    """The Ciotti & Bertin b_n expansion."""

    def test_known_values(self) -> None:
        assert float(sersic_b(1.0)) == pytest.approx(1.6783, abs=1e-3)
        assert float(sersic_b(4.0)) == pytest.approx(7.6692, abs=1e-3)


class TestRenderSersic:
    """Analytic flux normalisation of the Sersic render."""

    def test_exponential_conserves_flux(self) -> None:
        geom = _fine_geom()
        e, n = jnp.asarray(geom.e_grid), jnp.asarray(geom.n_grid)
        img = render_sersic(
            e,
            n,
            geom.subpixel_area,
            x0=0.0,
            y0=0.0,
            r_eff=0.2,
            n=1.0,
            q=1.0,
            theta=0.0,
            flux=10.0,
        )
        assert float(jnp.sum(img)) == pytest.approx(10.0, rel=0.02)

    def test_elliptical_conserves_flux(self) -> None:
        geom = _fine_geom()
        e, n = jnp.asarray(geom.e_grid), jnp.asarray(geom.n_grid)
        img = render_sersic(
            e,
            n,
            geom.subpixel_area,
            x0=0.0,
            y0=0.0,
            r_eff=0.2,
            n=1.0,
            q=0.5,
            theta=30.0,
            flux=7.0,
        )
        assert float(jnp.sum(img)) == pytest.approx(7.0, rel=0.02)

    def test_high_index_is_in_the_ballpark(self) -> None:
        geom = _fine_geom(npix=81)
        e, n = jnp.asarray(geom.e_grid), jnp.asarray(geom.n_grid)
        img = render_sersic(
            e,
            n,
            geom.subpixel_area,
            x0=0.0,
            y0=0.0,
            r_eff=0.2,
            n=4.0,
            q=1.0,
            theta=0.0,
            flux=10.0,
        )
        # de Vaucouleurs wings + central cusp under-sampling: looser tolerance.
        assert float(jnp.sum(img)) == pytest.approx(10.0, rel=0.1)


class TestRenderPoint:
    """The triangular point-source deposit."""

    def test_conserves_flux(self) -> None:
        geom = _fine_geom(npix=21, oversample=3)
        cg, rg = _index_grids(geom)
        img = render_point(
            cg,
            rg,
            jnp.asarray(geom.affine),
            geom.pix_center,
            geom.oversample,
            x0=0.1,
            y0=-0.05,
            flux=5.0,
        )
        assert float(jnp.sum(img)) == pytest.approx(5.0, rel=1e-5)


class TestConvolveAndBin:
    """FFT convolution and the flux-conserving bin-down."""

    def test_delta_psf_is_block_sum(self) -> None:
        over = jnp.asarray(np.arange(36, dtype=float).reshape(6, 6))
        psf = jnp.zeros((7, 7)).at[3, 3].set(1.0)
        out = convolve_and_bin(over, psf, 2)
        expected = np.arange(36).reshape(3, 2, 3, 2).sum(axis=(1, 3))
        assert out.shape == (3, 3)
        assert np.allclose(np.asarray(out), expected)

    def test_compact_source_conserves_flux(self) -> None:
        # A centred blob with empty borders: 'same' convolution loses no flux
        # (energy only leaves the array at the edges, which are ~0 here).
        yy, xx = np.indices((18, 18))
        over = jnp.asarray(
            10.0 * np.exp(-((xx - 8.5) ** 2 + (yy - 8.5) ** 2) / (2 * 1.5**2))
        )
        g = np.exp(
            -((np.arange(5) - 2) ** 2)[:, None] - ((np.arange(5) - 2) ** 2)[None]
        )
        psf = jnp.asarray(g / g.sum())
        out = convolve_and_bin(over, psf, 3)
        assert float(jnp.sum(out)) == pytest.approx(float(jnp.sum(over)), rel=1e-3)
