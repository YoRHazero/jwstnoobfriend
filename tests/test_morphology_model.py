"""End-to-end tests for the morphology model (build, preview, sample, recover).

Synthetic data is generated with the same render primitives the model uses, so a
short NUTS run should recover the injected flux and size. No real FITS or STPSF.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology import NoobImage, Param, Sersic
from noobfriend.inference.morphology._geometry import build_geometry
from noobfriend.inference.morphology._render import convolve_and_bin, render_sersic

CENTER = (150.0, 2.0)
TRUTH = {"r_eff": 0.30, "n": 1.0, "q": 0.8, "theta": 20.0}
FLUX = {"F150W": 100.0, "F444W": 200.0}
OVERSAMPLE = 3


def _psf(size: int = 15, sigma: float = 1.8) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * sigma**2))
    return g / g.sum()


def _synth(seed: int = 0) -> NoobImage:
    shape = (31, 31)
    wcs = tangent_plane_wcs(*CENTER, 3.1, shape)  # 0.1"/pix
    geom = build_geometry(wcs, CENTER, shape, OVERSAMPLE)
    e, n = jnp.asarray(geom.e_grid), jnp.asarray(geom.n_grid)
    psf = _psf()
    rng = np.random.default_rng(seed)

    bands = {}
    for band, flux in FLUX.items():
        over = render_sersic(
            e,
            n,
            geom.subpixel_area,
            x0=0.0,
            y0=0.0,
            r_eff=TRUTH["r_eff"],
            n=TRUTH["n"],
            q=TRUTH["q"],
            theta=TRUTH["theta"],
            flux=flux,
        )
        clean = np.asarray(convolve_and_bin(over, jnp.asarray(psf), OVERSAMPLE))
        sigma = 0.02 * clean.max()
        data = clean + rng.normal(0.0, sigma, clean.shape)
        bands[band] = {
            "data": data,
            "error": np.full(shape, sigma),
            "wcs": wcs,
            "psf": psf,
        }
    return NoobImage.from_bands(bands, z=6.1, center=CENTER)


class TestBuild:
    """Resolution of priors and the preview render (no sampling)."""

    def test_priors_and_free_count(self) -> None:
        model = _synth().model([Sersic("gal")], oversample=OVERSAMPLE)
        # 6 geometry axes + one free flux per band (2) = 8 free parameters.
        assert model.n_free == 8
        assert "gal_n" in model.priors and "gal_flux_F150W" in model.priors

    def test_preview_runs(self) -> None:
        model = _synth().model([Sersic("gal")], oversample=OVERSAMPLE)
        prev = model.preview()
        assert prev.total["F150W"].shape == (31, 31)
        assert "gal" in prev.components["F150W"]

    def test_preview_includes_supplied_background(self) -> None:
        model = _synth().model([Sersic("gal")], oversample=OVERSAMPLE)
        base = model.preview()
        with_bg = model.preview({"bg_F150W": 3.0})
        assert np.allclose(with_bg.total["F150W"], base.total["F150W"] + 3.0)
        assert np.allclose(with_bg.residual["F150W"], base.residual["F150W"] - 3.0)
        assert np.allclose(with_bg.total["F444W"], base.total["F444W"])

    def test_fully_masked_band_raises(self) -> None:
        img = _synth().select(["F150W"])
        mask = np.ones(img.bands["F150W"].shape, dtype=bool)
        with pytest.raises(ValueError, match="no usable pixels"):
            img.with_masks({"F150W": mask}).model([Sersic("gal")])

    def test_duplicate_component_name_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate component"):
            _synth().model([Sersic("gal"), Sersic("gal")])

    def test_expression_param_without_prior_raises(self) -> None:
        a = Param("a")  # no prior
        with pytest.raises(ValueError, match="explicit prior"):
            _synth().model([Sersic("gal", r_eff=a + 0.1)])


class TestSample:
    """A short NUTS run recovers the injected source."""

    def test_recovers_flux_and_size(self) -> None:
        model = _synth().model(
            [Sersic("gal")], oversample=OVERSAMPLE, nu=None, background=False
        )
        result = model.sample(draws=150, tune=200, chains=1, seed=1)

        diag = result.diagnostics()
        assert set(diag) == {"max_r_hat", "min_ess_bulk", "divergences"}

        f150 = float(np.mean(result.flux("gal", "F150W")))
        assert f150 == pytest.approx(FLUX["F150W"], rel=0.2)
        r_eff = float(np.mean(result.r_eff_kpc("gal")))
        # r_eff 0.30" at z=6.1 -> ~1.7 kpc; loose tolerance for a short run.
        assert 0.8 < r_eff < 3.0
