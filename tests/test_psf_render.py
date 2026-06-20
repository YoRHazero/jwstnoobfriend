"""Tests for rendering, convolving, and persisting built PSFs.

Synthetic frames + a linear fake WCS only -- no FITS / example_data. The core and
hybrid PSFs are built from synthetic stars (as in test_psf.py's TestBuild), then
rendered to native grids, convolved, and round-tripped through FITS.
"""

import astropy.units as u
import matplotlib
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from noobase.image import convolve_psf

matplotlib.use("Agg")

from noobfriend.extraction.psf import (  # noqa: E402
    CorePsf,
    EffectivePsf,
    SourceExtractor,
    render_core,
)


class _LinearWCS:
    """A minimal APE-14 WCS: pixel (x, y) -> sky via a linear scale."""

    def __init__(self, ra0: float = 53.0, dec0: float = -27.0, scale: float = 1e-4):
        self.ra0, self.dec0, self.scale = ra0, dec0, scale

    def pixel_to_world(self, x: np.ndarray, y: np.ndarray) -> SkyCoord:
        return SkyCoord(
            (self.ra0 + self.scale * np.asarray(x)) * u.deg,
            (self.dec0 + self.scale * np.asarray(y)) * u.deg,
        )


def _frame(positions: list[tuple], *, seed: int = 0) -> np.ndarray:
    """Synthetic 220x220 frame with Gaussian point sources at ``(y, x)``."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, size=(220, 220))
    yy, xx = np.mgrid[0:220, 0:220]
    for y, x in positions:
        img += 300.0 * np.exp(-0.5 * (((yy - y) / 2.0) ** 2 + ((xx - x) / 2.0) ** 2))
    return img


def _filled(cutout_size: int) -> SourceExtractor:
    """Return a selected extractor over four bright stars in phase-shifted frames."""
    wcs = _LinearWCS()
    ext = SourceExtractor(fwhm=4.0, cutout_size=cutout_size, nsigma=5.0)
    positions = [(70, 70), (70, 150), (150, 70), (150, 150)]
    for k in range(6):
        oy, ox = (0.37 * k) % 1 - 0.5, (0.61 * k) % 1 - 0.5
        frame = _frame([(y + oy, x + ox) for y, x in positions], seed=k)
        ext.add_from_frame(
            frame, np.ones_like(frame), wcs=wcs, filter="F210M", detector="nrca1"
        )
    return ext.select(filter="F210M", max_ellipticity=0.3)


@pytest.fixture(scope="module")
def core_psf() -> CorePsf:
    return _filled(29).build_psf_core(oversample=3, core_size=21)


@pytest.fixture(scope="module")
def effective_psf() -> EffectivePsf:
    sub = _filled(55)
    core = sub.build_psf_core(oversample=3, core_size=21)
    return sub.build_psf_wings(core, wing_size=51)


class TestRenderCore:
    """Decimating the oversampled core onto a native grid."""

    def test_default_size_and_unit_sum(self, core_psf: CorePsf) -> None:
        img = core_psf.render()  # native core size = core_size
        assert img.shape == (21, 21)
        assert img.sum() == pytest.approx(1.0)
        assert np.isfinite(img).all()

    def test_custom_size_pads_centered(self, core_psf: CorePsf) -> None:
        img = core_psf.render(size=31)
        assert img.shape == (31, 31)
        assert np.argmax(img) == 31 * 15 + 15  # peak at the centre pixel

    def test_unnormalized_is_central_phase_decimation(self, core_psf: CorePsf) -> None:
        o = core_psf.oversample
        raw = render_core(core_psf.core, o, normalize=False)
        expected = core_psf.core[o // 2 :: o, o // 2 :: o]
        assert np.allclose(
            raw, expected
        )  # raw render is the decimated core, no scaling
        assert render_core(core_psf.core, o).sum() == pytest.approx(1.0)  # normalized


class TestRenderEffective:
    """The hybrid sampling contract on a native grid."""

    def test_default_size_and_unit_sum(self, effective_psf: EffectivePsf) -> None:
        img = effective_psf.render()  # default size = wing_size
        assert img.shape == (51, 51)
        assert img.sum() == pytest.approx(1.0)
        assert np.isfinite(img).all()

    def test_unnormalized_is_ee_gauged(self, effective_psf: EffectivePsf) -> None:
        # noobase normalises encircled energy within ee_aperture_radius to ~1
        raw = effective_psf.render(normalize=False)
        c = (raw.shape[0] - 1) / 2.0
        yy, xx = np.mgrid[0 : raw.shape[0], 0 : raw.shape[1]]
        within = np.hypot(yy - c, xx - c) <= effective_psf.ee_aperture_radius
        assert raw[within].sum() == pytest.approx(1.0, abs=1e-3)

    def test_far_region_equals_wing(self, effective_psf: EffectivePsf) -> None:
        # beyond the feather, f_core = 0 so the render is exactly the wing plane
        raw = effective_psf.render(normalize=False)
        c = (raw.shape[0] - 1) / 2.0
        yy, xx = np.mgrid[0 : raw.shape[0], 0 : raw.shape[1]]
        far = np.hypot(yy - c, xx - c) > effective_psf.match_radius + 4
        assert np.allclose(raw[far], np.asarray(effective_psf.wing)[far])

    def test_size_crops(self, effective_psf: EffectivePsf) -> None:
        img = effective_psf.render(size=31)
        assert img.shape == (31, 31)
        assert img.sum() == pytest.approx(1.0)


class TestConvolveWithNoobase:
    """A rendered kernel pairs with noobase's flip-correct, NaN-aware convolution."""

    def test_delta_returns_kernel(self, core_psf: CorePsf) -> None:
        kernel = core_psf.render()
        image = np.zeros((41, 41))
        image[20, 20] = 1.0
        out = convolve_psf(image, kernel)
        # a delta convolved with the kernel reproduces the kernel at the centre
        assert out.shape == (41, 41)
        assert out[10:31, 10:31] == pytest.approx(kernel, abs=1e-6)

    def test_preserves_interior_flux(self, effective_psf: EffectivePsf) -> None:
        # a point source with margin >= kernel half-width keeps all its flux inside
        image = np.zeros((81, 81))
        image[40, 40] = 5.0
        out = convolve_psf(image, effective_psf.render())
        assert out.sum() == pytest.approx(5.0, rel=1e-3)  # noobase renorm: ~5e-5


class TestCorePsfIO:
    """CorePsf save / load round-trip."""

    def test_round_trip(self, core_psf: CorePsf, tmp_path) -> None:
        core_psf.save(tmp_path / "core.fits")
        loaded = CorePsf.load(tmp_path / "core.fits")
        assert np.allclose(loaded.core, core_psf.core)
        assert loaded.oversample == core_psf.oversample
        assert loaded.core_size == core_psf.core_size
        assert np.allclose(loaded.flux, core_psf.flux)
        assert loaded.converged == core_psf.converged
        assert loaded.n_stars == core_psf.n_stars
        assert loaded.n_sources == core_psf.n_sources
        assert np.array_equal(loaded.phase_grid, core_psf.phase_grid)

    def test_overwrite_required(self, core_psf: CorePsf, tmp_path) -> None:
        core_psf.save(tmp_path / "core.fits")
        with pytest.raises(FileExistsError):
            core_psf.save(tmp_path / "core.fits")
        core_psf.save(tmp_path / "core.fits", overwrite=True)


class TestEffectivePsfIO:
    """EffectivePsf save / load round-trip and faithful render."""

    def test_round_trip_fields(self, effective_psf: EffectivePsf, tmp_path) -> None:
        effective_psf.save(tmp_path / "psf.fits")
        loaded = EffectivePsf.load(tmp_path / "psf.fits")
        assert np.allclose(loaded.core_plane, effective_psf.core_plane)
        assert np.allclose(loaded.wing, effective_psf.wing)
        assert loaded.oversample == effective_psf.oversample
        assert loaded.match_radius == effective_psf.match_radius
        assert loaded.feather_width == effective_psf.feather_width
        assert loaded.ee_aperture_radius == effective_psf.ee_aperture_radius
        assert loaded.wing_size == effective_psf.wing_size
        assert loaded.n_wing_stars == effective_psf.n_wing_stars
        assert np.allclose(loaded.core.core, effective_psf.core.core)
        assert loaded.core.n_stars == effective_psf.core.n_stars

    def test_loaded_renders_identically(
        self, effective_psf: EffectivePsf, tmp_path
    ) -> None:
        effective_psf.save(tmp_path / "psf.fits")
        loaded = EffectivePsf.load(tmp_path / "psf.fits")
        assert np.allclose(loaded.render(), effective_psf.render())

    def test_wrong_loader_raises(self, core_psf: CorePsf, tmp_path) -> None:
        core_psf.save(tmp_path / "core.fits")
        with pytest.raises(ValueError, match="EffectivePsf"):
            EffectivePsf.load(tmp_path / "core.fits")


class TestPlot:
    """The .plot() diagnostic figure (image + radial profile)."""

    def test_core_plot_returns_figure(self, core_psf: CorePsf) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        fig = core_psf.plot()
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 2  # image panel + radial-profile panel (+ colorbar)
        plt.close(fig)

    def test_effective_plot_and_save(
        self, effective_psf: EffectivePsf, tmp_path
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        out = tmp_path / "psf.png"
        fig = effective_psf.plot(save=out)
        assert isinstance(fig, Figure)
        assert out.is_file() and out.stat().st_size > 0
        plt.close(fig)
