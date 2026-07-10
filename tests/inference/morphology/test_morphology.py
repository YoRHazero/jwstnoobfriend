"""Tests for the collapsed-sampler morphology decomposition module.

All tests use synthetic arrays and Gaussian fake PSFs; nothing here touches
``example_data`` or real observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.morphology import (
    AmplitudePriors,
    CenterPrior,
    KernelPSF,
    ModelSpec,
    MultistartConfig,
    NoobImage,
    NoobImageSet,
    NutsConfig,
    Priors,
    SamplerConfig,
    fit_model,
    recovery,
    simulate,
)

jax = pytest.importorskip("jax")


def _gaussian_kernel(size: int = 15, sigma: float = 1.3) -> np.ndarray:
    half = (size - 1) / 2.0
    yy, xx = np.indices((size, size)) - half
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _image(name: str = "b1", n: int = 31, pixel_scale: float = 0.05) -> NoobImage:
    return NoobImage(
        name=name,
        data=np.zeros((n, n)),
        error=np.ones((n, n)),
        psf=KernelPSF(_gaussian_kernel()),
        pixel_scale=pixel_scale,
        wavelength_um=2.0,
    )


class TestModelSpec:
    """Parsing and structure of the model family."""

    @pytest.mark.parametrize(
        ("name", "n_points", "has_host"),
        [
            ("point", 1, False),
            ("host", 0, True),
            ("point+host", 1, True),
            ("2point", 2, False),
            ("2point+host", 2, True),
        ],
    )
    def test_parse_round_trips(self, name, n_points, has_host):
        spec = ModelSpec.parse(name)
        assert spec == ModelSpec(n_points=n_points, has_host=has_host)
        assert spec.name == name

    def test_parse_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="unknown model"):
            ModelSpec.parse("sersic")

    def test_needs_at_least_one_component(self):
        with pytest.raises(ValueError):
            ModelSpec(n_points=0, has_host=False)

    def test_amplitude_count_includes_background(self):
        assert ModelSpec.parse("2point+host").n_amplitudes == 4
        assert ModelSpec.parse("point").component_names == ("point0",)


class TestPriors:
    """Data-driven prior defaults."""

    def test_from_images_covers_all_bands(self):
        images = NoobImageSet([_image("a"), _image("b")])
        priors = Priors.from_images(images)
        for band in ("a", "b"):
            flux, background = priors.amplitudes.for_band(band)
            assert flux > 0 and background > 0

    def test_center_prior_rejects_bad_scale(self):
        with pytest.raises(ValueError):
            CenterPrior(scale=0.0)

    def test_override_replaces_block(self):
        images = NoobImageSet([_image()])
        amplitudes = AmplitudePriors(
            flux_scale={"b1": 7.0}, background_scale={"b1": 1.0}
        )
        priors = Priors.from_images(images, amplitudes=amplitudes)
        assert priors.amplitudes.for_band("b1") == (7.0, 1.0)


class TestTransform:
    """Latent-to-physical maps: dimensions, containment, identifiability."""

    def _transform(self, name: str):
        from noobfriend.inference.morphology._transforms import build_transform

        return build_transform(
            ModelSpec.parse(name), Priors.from_images(NoobImageSet([_image()]))
        )

    @pytest.mark.parametrize(
        ("name", "dim"),
        [
            ("point", 2),
            ("2point", 4),
            ("host", 6),
            ("point+host", 8),
            ("2point+host", 10),
        ],
    )
    def test_latent_dimension(self, name, dim):
        assert self._transform(name).dim == dim

    def test_containment_holds_for_extreme_latents(self):
        # The primary point source must lie inside the one-r_eff host
        # ellipse for any latent value, including far tails.
        transform = self._transform("point+host")
        rng = np.random.default_rng(0)
        for _ in range(200):
            z = 10.0 * rng.standard_normal(transform.dim)
            phys = transform.physical(z, np)
            assert phys["host_offset_rho"] < 1.0
            assert 0.0 < phys["host_q"] <= 1.0

    def test_round_host_is_regular(self):
        # q -> 1 corresponds to the origin of the ellipticity disk, a
        # regular point: zero latents give exactly q == 1 without blowup.
        transform = self._transform("point+host")
        phys = transform.physical(np.zeros(transform.dim), np)
        assert phys["host_q"] == pytest.approx(1.0)
        assert np.isfinite(phys["host_theta_deg"])

    def test_host_only_has_free_center(self):
        transform = self._transform("host")
        phys = transform.physical(np.zeros(transform.dim), np)
        assert "host_x" in phys and "host_offset_rho" not in phys


class TestCollapse:
    """Marginal likelihood against brute-force dense Gaussian."""

    def test_marginal_matches_dense_gaussian(self):
        from scipy.stats import multivariate_normal

        from noobfriend.inference.morphology._collapse import marginal_loglik
        import jax.numpy as jnp

        rng = np.random.default_rng(1)
        n, k = 40, 3
        sigma = 0.5 + rng.random(n)
        x = rng.standard_normal((k, n))
        v = np.array([2.0, 5.0, 1.0])
        y = rng.standard_normal(n) * sigma

        x_w = x / sigma[None, :]
        y_w = y / sigma
        const = -0.5 * (n * np.log(2 * np.pi) + 2 * np.sum(np.log(sigma)))
        got = float(
            marginal_loglik(jnp.asarray(y_w), jnp.asarray(x_w), jnp.asarray(v), const)
        )
        cov = np.diag(sigma**2) + (x.T * v**2) @ x
        expected = multivariate_normal(mean=np.zeros(n), cov=cov).logpdf(y)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_conditional_matches_normal_equations(self):
        from noobfriend.inference.morphology._collapse import conditional_amplitudes
        import jax.numpy as jnp

        rng = np.random.default_rng(2)
        n, k = 30, 2
        x_w = rng.standard_normal((k, n))
        y_w = rng.standard_normal(n)
        v = np.array([3.0, 0.5])
        mean, factor = conditional_amplitudes(
            jnp.asarray(y_w), jnp.asarray(x_w), jnp.asarray(v)
        )
        precision = np.diag(1 / v**2) + x_w @ x_w.T
        cov = np.linalg.inv(precision)
        np.testing.assert_allclose(np.asarray(mean), cov @ (x_w @ y_w), rtol=1e-9)
        np.testing.assert_allclose(
            np.asarray(factor @ factor.T), cov, rtol=1e-9, atol=1e-12
        )


class TestRender:
    """Unit-amplitude component rendering."""

    def _static(self, n: int = 31):
        from noobfriend.inference.morphology._render import prepare_static

        return prepare_static(_image(n=n), oversample=4)

    def test_point_basis_is_unit_flux(self):
        from noobfriend.inference.morphology._render import source_images
        import jax.numpy as jnp

        static = self._static()
        phys = {"point0_x": jnp.asarray(0.013), "point0_y": jnp.asarray(-0.02)}
        image = np.asarray(source_images(static, ModelSpec.parse("point"), phys))[0]
        # ePSF lattice sums depend weakly on sub-pixel phase
        assert image.sum() == pytest.approx(1.0, abs=1e-3)

    def test_sersic_analytic_normalization(self):
        # A compact profile well inside a large stamp integrates to ~1
        # under the infinite-plane gamma-function normalization.
        from noobfriend.inference.morphology._render import source_images
        import jax.numpy as jnp

        static = self._static(n=101)
        phys = {
            "point0_x": jnp.asarray(0.0),
            "point0_y": jnp.asarray(0.0),
            "host_r_eff": jnp.asarray(0.12),
            "host_n": jnp.asarray(1.5),
            "host_q": jnp.asarray(0.7),
            "host_theta_deg": jnp.asarray(30.0),
            "host_x": jnp.asarray(0.0),
            "host_y": jnp.asarray(0.0),
        }
        images = np.asarray(source_images(static, ModelSpec.parse("point+host"), phys))
        assert images[1].sum() == pytest.approx(1.0, abs=0.02)

    def test_rejects_bad_oversample(self):
        from noobfriend.inference.morphology._render import prepare_static

        with pytest.raises(ValueError):
            prepare_static(_image(), oversample=0)

    def test_point_at_integer_offset_matches_decimated_kernel(self):
        # Rendering a point exactly one native pixel off centre must
        # reproduce the native-decimated ePSF shifted by one pixel: the
        # bicubic gather interpolates exactly at grid nodes.
        from noobfriend.inference.morphology._render import source_images
        import jax.numpy as jnp

        static = self._static()
        scale = static.pixel_scale
        centered = np.asarray(
            source_images(
                static,
                ModelSpec.parse("point"),
                {"point0_x": jnp.asarray(0.0), "point0_y": jnp.asarray(0.0)},
            )
        )[0]
        shifted = np.asarray(
            source_images(
                static,
                ModelSpec.parse("point"),
                {"point0_x": jnp.asarray(scale), "point0_y": jnp.asarray(0.0)},
            )
        )[0]
        np.testing.assert_allclose(shifted[:, 1:], centered[:, :-1], atol=1e-12)

    def test_oversampled_provider_round_trips(self):
        # A provider holding a 2x-oversampled stamp must decimate back to
        # the native kernel it was built from (up to normalization).
        from scipy.ndimage import zoom

        native = _gaussian_kernel(size=15, sigma=2.0)
        fine = zoom(native, 2.0, order=3)[:-1, :-1]  # 29x29, odd
        fine = fine / fine.sum()
        image = NoobImage(
            name="b1",
            data=np.zeros((31, 31)),
            error=np.ones((31, 31)),
            psf=KernelPSF(fine, oversample=2),
            pixel_scale=0.05,
        )
        back = image.psf.kernel_for(image, oversample=1)
        assert back.shape[0] % 2 == 1
        assert back.sum() == pytest.approx(1.0)
        peak = np.unravel_index(back.argmax(), back.shape)
        assert peak == ((back.shape[0] - 1) // 2, (back.shape[1] - 1) // 2)


class TestSimulateAndRecovery:
    """Forward simulation and truth comparison plumbing."""

    def test_simulate_is_deterministic_per_seed(self):
        images = NoobImageSet([_image()])
        kwargs = dict(
            params={"point0_x": 0.0, "point0_y": 0.0},
            fluxes={"point0": {"b1": 100.0}},
        )
        one = simulate(images, "point", seed=3, **kwargs)
        two = simulate(images, "point", seed=3, **kwargs)
        np.testing.assert_array_equal(one["b1"].data, two["b1"].data)

    def test_noise_free_simulation_matches_flux(self):
        images = NoobImageSet([_image()])
        sim = simulate(
            images,
            "point",
            params={"point0_x": 0.0, "point0_y": 0.0},
            fluxes={"point0": {"b1": 100.0}},
            noise=False,
        )
        assert sim["b1"].data.sum() == pytest.approx(100.0, rel=1e-3)


class TestStacking:
    """Per-chain stacking weights and resampling."""

    def test_bad_chain_gets_zero_weight(self):
        from noobfriend.inference.morphology._diagnostics import stacking_weights

        rng = np.random.default_rng(0)
        good = rng.normal(-1.0, 0.05, size=(1, 80, 40))
        bad = rng.normal(-3.0, 0.05, size=(1, 80, 40))
        weights = stacking_weights(np.concatenate([good, bad]))
        assert weights is not None
        assert weights[0] > 0.99
        assert weights.sum() == pytest.approx(1.0)

    def test_equivalent_chains_get_uniform_weights(self):
        from noobfriend.inference.morphology._diagnostics import stacking_weights

        rng = np.random.default_rng(1)
        loglik = rng.normal(-1.0, 0.05, size=(4, 80, 40))
        weights = stacking_weights(loglik)
        assert weights is not None
        np.testing.assert_allclose(weights, 0.25, atol=0.15)

    def _dummy_fit(self, weights):
        from noobfriend.inference.morphology.results import (
            ModelFit,
            SamplerHealth,
        )

        chains, draws = 2, 30
        params = {"point0_x": np.stack([np.zeros(draws), np.ones(draws)])}
        fluxes = {"point0": {"b1": np.stack([np.zeros(draws), np.ones(draws)])}}
        return ModelFit(
            spec=ModelSpec.parse("point"),
            params=params,
            fluxes=fluxes,
            z=np.zeros((chains, draws, 2)),
            health=SamplerHealth(1.5, 10, 10, 0, ("flagged",)),
            loo=None,
            rendered={},
            map_modes=(),
            timings={},
            images=NoobImageSet([_image()]),
            chain_weights=weights,
        )

    def test_stacked_resamples_by_weight(self):
        fit = self._dummy_fit(np.array([1.0, 0.0]))
        stacked = fit.stacked(seed=0)
        assert stacked.params["point0_x"].shape == (1, 60)
        assert stacked.params["point0_x"].max() == 0.0  # chain 1 excluded
        assert stacked.flux("point0", "b1").max() == 0.0

    def test_stacked_requires_weights(self):
        fit = self._dummy_fit(None)
        with pytest.raises(ValueError, match="stacking weights"):
            fit.stacked()


class TestEndToEnd:
    """Small full-pipeline regression: point-only fit on injected data."""

    def test_point_fit_recovers_truth(self):
        images = NoobImageSet([_image(n=25)])
        truth = {"point0_x": 0.02, "point0_y": -0.01}
        sim = simulate(
            images, "point", params=truth, fluxes={"point0": {"b1": 2000.0}}, seed=5
        )
        fit = fit_model(
            sim,
            "point",
            sampler=SamplerConfig(
                nuts=NutsConfig(chains=2, warmup=150, draws=150),
                multistart=MultistartConfig(n_starts=32, n_refine=2),
                compute_loo=False,
            ),
            seed=0,
        )
        report = recovery(fit, {**truth, "point0_flux_b1": 2000.0})
        assert fit.health.divergences == 0
        assert report.max_abs_z < 4.0
        assert set(fit.fluxes) == {"point0", "background"}
        assert fit.rendered["b1"]["total"].shape == (25, 25)
