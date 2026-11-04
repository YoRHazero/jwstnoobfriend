"""Result features on a synthetic posterior: access, decompose, plots, R1, I/O."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("arviz")
pytest.importorskip("h5netcdf")

from noobfriend.inference.spectrum import (  # noqa: E402
    LogUniform,
    NoobContinuum,
    NoobLine,
    NoobSpectrum,
    Uniform,
)
from noobfriend.inference.spectrum.mcmc import NonNormalPriorWarning  # noqa: E402
from noobfriend.inference.spectrum.priors import JointMember  # noqa: E402

from .._helpers import HALPHA, Z, halpha_spectrum  # noqa: E402
from ._helpers import fake_result  # noqa: E402

LAMBDA_0 = HALPHA * (1 + Z)
TRUTH = {"Ha.fwhm": 300.0, "Ha.flux": 40.0, "Ha.delta_v": 0.0, "continuum.c0": 1.0}


def ha() -> NoobLine:
    return NoobLine("Ha", rest=HALPHA, unit="angstrom")


def single(continuum: NoobContinuum | None = None) -> object:
    workspace = halpha_spectrum(seed=7).prepare(
        [ha()], continuum=continuum or NoobContinuum.polynomial(1, lambda_0=LAMBDA_0)
    )
    return fake_result(workspace, TRUTH)


class TestAccess:
    """Values mirror the declarations (T11a.2)."""

    def test_line_and_continuum(self) -> None:
        result = single()
        assert result["Ha"].fwhm.median == pytest.approx(300.0, rel=0.02)
        assert result["Ha.flux"].name == "Ha.flux"
        assert result["continuum"].coefficient(0).median == pytest.approx(1.0, rel=0.05)
        with pytest.raises(AttributeError, match="not a free parameter"):
            _ = result["Ha"].flux_ratio_to_base

    def test_frames_and_kernels(self) -> None:
        line = ha().convolve("laplace", LogUniform(100.0, 3000.0), fraction=Uniform())
        spectrum = NoobSpectrum.join(
            {"a": halpha_spectrum(seed=1), "b": halpha_spectrum(seed=2)}
        )
        workspace = spectrum.prepare(
            [line], continuum=NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)
        )
        result = fake_result(workspace, TRUTH | {"Ha.laplace.fwhm": 900.0})
        assert result["Ha"].laplace.fwhm.median == pytest.approx(900.0, rel=0.05)
        frame = result.frames["b"]
        assert frame.noise.scale.median == 1.0
        assert frame.continuum.coefficient(0).median == pytest.approx(1.0, rel=0.1)
        with pytest.raises(AttributeError):
            _ = frame.noise.rho

    def test_summary_text(self) -> None:
        text = single().summary()
        for fragment in (
            "Sampling",
            "verdict: CONVERGED",
            "Posterior",
            "PSIS-LOO",
            "Pareto k",
        ):
            assert fragment in text


class TestDecompose:
    """Model bands, residuals and figures (T11b)."""

    def test_bands_and_residual(self) -> None:
        result = single()
        frame = result.decompose(draws=100, oversample=4)["frame_0"]
        assert frame.grid.size == 4 * frame.n_fitted
        assert np.all(frame.total.lower <= frame.total.median)
        assert np.all(frame.total.median <= frame.total.upper)
        # Medians are not additive, but nearly so for a tight posterior.
        np.testing.assert_allclose(
            frame.continuum.median + frame.components["Ha"].median,
            frame.total.median,
            rtol=1e-2,
        )
        assert frame.chi2 == pytest.approx(np.nansum(frame.residual**2))
        assert frame.chi2 / frame.n_fitted == pytest.approx(1.0, abs=0.25)

    def test_intrinsic_components_differ_from_total(self) -> None:
        frame = single().decompose(view="intrinsic", draws=50)["frame_0"]
        assert (
            frame.components["Ha"].median.max()
            > (frame.total.median - frame.continuum.median).max()
        )

    def test_plot_axes(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rng = np.random.default_rng(3)
        flux_2d = rng.normal(size=(12, 300)) + 0.1
        obs = halpha_spectrum().frames["frame_0"].obs
        spectrum = NoobSpectrum.from_2d(
            flux_2d,
            np.ones_like(flux_2d),
            obs=obs,
            collapse_window=(4, 8),
            unit="angstrom",
            resolving_power=1600.0,
            z=Z,
        )
        workspace = spectrum.prepare(
            [ha()], continuum=NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)
        )
        figure = (
            fake_result(workspace)
            .decompose(draws=20)
            .plot(continuum="subtract", normalize=True)
        )
        assert len(figure.axes) == 3

    def test_corner(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        figure = single().plot_corner()
        assert figure.axes


class TestAsPrior:
    """Posterior to joint prior (T7)."""

    def test_round_trip_declarations(self) -> None:
        nii = ha().derive("NII", rest=6585.3).flux(ratio_to_base=Uniform(0.0, 1.0))
        workspace = halpha_spectrum(seed=7).prepare(
            [ha(), nii], continuum=NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)
        )
        result = fake_result(workspace, TRUTH | {"NII.flux_ratio_to_base": 0.3})
        priors = result.as_prior(["Ha", "NII"])
        assert priors["NII"].base is priors["Ha"]
        assert priors["NII"].center_rule.value == 0.0
        member = priors["Ha"].fwhm_rule.value
        assert isinstance(member, JointMember)
        joint = member.joint
        assert joint.names == (
            "Ha.delta_v",
            "Ha.fwhm",
            "Ha.flux",
            "NII.flux_ratio_to_base",
        )
        draws = result.idata["posterior"].dataset["Ha.fwhm"].values.reshape(-1)
        assert joint.mean[1] == pytest.approx(np.log(draws).mean())
        other = halpha_spectrum(seed=99).prepare(
            list(priors.values()),
            continuum=NoobContinuum.polynomial(0, lambda_0=LAMBDA_0),
        )
        assert other.joint_priors[0].parameters == joint.names

    def test_base_must_be_included(self) -> None:
        nii = ha().derive("NII", rest=6585.3)
        workspace = halpha_spectrum().prepare(
            [ha(), nii], continuum=NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)
        )
        with pytest.raises(ValueError, match="include 'Ha'"):
            fake_result(workspace, TRUTH).as_prior(["NII"])

    def test_independent_continuum_is_refused(self) -> None:
        spectrum = NoobSpectrum.join([halpha_spectrum(seed=1), halpha_spectrum(seed=2)])
        continuum = NoobContinuum.polynomial(
            0, lambda_0=LAMBDA_0, sharing="independent"
        )
        result = fake_result(spectrum.prepare([ha()], continuum=continuum), TRUTH)
        with pytest.raises(ValueError, match="independent"):
            result.as_prior(["continuum"])

    def test_non_normal_posterior_warns(self) -> None:
        import xarray as xr

        result = single()
        posterior = result.idata["posterior"].dataset
        shape = posterior["Ha.delta_v"].shape
        skewed = np.exp(2.0 * np.random.default_rng(1).normal(size=shape))
        result.idata["posterior"] = xr.DataTree(
            posterior.assign({"Ha.delta_v": (("chain", "draw"), skewed)})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.as_prior(["Ha"])
        assert any(issubclass(w.category, NonNormalPriorWarning) for w in caught)


class TestPersistence:
    """One netCDF file, reattached by fingerprint (T11c)."""

    def test_round_trip(self, tmp_path: object) -> None:
        result = single()
        path = result.save(tmp_path / "fit.nc")  # type: ignore[operator]
        with pytest.raises(FileExistsError):
            result.save(path)
        loaded = result.workspace.load(path)
        assert loaded["Ha"].fwhm.median == result["Ha"].fwhm.median
        assert loaded.sampling.verdict == result.sampling.verdict
        assert loaded.loo.elpd == pytest.approx(result.loo.elpd)

    def test_mismatch_names_the_changed_part(self, tmp_path: object) -> None:
        result = single(NoobContinuum.polynomial(0, lambda_0=LAMBDA_0))
        path = result.save(tmp_path / "fit.nc")  # type: ignore[operator]
        continuum = NoobContinuum.polynomial(0, lambda_0=LAMBDA_0)
        other_data = halpha_spectrum(seed=8).prepare([ha()], continuum=continuum)
        other_model = halpha_spectrum(seed=7).prepare(
            [ha().fwhm(250.0)], continuum=continuum
        )
        with pytest.raises(ValueError, match="the data"):
            other_data.load(path)
        with pytest.raises(ValueError, match="the model") as error:
            other_model.load(path)
        assert "the data" not in str(error.value)
