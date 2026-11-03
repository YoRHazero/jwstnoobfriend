"""Tests for the maximum-likelihood backend."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import (
    LogUniform,
    Noise,
    NoobContinuum,
    NoobLine,
    NoobSpectrum,
    Uniform,
)
from noobfriend.inference.spectrum._wavelength import C_KMS
from noobfriend.inference.spectrum.mle import NoobMLEResult
from noobfriend.inference.spectrum.priors import (
    NONNEGATIVE,
    POSITIVE,
    REAL,
    JointNormal,
)

from .._helpers import HALPHA, Z, gaussian_line, halpha_spectrum

LAMBDA_0 = HALPHA * (1 + Z)
CONTINUUM = NoobContinuum.polynomial(1, lambda_0=LAMBDA_0)


def ha() -> NoobLine:
    return NoobLine("Ha", rest=HALPHA, unit="angstrom")


@pytest.fixture(scope="module")
def single() -> NoobMLEResult:
    spectrum = halpha_spectrum(seed=3, flux=40.0, fwhm_kms=300.0, delta_v_kms=50.0)
    return spectrum.prepare([ha()], continuum=CONTINUUM).mle()


class TestSolution:
    """Known-truth recovery and error bars (T10.4)."""

    def test_recovers_truth_within_errors(self, single: NoobMLEResult) -> None:
        for name, truth in (("fwhm", 300.0), ("flux", 40.0), ("delta_v", 50.0)):
            estimate = getattr(single["Ha"], name)
            assert abs(estimate.value - truth) < 4 * estimate.stderr, name
            assert not estimate.at_bound
        assert single.reliable
        assert single.solution.reduced_chi2 == pytest.approx(1.0, abs=0.2)

    def test_derived_errors_by_delta_method(self, single: NoobMLEResult) -> None:
        nominal = single.workspace["Ha"].nominal
        center = single["Ha"].center
        delta_v = single["Ha"].delta_v
        assert center.stderr == pytest.approx(
            nominal / C_KMS * delta_v.stderr, rel=1e-4
        )
        index = single.parameter_names.index("Ha.fwhm")
        assert single["Ha.fwhm"].stderr == pytest.approx(
            np.sqrt(single.covariance[index, index])
        )

    def test_summary_and_plot(self, single: NoobMLEResult) -> None:
        import matplotlib

        matplotlib.use("Agg")
        text = single.summary()
        for fragment in (
            "Maximum likelihood",
            "AIC",
            "error bars: reliable",
            "Ha.fwhm",
        ):
            assert fragment in text
        assert len(single.plot().axes) == 2


class TestReadingRules:
    """What the MLE reads from the workspace (T10.2)."""

    def test_noise_model_is_ignored(self) -> None:
        spectrum = halpha_spectrum(seed=4)
        line = [ha().fwhm(LogUniform(100.0, 500.0))]
        plain = spectrum.prepare(line, continuum=CONTINUUM).mle()
        noisy = (
            spectrum.with_noise(Noise(scale=Uniform(0.5, 3.0)))
            .prepare(line, continuum=CONTINUUM)
            .mle()
        )
        assert noisy.solution.chi2 == pytest.approx(plain.solution.chi2)

    def test_joint_prior_is_ignored(self) -> None:
        joint = JointNormal(
            ("Ha.fwhm", "Ha.flux", "Ha.delta_v"),
            (POSITIVE, NONNEGATIVE, REAL),
            np.array([np.log(250.0), np.log(10.0), 0.0]),
            np.diag([1e-4, 1e-4, 1.0]),
        )
        line = ha().fwhm(joint.member("Ha.fwhm")).flux(joint.member("Ha.flux"))
        result = (
            halpha_spectrum(seed=3, flux=40.0)
            .prepare([line], continuum=CONTINUUM)
            .mle()
        )
        assert result["Ha"].flux.value == pytest.approx(40.0, rel=0.05)

    def test_omitted_bounds_are_unbounded(self) -> None:
        line = ha().center(delta_v_kms=Uniform())
        result = (
            halpha_spectrum(seed=5, delta_v_kms=50.0)
            .prepare([line], continuum=CONTINUUM)
            .mle()
        )
        assert result["Ha"].delta_v.value == pytest.approx(50.0, abs=10.0)

    def test_multi_frame_is_refused(self) -> None:
        spectrum = NoobSpectrum.join([halpha_spectrum(seed=1), halpha_spectrum(seed=2)])
        with pytest.raises(ValueError, match="select"):
            spectrum.prepare([ha()], continuum=CONTINUUM).mle()


class TestSelection:
    """Multi-start search, bounds and cancellation (T10.3)."""

    def test_absent_line_sits_on_its_bound(self) -> None:
        # A dip where [N II] would be pushes its flux ratio onto the 0 bound.
        clean = halpha_spectrum(seed=6).frames["frame_0"]
        dip = gaussian_line(
            clean.obs, center=6585.3 * (1 + Z), fwhm_kms=300.0, flux=5.0
        )
        spectrum = NoobSpectrum(
            clean.obs,
            clean.flux - dip,
            clean.error,
            unit="angstrom",
            resolving_power=1600.0,
            z=Z,
        )
        nii = ha().derive("NII", rest=6585.3).flux(ratio_to_base=Uniform(0.0, 1.0))
        result = spectrum.prepare([ha(), nii], continuum=CONTINUUM).mle()
        assert result["NII"].flux_ratio_to_base.at_bound
        assert not result.reliable

    def test_emission_absorption_pair_does_not_cancel(self) -> None:
        absorption = NoobLine(
            "Ha_abs", rest=HALPHA, unit="angstrom", contribution="absorption"
        )
        result = (
            halpha_spectrum(seed=7)
            .prepare([ha(), absorption], continuum=CONTINUUM)
            .mle()
        )
        assert result.solution.cancellation_fraction < 0.37
        assert result["Ha"].flux.value == pytest.approx(40.0, rel=0.1)

    def test_power_law_continuum(self) -> None:
        continuum = NoobContinuum.power_law(lambda_0=LAMBDA_0)
        result = (
            halpha_spectrum(seed=8, continuum=2.0)
            .prepare([ha()], continuum=continuum)
            .mle()
        )
        assert result["continuum"].level.value == pytest.approx(2.0, rel=0.01)
        assert (
            abs(result["continuum"].index.value) < 4 * result["continuum"].index.stderr
        )

    def test_options_are_validated(self) -> None:
        workspace = halpha_spectrum().prepare([ha()], continuum=CONTINUUM)
        with pytest.raises(ValueError, match="relative_likelihood_min"):
            workspace.mle(relative_likelihood_min=0.0)
        with pytest.raises(ValueError, match="multipliers"):
            workspace.mle(absorption_bound_multipliers=(-1.0,))
