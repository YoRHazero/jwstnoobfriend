"""Tests for robust workspace maximum-likelihood fitting."""

from __future__ import annotations

from inspect import Parameter, signature
from math import log, pi, sqrt

import numpy as np
import pytest

import noobfriend.inference.spectrum as spectrum_api
import noobfriend.inference.spectrum.workspace as workspace_api
from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
from noobfriend.inference.spectrum.workspace.compiler import C_KMS
from noobfriend.inference.spectrum.workspace.mle import MLEFitResult, MLELineResult, MLESolution
from noobfriend.inference.spectrum.workspace.mle.fit import fit_workspace_mle
from noobfriend.inference.spectrum.workspace.mle.options import build_mle_options


def _gaussian(
    wavelength: np.ndarray,
    *,
    center: float,
    flux: float,
    fwhm_kms: float,
    delta_v_kms: float = 0.0,
    resolving_power: float | None = None,
) -> np.ndarray:
    shifted_center = center * (1.0 + delta_v_kms / C_KMS)
    effective_fwhm = fwhm_kms
    if resolving_power is not None:
        effective_fwhm = sqrt(fwhm_kms**2 + (C_KMS / resolving_power) ** 2)
    fwhm_wavelength = shifted_center * effective_fwhm / C_KMS
    sigma = fwhm_wavelength / (2.0 * sqrt(2.0 * log(2.0)))
    template = np.exp(-0.5 * ((wavelength - shifted_center) / sigma) ** 2) / (sigma * sqrt(2.0 * pi))
    return flux * template


def test_workspace_mle_is_the_only_default_value_source() -> None:
    public = signature(NoobFitWorkspace.mle).parameters

    assert public["absorption_bound_multipliers"].default == (0.5, 1.0, 2.0, 5.0)
    assert public["cancellation_threshold"].default == pytest.approx(0.37)
    assert public["relative_likelihood_min"].default == pytest.approx(0.5)
    assert public["random_seed"].default == 1729

    for helper in (fit_workspace_mle, build_mle_options):
        parameters = signature(helper).parameters
        for name in (
            "absorption_bound_multipliers",
            "cancellation_threshold",
            "relative_likelihood_min",
            "random_seed",
        ):
            assert parameters[name].default is Parameter.empty


def test_mle_result_types_are_exported_only_by_the_mle_backend() -> None:
    assert MLEFitResult is not None
    assert MLESolution is not None
    assert MLELineResult is not None
    for name in ("MLEFitResult", "MLESolution", "MLELineResult"):
        assert name not in spectrum_api.__all__
        assert name not in workspace_api.__all__


def test_workspace_mle_recovers_one_free_gaussian_line() -> None:
    wavelength = np.linspace(6500.0, 6630.0, 261)
    line = NoobLine("Ha", obs=6564.61)
    continuum = 1.7e-17 + 2.0e-20 * (wavelength - 6564.61)
    signal = _gaussian(wavelength, center=6564.61, flux=3.2e-15, fwhm_kms=240.0, delta_v_kms=35.0)
    error = np.full_like(wavelength, 1.0e-18)
    noise = np.random.default_rng(4).normal(0.0, error)
    workspace = NoobSpectrum(continuum + signal + noise, error, obs=wavelength).prepare([line])

    result = workspace.mle()
    fitted = result.solution.for_line(line)

    assert isinstance(result, MLEFitResult)
    assert result.candidate_count == 9
    assert result.converged_candidate_count > 0
    assert result.refinement_applied is False
    assert result.absorption_bound_multipliers == (0.5, 1.0, 2.0, 5.0)
    assert result.cancellation_threshold == pytest.approx(0.37)
    assert result.relative_likelihood_min == pytest.approx(0.5)
    assert result.random_seed == 1729
    assert not hasattr(result, "selected_solution")
    assert result.solution.aic == pytest.approx(result.solution.chi2 + 2.0 * 5)
    assert result.solution.bic == pytest.approx(result.solution.chi2 + 5.0 * log(wavelength.size))
    assert fitted.flux == pytest.approx(3.2e-15, rel=0.05)
    assert fitted.fwhm == pytest.approx(240.0, rel=0.08)
    assert fitted.delta_v_kms == pytest.approx(35.0, abs=8.0)
    assert result.solution.model_flux.shape == wavelength.shape
    assert result.solution.model_flux.flags.writeable is False
    assert tuple(result.solution.continuum_parameters) == ("c", "k1")
    assert "Maximum-likelihood fit" in result.summary()
    assert "<strong>MLE AIC:</strong>" in result.summary()
    assert "<strong>MLE BIC:</strong>" in result.summary()

    figure = result.plot(model_oversample=8, size=500)
    assert len(figure.axes) == 2
    axis = figure.axes[-2]
    residual_axis = figure.axes[-1]
    lines = {line.get_label(): line for line in axis.get_lines()}
    assert lines["data"].get_color().lower() == "#1a1a1a"
    assert lines["continuum"].get_color().lower() == "#7f7f7f"
    assert lines["total model"].get_color().lower() == "#d62728"
    assert lines["total model"].get_xdata().size == (wavelength.size - 1) * 8 + 1
    assert residual_axis.get_ylabel() == "Residual"
    residual_points = next(
        line for line in residual_axis.get_lines() if line.get_marker() == "o"
    )
    assert np.allclose(
        residual_points.get_ydata(),
        result.solution.residual * workspace.spectrum.error,
        equal_nan=True,
    )
    from matplotlib.container import ErrorbarContainer

    assert any(
        isinstance(container, ErrorbarContainer)
        for container in residual_axis.containers
    )
    assert axis.get_legend() is not None
    assert not figure.legends

    external_2d = np.tile(workspace.spectrum.flux, (3, 1))
    figure_2d = result.plot(flux_2d=external_2d, size=500)
    assert len(figure_2d.axes) == 3
    without_residual = result.plot(show_residuals=False, size=500)
    assert len(without_residual.axes) == 1

    import matplotlib.pyplot as plt

    plt.close(figure)
    plt.close(figure_2d)
    plt.close(without_residual)


def test_workspace_mle_accepts_sequence_options_and_records_normalized_values() -> None:
    wavelength = np.linspace(6530.0, 6600.0, 141)
    absorption = NoobLine("Ha", obs=6564.61, contribution="absorption")
    continuum = np.full_like(wavelength, 2.0e-17)
    signal = _gaussian(wavelength, center=6564.61, flux=1.0e-15, fwhm_kms=180.0)
    error = np.full_like(wavelength, 1.0e-18)
    workspace = NoobSpectrum(continuum - signal, error, obs=wavelength).prepare([absorption])

    result = workspace.mle(
        absorption_bound_multipliers=[2.0, 0.5],
        cancellation_threshold=None,
        relative_likelihood_min=0.8,
        random_seed=23,
    )
    repeated = workspace.mle(
        absorption_bound_multipliers=[2.0, 0.5],
        cancellation_threshold=None,
        relative_likelihood_min=0.8,
        random_seed=23,
    )

    assert result.absorption_bound_multipliers == (0.5, 2.0)
    assert result.cancellation_threshold is None
    assert result.relative_likelihood_min == pytest.approx(0.8)
    assert result.random_seed == 23
    assert result.candidate_count == 18
    assert result.refinement_applied is False
    assert np.array_equal(result.solution.model_flux, repeated.solution.model_flux)
    assert "relative likelihood min:</strong> 0.8" in result.summary()


def test_workspace_mle_keeps_fixed_and_ratio_rules() -> None:
    wavelength = np.linspace(4920.0, 5040.0, 241)
    base = NoobLine("OIII5007", obs=5008.24).center(delta_v_kms=20.0).fwhm(override=280.0)
    derived = base.derive("OIII4959", obs=4960.30).flux(ratio=0.335)
    continuum = np.full_like(wavelength, 2.0e-17)
    signal = _gaussian(wavelength, center=5008.24, flux=2.5e-15, fwhm_kms=280.0, delta_v_kms=20.0)
    signal += _gaussian(wavelength, center=4960.30, flux=0.335 * 2.5e-15, fwhm_kms=280.0, delta_v_kms=20.0)
    error = np.full_like(wavelength, 1.0e-18)
    workspace = NoobSpectrum(continuum + signal, error, obs=wavelength).prepare([base, derived])

    result = workspace.mle()
    base_fit = result.solution.for_line(base)
    derived_fit = result.solution.for_line(derived)

    assert base_fit.delta_v_kms == pytest.approx(20.0)
    assert base_fit.fwhm == pytest.approx(280.0)
    assert derived_fit.delta_v_kms == pytest.approx(20.0)
    assert derived_fit.fwhm == pytest.approx(280.0)
    assert derived_fit.flux == pytest.approx(0.335 * base_fit.flux)


def test_workspace_mle_preserves_explicit_absorption_flux_bounds() -> None:
    wavelength = np.linspace(6530.0, 6600.0, 141)
    absorption = NoobLine("Ha", obs=6564.61, contribution="absorption").flux(
        override=(0.0, 1.5e-15)
    )
    continuum = np.full_like(wavelength, 2.0e-17)
    signal = _gaussian(wavelength, center=6564.61, flux=1.0e-15, fwhm_kms=180.0)
    error = np.full_like(wavelength, 1.0e-18)
    workspace = NoobSpectrum(continuum - signal, error, obs=wavelength).prepare([absorption])

    result = workspace.mle()
    fitted = result.solution.for_line(absorption)

    assert result.candidate_count == 9
    assert fitted.flux <= 1.5e-15
    assert fitted.flux == pytest.approx(1.0e-15, rel=0.05)


def test_workspace_mle_fits_halpha_nii_broad_and_close_absorption() -> None:
    z = 5.1
    resolving_power = 2600.0
    wavelength = np.arange(39750.0, 40305.0, 5.0)
    ha_center = 6564.61 * (1.0 + z)
    nii6583_center = 6585.27 * (1.0 + z)
    nii6548_center = 6549.86 * (1.0 + z)
    continuum = 3.0e-19 + 1.5e-22 * (wavelength - ha_center)
    signal = _gaussian(
        wavelength,
        center=ha_center,
        flux=6.5e-17,
        fwhm_kms=320.0,
        resolving_power=resolving_power,
    )
    signal += _gaussian(
        wavelength,
        center=ha_center,
        flux=8.8e-17,
        fwhm_kms=1600.0,
        resolving_power=resolving_power,
    )
    signal -= _gaussian(
        wavelength,
        center=ha_center,
        flux=2.8e-17,
        fwhm_kms=80.0,
        delta_v_kms=-10.0,
        resolving_power=resolving_power,
    )
    signal += _gaussian(
        wavelength,
        center=nii6583_center,
        flux=4.8e-17,
        fwhm_kms=260.0,
        resolving_power=resolving_power,
    )
    signal += _gaussian(
        wavelength,
        center=nii6548_center,
        flux=0.335 * 4.8e-17,
        fwhm_kms=260.0,
        resolving_power=resolving_power,
    )
    error = np.full_like(wavelength, 7.0e-20)
    data = continuum + signal + np.random.default_rng(156).normal(0.0, error)

    ha_narrow = NoobLine("Ha", rest=6564.61, z=z)
    nii6583 = ha_narrow.derive("NII6583", rest=6585.27).fwhm(locked=False)
    nii6548 = nii6583.derive("NII6548", rest=6549.86).flux(ratio=0.335)
    ha_broad = (
        ha_narrow.derive(component="broad")
        .center(delta_v_kms=(-300.0, 300.0))
        .fwhm(locked=False)
    )
    ha_absorption = NoobLine("Ha", rest=6564.61, z=z, contribution="absorption")
    workspace = NoobSpectrum(
        data,
        error,
        obs=wavelength,
        z=z,
        resolving_power=resolving_power,
    ).prepare([ha_narrow, nii6583, nii6548, ha_broad, ha_absorption])

    result = workspace.mle()
    solution = result.solution

    assert result.candidate_count == 40
    assert result.converged_candidate_count >= 36
    assert result.refinement_applied is True
    assert result.refinement_converged_count > 0
    assert solution.relative_likelihood >= 0.499999
    assert solution.for_line(ha_narrow).flux == pytest.approx(6.5e-17, rel=0.5)
    assert solution.for_line(nii6583).flux == pytest.approx(4.8e-17, rel=0.5)
    assert solution.for_line(ha_broad).flux == pytest.approx(8.8e-17, rel=0.7)
    assert solution.for_line(ha_absorption).flux == pytest.approx(2.8e-17, rel=0.8)
    assert abs(solution.for_line(ha_narrow).delta_v_kms) < 50.0
    assert abs(solution.for_line(nii6583).delta_v_kms - solution.for_line(ha_narrow).delta_v_kms) < 1e-8

    no_refinement = workspace.mle(
        absorption_bound_multipliers=(1.0,),
        cancellation_threshold=None,
    )
    assert no_refinement.refinement_applied is False


def test_workspace_mle_rejects_non_gaussian_profile_with_resolving_power() -> None:
    wavelength = np.linspace(4900.0, 5100.0, 101)
    spectrum = NoobSpectrum(
        np.ones_like(wavelength),
        np.full_like(wavelength, 0.1),
        obs=wavelength,
        resolving_power=3000.0,
    )
    workspace = spectrum.prepare([NoobLine("line", obs=5000.0, profile="lorentzian")])

    with pytest.raises(NotImplementedError, match="gaussian"):
        workspace.mle()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"absorption_bound_multipliers": []}, ValueError, "must not be empty"),
        ({"absorption_bound_multipliers": "1,2"}, TypeError, "sequence"),
        ({"absorption_bound_multipliers": [0.0]}, ValueError, "positive"),
        ({"absorption_bound_multipliers": [1.0, 1.0]}, ValueError, "unique"),
        ({"absorption_bound_multipliers": [float("inf")]}, ValueError, "finite"),
        ({"cancellation_threshold": -0.1}, ValueError, "between 0 and 1"),
        ({"cancellation_threshold": 1.1}, ValueError, "between 0 and 1"),
        ({"cancellation_threshold": float("nan")}, ValueError, "finite"),
        ({"relative_likelihood_min": 0.0}, ValueError, "greater than 0"),
        ({"relative_likelihood_min": 1.1}, ValueError, "at most 1"),
        ({"relative_likelihood_min": float("nan")}, ValueError, "finite"),
        ({"random_seed": -1}, ValueError, "nonnegative integer"),
        ({"random_seed": 1.5}, TypeError, "nonnegative integer"),
        ({"random_seed": True}, TypeError, "nonnegative integer"),
    ],
)
def test_workspace_mle_validates_advanced_options(options, error, message) -> None:
    wavelength = np.linspace(4990.0, 5010.0, 21)
    workspace = NoobSpectrum(
        np.ones_like(wavelength),
        np.full_like(wavelength, 0.1),
        obs=wavelength,
    ).prepare([NoobLine("line", obs=5000.0)])

    with pytest.raises(error, match=message):
        workspace.mle(**options)
