"""Tests for PyMC workspace sampling and posterior results."""

from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

import noobfriend.inference.spectrum as spectrum_api
import noobfriend.inference.spectrum.workspace as workspace_api
from noobfriend.inference.spectrum import NoobLine, NoobSpectrum, NoobSpectrumModel
from noobfriend.inference.spectrum.workspace.mcmc import (
    MCMCComponentPosterior,
    MCMCComponentPrior,
    MCMCCriteria,
    MCMCDiagnostics,
    MCMCDivergenceWarning,
    MCMCFitResult,
    MCMCInputs,
    MCMCPSISLOO,
    MCMCParameterPosterior,
    MCMCParameterPrior,
    MCMCPosterior,
    MCMCPriors,
    MCMCSampling,
    MCMCWAIC,
)
from noobfriend.inference.spectrum.workspace.mcmc.options import build_mcmc_options

from ._helpers import gaussian

pytest.importorskip("pymc")


def _workspace():
    wavelength = np.linspace(4990.0, 5010.0, 41)
    line = (
        NoobLine("line", obs=5000.0)
        .flux(override=8.0)
        .fwhm(override=180.0)
        .center(delta_v_kms=0.0)
    )
    error = np.full_like(wavelength, 0.1)
    data = 0.5 + gaussian(wavelength, center=5000.0, flux=8.0, fwhm_kms=180.0)
    return NoobSpectrum(data, error, obs=wavelength).prepare(
        [line], continuum_order=0
    ), line


def test_model_sample_is_the_only_default_value_source() -> None:
    public = signature(NoobSpectrumModel.sample).parameters

    assert public["draws"].default == 1000
    assert public["tune"].default == 2000
    assert public["chains"].default == 16
    assert public["cores"].default is None
    assert public["target_accept"].default == pytest.approx(0.95)
    assert public["random_seed"].default == 1729
    assert public["progressbar"].default is True


def test_mcmc_result_types_are_exported_only_by_mcmc_backend() -> None:
    for result_type in (
        MCMCComponentPosterior,
        MCMCComponentPrior,
        MCMCCriteria,
        MCMCDiagnostics,
        MCMCDivergenceWarning,
        MCMCFitResult,
        MCMCInputs,
        MCMCPSISLOO,
        MCMCParameterPosterior,
        MCMCParameterPrior,
        MCMCPosterior,
        MCMCPriors,
        MCMCSampling,
        MCMCWAIC,
    ):
        assert result_type is not None
        assert result_type.__name__ not in spectrum_api.__all__
        assert result_type.__name__ not in workspace_api.__all__


def test_mcmc_options_limit_automatic_cores(monkeypatch) -> None:
    monkeypatch.setattr(
        "noobfriend.inference.spectrum.workspace.mcmc.options.os.cpu_count", lambda: 8
    )

    automatic = build_mcmc_options(
        draws=100,
        tune=200,
        chains=16,
        cores=None,
        target_accept=0.95,
        random_seed=1,
        progressbar=False,
    )
    explicit = build_mcmc_options(
        draws=100,
        tune=200,
        chains=4,
        cores=12,
        target_accept=0.99,
        random_seed=2,
        progressbar=True,
    )

    assert automatic.cores == 8
    assert explicit.cores == 4


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"draws": 0}, ValueError, "positive"),
        ({"draws": 1.5}, TypeError, "nonnegative integer"),
        ({"tune": -1}, ValueError, "nonnegative"),
        ({"chains": 0}, ValueError, "positive"),
        ({"cores": 0}, ValueError, "positive"),
        ({"target_accept": 0.79}, ValueError, "at least 0.8"),
        ({"target_accept": 1.0}, ValueError, "less than 1"),
        ({"target_accept": float("nan")}, ValueError, "finite"),
        ({"random_seed": -1}, ValueError, "nonnegative"),
        ({"progressbar": 1}, TypeError, "bool"),
    ],
)
def test_model_sample_validates_options_before_sampling(
    options, error, message
) -> None:
    workspace, _ = _workspace()

    with pytest.raises(error, match=message):
        workspace.model().sample(**options)


def test_model_sample_uses_built_graph_and_returns_physical_result(monkeypatch) -> None:
    workspace, _ = _workspace()
    model = workspace.model()

    def reject_rebuild(*_args, **_kwargs):
        raise AssertionError("sampling must not rebuild the PyMC model")

    monkeypatch.setattr(
        "noobfriend.inference.spectrum.workspace.mcmc.model.build_workspace_model",
        reject_rebuild,
    )

    result = model.sample(
        draws=50,
        tune=50,
        chains=2,
        cores=1,
        target_accept=0.95,
        random_seed=19,
        progressbar=False,
    )
    component = result.posterior["line"]
    continuum = result.posterior["continuum"]

    assert isinstance(result, MCMCFitResult)
    assert result.posterior.components == ("line", "continuum")
    assert component.parameters == ("flux", "fwhm", "center")
    assert continuum.parameters == ("c",)
    assert component["flux"].samples.shape == (2, 50)
    assert component["flux"].samples.flags.writeable is False
    assert component["flux"].median == pytest.approx(8.0)
    assert component["fwhm"].median == pytest.approx(180.0)
    assert component["center"].median == pytest.approx(5000.0)

    assert result.inputs.data is workspace.spectrum
    assert result.inputs.workspace is workspace
    assert result.inputs.options.target_accept == pytest.approx(0.95)
    assert result.inputs.options.cores == 1
    assert result.inputs.priors.components == result.posterior.components
    assert result.inputs.priors["line"].parameters == component.parameters
    assert result.inputs.priors["continuum"].parameters == continuum.parameters

    assert result.sampling.diagnostics.divergences >= 0
    assert np.isfinite(result.sampling.diagnostics.max_rhat)
    assert result.sampling.elapsed_seconds > 0.0
    assert np.isfinite(result.criteria.psis_loo.elpd)
    assert np.isfinite(result.criteria.waic.elpd)
    assert result.criteria.psis_loo.pointwise.shape == (41,)
    assert result.criteria.psis_loo.pareto_k.shape == (41,)
    assert result.criteria.waic.pointwise.shape == (41,)
    assert result.criteria.psis_loo.scale == "log"
    assert result.criteria.waic.scale == "log"
    assert "log_likelihood" in result.idata.children

    assert "line" in repr(result.posterior)
    assert "continuum" in repr(result.posterior)
    assert "flux" in repr(component)
    assert "fwhm" in component._repr_html_()
    assert "continuum" in result.posterior._repr_html_()
    with pytest.raises(KeyError, match="valid components: line, continuum"):
        result.posterior["missing"]
    with pytest.raises(KeyError, match="valid parameters: flux, fwhm, center"):
        result.posterior["line"]["missing"]

    figure = result.plot(model_oversample=4, posterior_draws=50, size=500)
    assert len(figure.axes) == 2
    axis = figure.axes[-2]
    residual_axis = figure.axes[-1]
    lines = {line.get_label(): line for line in axis.get_lines()}
    assert lines["data"].get_color().lower() == "#1a1a1a"
    assert lines["continuum"].get_color().lower() == "#7f7f7f"
    assert lines["total + 94% HDI"].get_color().lower() == "#d62728"
    assert lines["total + 94% HDI"].get_xdata().size == 161
    assert residual_axis.get_ylabel() == "Residual"
    residual_points = next(
        line for line in residual_axis.get_lines() if line.get_marker() == "o"
    )
    model_on_data = np.interp(
        workspace.spectrum.obs,
        lines["total + 94% HDI"].get_xdata(),
        lines["total + 94% HDI"].get_ydata(),
    )
    assert np.allclose(
        residual_points.get_ydata(),
        workspace.spectrum.flux - model_on_data,
    )
    from matplotlib.container import ErrorbarContainer

    assert any(
        isinstance(container, ErrorbarContainer)
        for container in residual_axis.containers
    )
    assert axis.get_legend() is not None
    assert not figure.legends

    figure_2d = result.plot(
        model_oversample=2,
        posterior_draws=25,
        flux_2d=np.tile(workspace.spectrum.flux, (3, 1)).T,
        dispersion="column",
        size=500,
    )
    assert len(figure_2d.axes) == 3

    corner = result.plot_corner(
        variables=(("line", "flux"), ("line", "fwhm")),
        max_draws=50,
        figsize=(5.0, 5.0),
    )
    pareto = result.plot_pareto_k(size=500)
    assert corner.axes
    assert pareto.axes[0].get_ylabel() == "Pareto k"

    import matplotlib.pyplot as plt

    for output in (figure, figure_2d, corner, pareto):
        plt.close(output)


def test_component_mappings_report_valid_keys() -> None:
    workspace, _ = _workspace()
    model = workspace.model()

    with pytest.raises(KeyError, match="valid components: line, continuum"):
        model.priors["missing"]
    with pytest.raises(KeyError, match="valid parameters: flux, fwhm, center"):
        model.priors["line"]["missing"]


def test_workspace_has_no_parallel_mcmc_entrypoint() -> None:
    workspace, _ = _workspace()

    assert not hasattr(workspace, "mcmc")


def test_kernel_line_builds_graph_variables_and_prior_surface() -> None:
    from noobfriend.inference.spectrum.line import kernels

    wavelength = np.linspace(4900.0, 5100.0, 201)
    narrow = NoobLine("n", obs=5000.0)
    broad = (
        NoobLine("b", obs=5000.0, component="broad")
        .fwhm(override=(800.0, 5000.0))
        .convolve(kernels.laplace, fwhm=(200.0, 8000.0))
    )
    spectrum = NoobSpectrum(
        np.ones_like(wavelength),
        np.full_like(wavelength, 0.1),
        obs=wavelength,
        resolving_power=1600.0,
    )
    model = spectrum.prepare([narrow, broad]).model()

    names = {variable.name for variable in model.pymc_model.free_RVs}
    assert "source__1__b__log_laplace__fwhm" in names
    assert "source__1__b__log_effective_fwhm" in names
    assert "source__1__b__log_laplace__fraction" not in names  # fixed 1.0 default

    bounded = (
        NoobLine("b2", obs=5000.0, component="broad")
        .fwhm(override=(800.0, 5000.0))
        .convolve(kernels.laplace, fwhm=(200.0, 8000.0), fraction=(0.2, 1.0))
    )
    model2 = spectrum.prepare([narrow, bounded]).model()
    names2 = {variable.name for variable in model2.pymc_model.free_RVs}
    assert "source__1__b2__log_laplace__fraction" in names2
    prior = model2.priors["b2"]["laplace__fraction"]
    assert prior.family == "loguniform"
    assert prior.bounds == (0.2, 1.0)

    prior = model.priors["b"]["laplace__fwhm"]
    assert prior.family == "loguniform"
    assert prior.bounds == (200.0, 8000.0)
    assert "laplace__fwhm" in model.priors["b"].parameters
