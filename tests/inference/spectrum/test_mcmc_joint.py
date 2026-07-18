"""Tests for joint multi-frame spectrum MCMC (shared lines, per-frame continuum)."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import (
    NoobLine,
    NoobSpectrum,
    NoobSpectrumSet,
)

from ._helpers import gaussian

pytest.importorskip("pymc")

_REST_UM = 0.65646
_Z = 5.228
_CENTER = _REST_UM * (1.0 + _Z)


def _frame(
    *,
    seed: int,
    continuum_offset: float,
    slope: float = 0.3,
    resolving_power: float = 1600.0,
    n: int = 160,
) -> NoobSpectrum:
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(3.95, 4.17, n)
    line = gaussian(
        wavelength,
        center=_CENTER,
        flux=1.5e-3,
        fwhm_kms=350.0,
        resolving_power=resolving_power,
    )
    continuum = continuum_offset + slope * (wavelength - _CENTER)
    flux = continuum + line + rng.normal(0.0, 8e-3, n)
    return NoobSpectrum(
        flux,
        np.full(n, 8e-3),
        obs=wavelength,
        z=_Z,
        unit="um",
        resolving_power=resolving_power,
    )


def _line() -> NoobLine:
    return NoobLine("Ha", rest=_REST_UM, z=_Z, unit="um")


def _rv_names(model, prefix: str) -> list[str]:
    return sorted(v.name for v in model.free_RVs if v.name.startswith(prefix))


def _det_names(model, prefix: str) -> list[str]:
    return sorted(d.name for d in model.deterministics if d.name.startswith(prefix))


def test_pooled_continuum_builds_non_centered_hierarchy() -> None:
    joint = NoobSpectrumSet(
        [_frame(seed=1, continuum_offset=0.05), _frame(seed=2, continuum_offset=0.09)]
    )
    model = joint.prepare([_line()], continuum_sharing="pooled").model().pymc_model

    assert _rv_names(model, "continuum") == [
        "continuum__c__mu_normalized",
        "continuum__c__tau_normalized",
        "continuum__c__z",
        "continuum__k1__mu_normalized",
        "continuum__k1__tau_normalized",
        "continuum__k1__z",
    ]
    assert _det_names(model, "continuum") == [
        "continuum__c",
        "continuum__c__mu",
        "continuum__c__tau",
        "continuum__k1",
        "continuum__k1__mu",
        "continuum__k1__tau",
    ]
    assert list(model.coords["frame"]) == ["frame_0", "frame_1"]


def test_independent_and_shared_continuum_variable_names() -> None:
    joint = NoobSpectrumSet(
        [_frame(seed=1, continuum_offset=0.05), _frame(seed=2, continuum_offset=0.09)]
    )

    independent = (
        joint.prepare([_line()], continuum_sharing="independent").model().pymc_model
    )
    shared = joint.prepare([_line()], continuum_sharing="shared").model().pymc_model

    # Both keep one raw normal per coefficient; independent carries a frame dim.
    assert _rv_names(independent, "continuum") == [
        "continuum__c__normalized",
        "continuum__k1__normalized",
    ]
    assert independent.named_vars_to_dims["continuum__c"] == ("frame",)
    assert _rv_names(shared, "continuum") == [
        "continuum__c__normalized",
        "continuum__k1__normalized",
    ]
    assert "continuum__c" not in shared.named_vars_to_dims


def test_single_frame_set_keeps_scalar_continuum() -> None:
    # A one-frame set must produce the exact single-spectrum continuum RVs.
    single = _frame(seed=1, continuum_offset=0.05)
    direct = single.prepare([_line()]).model().pymc_model
    wrapped = NoobSpectrumSet([single]).prepare([_line()]).model().pymc_model

    assert _rv_names(wrapped, "continuum") == _rv_names(direct, "continuum")
    assert _rv_names(wrapped, "continuum") == [
        "continuum__c__normalized",
        "continuum__k1__normalized",
    ]
    assert "continuum__c" not in wrapped.named_vars_to_dims


@pytest.fixture(scope="module")
def joint_result():
    joint = NoobSpectrumSet(
        {
            "exp_1": _frame(seed=1, continuum_offset=0.05),
            "exp_2": _frame(seed=2, continuum_offset=0.09),
        }
    )
    return (
        joint.prepare([_line()], continuum_sharing="pooled")
        .model()
        .sample(
            draws=200,
            tune=400,
            chains=2,
            cores=1,
            target_accept=0.95,
            random_seed=17,
            progressbar=False,
        )
    )


def test_joint_pooled_sample_recovers_per_frame_continuum(joint_result) -> None:
    result = joint_result
    continuum = result.posterior["continuum"]
    assert continuum.parameters == (
        "c__mu",
        "c__tau",
        "c[exp_1]",
        "c[exp_2]",
        "k1__mu",
        "k1__tau",
        "k1[exp_1]",
        "k1[exp_2]",
    )
    # Per-frame offsets recovered and ordered like the injected values.
    assert continuum["c[exp_1]"].median == pytest.approx(0.05, abs=0.01)
    assert continuum["c[exp_2]"].median == pytest.approx(0.09, abs=0.01)
    assert continuum["c[exp_1]"].median < continuum["c[exp_2]"].median

    # Shared line parameters recovered.
    line = result.posterior["Ha"]
    assert line["flux"].median == pytest.approx(1.5e-3, rel=0.25)
    assert line["fwhm"].median == pytest.approx(350.0, rel=0.35)

    # Pixel-level PSIS-LOO spans both frames' concatenated pixels.
    log_likelihood = result.idata.log_likelihood["observed_flux_normalized"]
    assert log_likelihood.shape[-1] == 320
    assert np.isfinite(result.criteria.psis_loo.elpd)

    # Provenance is the joined frame set.
    assert isinstance(result.inputs.data, NoobSpectrumSet)
    assert result.inputs.data.frame_ids == ("exp_1", "exp_2")


def test_frame_fits_expose_per_frame_goodness_of_fit(joint_result) -> None:
    fits = joint_result.frame_fits()

    assert tuple(fit.frame_id for fit in fits) == ("exp_1", "exp_2")
    for fit in fits:
        assert fit.n_pixels == 160
        assert fit.wavelength.shape == (160,)
        assert set(fit.components) == {"Ha"}
        assert np.isfinite(fit.chi_square)
        # model == continuum + the signed line contribution stored per component.
        assert np.allclose(fit.model, fit.components["Ha"])
        # Consistent frames fit the shared model well.
        assert fit.chi_square_per_pixel < 3.0
        # Read-only arrays like the rest of the result surface.
        assert fit.model.flags.writeable is False


def test_joint_plot_renders_one_panel_per_frame(joint_result) -> None:
    import matplotlib.pyplot as plt

    figure = joint_result.plot(title="joint")
    assert len(figure.axes) == 2

    frames_figure = joint_result.plot_frames(show_components=False)
    assert len(frames_figure.axes) == 2

    plt.close("all")


def test_joint_plot_rejects_two_dimensional_overlays(joint_result) -> None:
    with pytest.raises(ValueError, match="single-frame only"):
        joint_result.plot(flux_2d=np.zeros((4, 160)))


def test_single_frame_frame_fits_use_scalar_continuum() -> None:
    result = (
        _frame(seed=5, continuum_offset=0.06)
        .prepare([_line()])
        .model()
        .sample(
            draws=100,
            tune=300,
            chains=2,
            cores=1,
            random_seed=11,
            progressbar=False,
        )
    )

    fits = result.frame_fits()
    assert len(fits) == 1
    assert fits[0].frame_id == "frame_0"
    assert fits[0].n_pixels == 160
    assert np.isfinite(fits[0].chi_square)
