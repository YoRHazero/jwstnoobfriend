"""Tests for sampling-ready spectrum models."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import (
    NoobLine,
    NoobSpectrum,
    NoobSpectrumModel,
)
from noobfriend.inference.spectrum.workspace import LineHandle

pm = pytest.importorskip("pymc")


def _workspace(*, resolving_power: float | None = None):
    obs = np.linspace(39000.0, 41000.0, 100)
    spectrum = NoobSpectrum(
        np.ones_like(obs),
        np.full_like(obs, 0.1),
        obs=obs,
        z=7.0,
        resolving_power=resolving_power,
    )
    base = NoobLine("OIII", rest=5008.24, z=7.0)
    broad = (
        base.derive(component="broad")
        .center(delta_v_kms=(-200.0, 200.0))
        .fwhm(locked=False)
    )
    return spectrum.prepare(
        [base, broad], continuum_order=2, continuum_lambda_0=40000.0
    )


def test_workspace_model_returns_built_sampling_model() -> None:
    workspace = _workspace()

    model = workspace.model()
    graph = model.pymc_model

    assert isinstance(model, NoobSpectrumModel)
    assert model.workspace is workspace
    assert isinstance(graph, pm.Model)
    assert graph.observed_RVs[0].name == "observed_flux_normalized"
    assert "continuum__c" in graph.named_vars
    assert "continuum__k1" in graph.named_vars
    assert "continuum__k2" in graph.named_vars
    for handle in workspace.handles:
        safe_id = handle.id.replace(".", "_")
        prefix = f"line__{handle.index}__{safe_id}"
        assert f"{prefix}__flux" in graph.named_vars
        assert f"{prefix}__fwhm" in graph.named_vars
        assert f"{prefix}__center" in graph.named_vars
        assert f"{prefix}__delta_v_kms" in graph.named_vars
    logp = graph.compile_logp()(graph.initial_point())
    assert np.isfinite(logp)


def test_workspace_model_uses_data_range_flux_prior_without_mle(monkeypatch) -> None:
    workspace = _workspace()

    def reject_mle(*_args, **_kwargs):
        raise AssertionError("model construction must not run MLE")

    monkeypatch.setattr(type(workspace), "mle", reject_mle)
    model = workspace.model()
    priors = model.priors
    flux_priors = [priors[component]["flux"] for component in workspace.ids]

    assert priors.components == (*workspace.ids, "continuum")
    assert all(prior.family == "halfnormal" for prior in flux_priors)
    assert all(prior.reference > 0.0 for prior in flux_priors)
    assert all(
        prior.scale == pytest.approx(5.0 * prior.reference) for prior in flux_priors
    )


def test_workspace_model_preserves_physical_bounds_in_prior_samples() -> None:
    obs = np.linspace(4950.0, 5050.0, 101)
    line = (
        NoobLine("line", obs=5000.0)
        .flux(override=(1.0, 3.0))
        .fwhm(override=(80.0, 240.0))
        .center(delta_v_kms=(-20.0, 30.0))
    )
    workspace = NoobSpectrum(
        np.ones_like(obs), np.full_like(obs, 0.1), obs=obs
    ).prepare([line])
    model = workspace.model()

    with model.pymc_model:
        prior = pm.sample_prior_predictive(draws=100, random_seed=8)
    prefix = "line__0__line"
    flux = np.asarray(prior.prior[f"{prefix}__flux"])
    fwhm = np.asarray(prior.prior[f"{prefix}__fwhm"])
    center = np.asarray(prior.prior[f"{prefix}__delta_v_kms"])

    assert np.all((1.0 <= flux) & (flux <= 3.0))
    assert np.all((80.0 <= fwhm) & (fwhm <= 240.0))
    assert np.all((-20.0 <= center) & (center <= 30.0))
    assert model.priors["line"]["flux"].family == "uniform"
    assert model.priors["line"]["fwhm"].family == "loguniform"
    assert model.priors["line"]["center"].family == "uniform"


def test_workspace_model_reparameterizes_overlap_and_effective_fwhm() -> None:
    obs = np.linspace(6500.0, 6630.0, 131)
    emission = NoobLine("Ha", obs=6564.61)
    absorption = NoobLine("Ha", obs=6564.61, contribution="absorption")
    workspace = NoobSpectrum(
        np.ones_like(obs),
        np.full_like(obs, 0.1),
        obs=obs,
        resolving_power=2600.0,
    ).prepare([emission, absorption])

    model = workspace.model()
    graph = model.pymc_model

    assert model.reparameterized_flux_pairs == ((workspace.ids[0], workspace.ids[1]),)
    assert model.effective_fwhm_sources == workspace.ids
    assert "flux_pair__0__net_raw" in graph.named_vars
    assert "flux_pair__0__cancellation_raw" in graph.named_vars
    assert any(name.endswith("__log_effective_fwhm") for name in graph.named_vars)
    assert np.isfinite(graph.compile_logp()(graph.initial_point()))


def test_workspace_model_preserves_ratio_and_locked_line_semantics() -> None:
    obs = np.linspace(4920.0, 5040.0, 121)
    base = NoobLine("OIII5007", obs=5008.24)
    derived = base.derive("OIII4959", obs=4960.30).flux(ratio=0.335)
    workspace = NoobSpectrum(
        np.ones_like(obs), np.full_like(obs, 0.1), obs=obs
    ).prepare([base, derived])
    model = workspace.model()

    with model.pymc_model:
        prior = pm.sample_prior_predictive(draws=30, random_seed=12)
    base_prefix = "line__0__OIII5007"
    derived_prefix = "line__1__OIII4959"

    assert np.asarray(prior.prior[f"{derived_prefix}__flux"]) == pytest.approx(
        0.335 * np.asarray(prior.prior[f"{base_prefix}__flux"])
    )
    assert np.asarray(prior.prior[f"{derived_prefix}__fwhm"]) == pytest.approx(
        np.asarray(prior.prior[f"{base_prefix}__fwhm"])
    )
    assert np.asarray(prior.prior[f"{derived_prefix}__delta_v_kms"]) == pytest.approx(
        np.asarray(prior.prior[f"{base_prefix}__delta_v_kms"])
    )
    assert model.priors["OIII4959"]["flux"].family == "ratio"
    assert model.priors["OIII4959"]["flux"].target == "OIII5007"
    assert model.priors["OIII4959"]["fwhm"].family == "locked"
    assert model.priors["OIII4959"]["center"].family == "locked"


def test_workspace_model_supports_non_gaussian_without_resolving_power() -> None:
    obs = np.linspace(4900.0, 5100.0, 101)
    workspace = NoobSpectrum(
        np.ones_like(obs), np.full_like(obs, 0.1), obs=obs
    ).prepare([NoobLine("line", obs=5000.0, profile="lorentzian")])

    model = workspace.model()

    assert isinstance(model, NoobSpectrumModel)
    assert isinstance(model.pymc_model, pm.Model)


@pytest.mark.parametrize("profile", ["exponential", "lorentzian"])
def test_workspace_model_supports_non_gaussian_with_resolving_power(
    profile: str,
) -> None:
    obs = np.linspace(4900.0, 5100.0, 101)
    workspace = NoobSpectrum(
        np.ones_like(obs),
        np.full_like(obs, 0.1),
        obs=obs,
        resolving_power=3000.0,
    ).prepare([NoobLine("line", obs=5000.0, profile=profile)])

    model = workspace.model()

    assert isinstance(model, NoobSpectrumModel)
    graph = model.pymc_model
    point = graph.initial_point()
    assert np.isfinite(graph.compile_logp()(point))
    # The gradient path exercises the custom Faddeeva Op's pullback.
    assert np.all(np.isfinite(graph.compile_dlogp()(point)))


def _probe_handle(profile: str) -> LineHandle:
    """Return a minimal handle carrying only the profile family templates read."""
    line = NoobLine("probe", obs=5007.0, profile=profile)
    return LineHandle(
        id="probe",
        index=0,
        line=line,
        observed_wavelength=5007.0,
        rest_wavelength=None,
        component="narrow",
        contribution="emission",
        profile=profile,
    )


def test_symbolic_profile_matches_numeric_profile_template() -> None:
    """The PyTensor sampling profile must equal the NumPy reconstruction profile.

    Sampling uses ``symbolic_profile`` while every plot, chi-square, and
    cross-validation reconstruction uses ``profile_template``; a silent drift
    between them would make the shown model disagree with the sampled one. Both
    now share one core, so this pins them together across every profile family,
    the instrumental-LSF fold, and the convolution-kernel branches (including
    the numeric zero-weight-branch pruning that the symbolic path omits).
    """
    pt = pytest.importorskip("pytensor.tensor")

    from noobfriend.inference.spectrum.workspace.compiler import (
        C_KMS,
        profile_template,
        symbolic_profile,
    )

    # Reaches deep into the wings (~70 sigma), where the EMG closed form used to
    # produce 0 * inf = NaN; both paths must stay finite and agree there.
    wavelength = np.linspace(4847.0, 5167.0, 321)
    center = 5007.0
    fwhm = 320.0

    def numeric(profile, *, resolving_power=None, kernels=()):
        return profile_template(
            wavelength,
            handle=_probe_handle(profile),
            center=center,
            fwhm_kms=fwhm,
            resolving_power=resolving_power,
            kernels=kernels,
        )

    def symbolic(profile, *, resolving_power=None, kernels=()):
        expression = symbolic_profile(
            pt,
            profile=profile,
            wavelength=wavelength,
            center=pt.as_tensor_variable(float(center)),
            fwhm_kms=pt.as_tensor_variable(float(fwhm)),
            kernels=tuple(
                (
                    kind,
                    pt.as_tensor_variable(float(width)),
                    pt.as_tensor_variable(float(fraction)),
                )
                for kind, width, fraction in kernels
            ),
            instrumental_fwhm_kms=(
                None
                if resolving_power is None
                else pt.as_tensor_variable(C_KMS / resolving_power)
            ),
        )
        return np.asarray(expression.eval())

    def agree(num, sym):
        assert np.all(np.isfinite(num)), "numeric profile has non-finite values"
        assert np.all(np.isfinite(sym)), "symbolic profile has non-finite values"
        np.testing.assert_allclose(num, sym, rtol=1e-7, atol=1e-12)

    for profile in ("gaussian", "lorentzian", "exponential"):
        agree(numeric(profile), symbolic(profile))

    # Instrumental LSF: both paths fold it internally from the same width —
    # quadrature for a gaussian base, the Normal–Laplace closed form for an
    # exponential base, the Voigt profile for a lorentzian base.
    resolving_power = 1600.0
    for profile in ("gaussian", "exponential", "lorentzian"):
        agree(
            numeric(profile, resolving_power=resolving_power),
            symbolic(profile, resolving_power=resolving_power),
        )

    # Convolution kernels: the exact EMG closed form and the branch mixture,
    # including the deep wings that previously overflowed to NaN.
    for kernels in (
        (("laplace", 140.0, 1.0),),
        (("laplace", 140.0, 0.5),),
        (("gaussian", 90.0, 0.6),),
    ):
        agree(
            numeric("gaussian", kernels=kernels),
            symbolic("gaussian", kernels=kernels),
        )


def test_symbolic_voigt_gradients_match_finite_differences() -> None:
    """The Faddeeva Op's closed-form pullback must differentiate correctly.

    Lorentzian + resolving_power sampling rides on the custom Wofz Op, whose
    gradients are hand-derived from ``w'(z) = -2 z w(z) + 2i / sqrt(pi)``; a
    sign or factor slip there would silently bias every NUTS trajectory, so
    check them against finite differences in all three Voigt arguments.
    """
    pytest.importorskip("pytensor.tensor")
    import pytensor.tensor as pt
    from pytensor.gradient import verify_grad

    from noobfriend.inference.spectrum.workspace.compiler._faddeeva import (
        symbolic_voigt,
    )

    rng = np.random.default_rng(0)
    delta = np.linspace(-40.0, 40.0, 81)
    verify_grad(
        lambda offsets, sigma, gamma: symbolic_voigt(pt, offsets, sigma, gamma),
        [delta, np.float64(1.3), np.float64(2.1)],
        rng=rng,
    )
