"""Tests for declarative parameter priors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from noobfriend.inference.spectrum import NoobLine, NoobSpectrum
from noobfriend.inference.spectrum.line import (
    LogUniform,
    Normal,
    Prior,
    TruncatedNormal,
    Uniform,
    prior_support,
)
from noobfriend.inference.spectrum.workspace.compiler import (
    NORMAL_RANGE_SIGMAS,
    compile_line_graph,
    prior_initial,
    prior_range,
    scale_prior,
)


def _spectrum(resolving_power: float | None = None) -> NoobSpectrum:
    wavelength = np.linspace(39000.0, 41000.0, 300)
    return NoobSpectrum(
        np.ones_like(wavelength),
        np.full_like(wavelength, 0.05),
        obs=wavelength,
        resolving_power=resolving_power,
    )


# --------------------------------------------------------------------------
# distribution value objects
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Normal(0.0, 0.0), "sigma must be positive"),
        (lambda: Normal(float("inf"), 1.0), "mu must be finite"),
        (lambda: TruncatedNormal(1.0, 1.0, low=5.0, high=5.0), "must be increasing"),
        (lambda: Uniform(5.0, 1.0), "must be increasing"),
        (lambda: LogUniform(0.0, 5.0), "low must be positive"),
    ],
)
def test_distributions_reject_degenerate_parameters(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_distribution_fields_are_coerced_to_float() -> None:
    prior = TruncatedNormal(1200, 400, low=800, high=5000)

    assert isinstance(prior.mu, float)
    assert isinstance(prior.low, float)
    assert prior_support(prior) == (800.0, 5000.0)
    assert prior_support(Normal(0, 120)) == (float("-inf"), float("inf"))


# --------------------------------------------------------------------------
# declaration layer: which prior each parameter's domain admits
# --------------------------------------------------------------------------


def test_every_parameter_accepts_a_declared_prior() -> None:
    line = (
        NoobLine("Ha", obs=40000.0)
        .center(delta_v_kms=Normal(0.0, 120.0))
        .fwhm(override=TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0))
        .flux(override=LogUniform(1e-18, 1e-16))
        .convolve(
            "laplace",
            fwhm=TruncatedNormal(800.0, 200.0, low=100.0, high=3000.0),
            fraction=Uniform(0.2, 1.0),
        )
    )

    assert line.center_rule.is_prior
    assert line.fwhm_rule.is_prior
    assert line.flux_rule.is_prior
    assert all(rule.is_prior for _, rule in line.kernels[0].rules)


@pytest.mark.parametrize(
    "declare",
    [
        lambda line: line.fwhm(override=Normal(1200.0, 400.0)),
        lambda line: line.flux(override=Normal(1e-17, 3e-18)),
        lambda line: line.convolve("laplace", fwhm=Normal(800.0, 200.0), fraction=1.0),
    ],
)
def test_unbounded_normal_is_refused_outside_the_real_line(declare) -> None:
    """Widths, fluxes, and fractions are positive; only a velocity offset is not."""
    with pytest.raises(ValueError, match="unbounded Normal prior"):
        declare(NoobLine("Ha", obs=40000.0))


def test_positive_parameters_reject_a_prior_reaching_below_zero() -> None:
    line = NoobLine("Ha", obs=40000.0)

    with pytest.raises(ValueError, match="support must be positive"):
        line.fwhm(override=Uniform(-5.0, 400.0))
    with pytest.raises(ValueError, match="support must be nonnegative"):
        line.flux(override=Uniform(-1.0, 1.0))
    with pytest.raises(ValueError, match=r"support must lie in \(0, 1\]"):
        line.convolve(
            "laplace", fwhm=LogUniform(100.0, 3000.0), fraction=Uniform(0.5, 2.0)
        )


def test_fast_path_keeps_its_original_meaning() -> None:
    line = NoobLine("Ha", obs=40000.0)

    assert line.fwhm(override=180.0).fwhm_rule.is_fixed
    assert line.fwhm(override=(80.0, 400.0)).fwhm_rule.is_bounded
    assert not line.fwhm(override=(80.0, 400.0)).fwhm_rule.is_prior


# --------------------------------------------------------------------------
# compilation: prior -> optimizer range and representative value
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prior", "expected_range", "expected_initial"),
    [
        (Normal(0.0, 120.0), (-360.0, 360.0), 0.0),
        (
            TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0),
            (800.0, 5000.0),
            1200.0,
        ),
        (TruncatedNormal(100.0, 400.0, low=800.0, high=5000.0), (800.0, 5000.0), 800.0),
        (Uniform(0.2, 1.0), (0.2, 1.0), 0.6),
        (LogUniform(100.0, 10000.0), (100.0, 10000.0), 1000.0),
    ],
)
def test_prior_range_and_initial(
    prior: Prior, expected_range: tuple[float, float], expected_initial: float
) -> None:
    assert prior_range(prior) == pytest.approx(expected_range)
    assert prior_initial(prior) == pytest.approx(expected_initial)


def test_unbounded_normal_range_uses_three_sigma() -> None:
    lower, upper = prior_range(Normal(50.0, 20.0))

    assert (upper - lower) / 2.0 == pytest.approx(NORMAL_RANGE_SIGMAS * 20.0)


def test_scale_prior_rescales_every_value() -> None:
    scaled = scale_prior(TruncatedNormal(10.0, 2.0, low=1.0, high=20.0), 3.0)

    assert scaled == TruncatedNormal(30.0, 6.0, low=3.0, high=60.0)
    with pytest.raises(ValueError, match="must be positive"):
        scale_prior(Uniform(1.0, 2.0), 0.0)


def test_compiled_spec_carries_the_declared_prior() -> None:
    line = (
        NoobLine("Ha", obs=40000.0, component="broad")
        .center(delta_v_kms=Normal(0.0, 120.0))
        .fwhm(override=TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0))
    )
    graph = compile_line_graph(_spectrum().prepare([line]))

    center = next(iter(graph.center_sources.values()))
    fwhm = next(iter(graph.fwhm_sources.values()))
    assert center.prior == Normal(0.0, 120.0)
    assert (center.lower, center.upper) == pytest.approx((-360.0, 360.0))
    assert fwhm.prior == TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0)
    assert fwhm.initial == pytest.approx(1200.0)


def test_wavelength_center_prior_is_converted_to_velocity() -> None:
    """A delta_wavelength prior compiles in km/s, like a delta_wavelength range."""
    line = NoobLine("Ha", obs=40000.0).center(
        delta_wavelength=TruncatedNormal(0.0, 2.0, low=-8.0, high=8.0)
    )
    graph = compile_line_graph(_spectrum().prepare([line]))

    spec = next(iter(graph.center_sources.values()))
    factor = 299792.458 / 40000.0
    assert spec.prior == TruncatedNormal(
        0.0, 2.0 * factor, low=-8.0 * factor, high=8.0 * factor
    )
    assert (spec.lower, spec.upper) == pytest.approx((-8.0 * factor, 8.0 * factor))


def test_compiled_spec_without_a_prior_is_unchanged() -> None:
    line = NoobLine("Ha", obs=40000.0).fwhm(override=(80.0, 400.0))
    graph = compile_line_graph(_spectrum().prepare([line]))

    assert next(iter(graph.fwhm_sources.values())).prior is None


# --------------------------------------------------------------------------
# maximum likelihood: a prior contributes its range, never its shape
# --------------------------------------------------------------------------


def test_mle_problem_boxes_a_prior_by_its_range() -> None:
    from noobfriend.inference.spectrum.workspace.mle.problem import build_mle_problem

    line = (
        NoobLine("Ha", obs=40000.0, component="broad")
        .center(delta_v_kms=Normal(0.0, 120.0))
        .fwhm(override=TruncatedNormal(1400.0, 300.0, low=800.0, high=5000.0))
    )
    problem = build_mle_problem(
        _spectrum().prepare([line]), absorption_bound_multiplier=1.0
    )

    assert problem.lower[problem.center_slice] == pytest.approx([-360.0])
    assert problem.upper[problem.center_slice] == pytest.approx([360.0])
    assert problem.lower[problem.fwhm_slice] == pytest.approx([800.0])
    assert problem.upper[problem.fwhm_slice] == pytest.approx([5000.0])
    assert problem.initial[problem.fwhm_slice] == pytest.approx([1400.0])


def test_flux_prior_supplies_the_optimizer_flux_bounds() -> None:
    from noobfriend.inference.spectrum.workspace.compiler import flux_bounds

    line = NoobLine("Ha", obs=40000.0).flux(
        override=TruncatedNormal(1e-17, 3e-18, low=0.0, high=1e-16)
    )

    assert flux_bounds(line) == pytest.approx((0.0, 1e-16))


# --------------------------------------------------------------------------
# sampling: the declared prior is what the model actually imposes
# --------------------------------------------------------------------------


#: Quantiles compared against a declared prior, and the tolerance allowed on
#: them. Measured deviations sit near 1%; an omitted or mis-signed Jacobian
#: term shifts the recovered distribution by tens of percent.
_CHECKED_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_QUANTILE_TOLERANCE = 0.04


def _assert_matches_prior(samples: np.ndarray, reference) -> None:
    """Compare a sampled marginal against the distribution it should be.

    Quantiles rather than a Kolmogorov-Smirnov test: posterior draws are
    autocorrelated, so a KS test run at its nominal sample size rejects a
    correct distribution.
    """
    assert samples.mean() == pytest.approx(reference.mean(), rel=0.03)
    assert np.quantile(samples, _CHECKED_QUANTILES) == pytest.approx(
        reference.ppf(_CHECKED_QUANTILES), rel=_QUANTILE_TOLERANCE
    )


def _prior_only_samples(prior: Prior, resolving_power: float | None) -> np.ndarray:
    """Sample a width whose likelihood carries no information about it.

    Fixing the line flux at zero removes the line from the model's mean, so
    the width's posterior is exactly its prior. That turns a prior into
    something a sampler can be asked to reproduce.
    """
    line = (
        NoobLine("Ha", obs=40000.0, component="broad")
        .fwhm(override=prior)
        .flux(override=0.0)
    )
    result = (
        _spectrum(resolving_power)
        .prepare([line], continuum_order=0)
        .model()
        .sample(draws=3000, tune=1000, chains=4, random_seed=1729, progressbar=False)
    )
    return result.posterior["Ha"]["fwhm"].samples.reshape(-1)


@pytest.mark.slow
@pytest.mark.parametrize("resolving_power", [1600.0, None])
def test_declared_normal_prior_is_recovered(resolving_power: float | None) -> None:
    """A truncated normal survives the effective-width reparameterization.

    With ``resolving_power`` set the width is sampled in effective-width
    coordinates, so the declared prior is imposed through a Jacobian
    correction rather than by a PyMC distribution. Recovering the same
    distribution on both paths is what shows that correction is right.
    """
    prior = TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0)
    reference = stats.truncnorm(
        a=(prior.low - prior.mu) / prior.sigma,
        b=(prior.high - prior.mu) / prior.sigma,
        loc=prior.mu,
        scale=prior.sigma,
    )

    samples = _prior_only_samples(prior, resolving_power)

    assert samples.std() == pytest.approx(reference.std(), rel=0.05)
    _assert_matches_prior(samples, reference)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("prior", "reference"),
    [
        (
            LogUniform(800.0, 5000.0),
            stats.loguniform(800.0, 5000.0),
        ),
        (
            Uniform(800.0, 5000.0),
            stats.uniform(loc=800.0, scale=4200.0),
        ),
    ],
)
def test_declared_range_priors_are_recovered(prior: Prior, reference) -> None:
    samples = _prior_only_samples(prior, resolving_power=1600.0)

    _assert_matches_prior(samples, reference)


@pytest.mark.parametrize("resolving_power", [1600.0, None])
def test_log_uniform_reproduces_the_legacy_range_exactly(
    resolving_power: float | None,
) -> None:
    """``LogUniform(lo, hi)`` is the declared spelling of ``(lo, hi)``.

    The historical fast path has always meant a log-uniform width, so naming
    it explicitly must not perturb the model by even one variable or one
    floating-point operation.
    """
    rng = np.random.default_rng(0)
    wavelength = np.linspace(39000.0, 41000.0, 300)
    flux = (
        1.0
        + 2.0 * np.exp(-0.5 * ((wavelength - 40000.0) / 120.0) ** 2)
        + rng.normal(0.0, 0.05, wavelength.size)
    )

    def build(width):
        spectrum = NoobSpectrum(
            flux,
            np.full_like(wavelength, 0.05),
            obs=wavelength,
            resolving_power=resolving_power,
        )
        line = NoobLine("Ha", obs=40000.0, component="broad").fwhm(override=width)
        return spectrum.prepare([line]).model().pymc_model

    legacy = build((800.0, 5000.0))
    declared = build(LogUniform(800.0, 5000.0))

    assert sorted(v.name for v in legacy.free_RVs) == sorted(
        v.name for v in declared.free_RVs
    )
    legacy_logp, declared_logp = legacy.compile_logp(), declared.compile_logp()
    reference_point = legacy.initial_point()
    for _ in range(50):
        point = {
            name: value + rng.normal(0.0, 0.3, np.shape(value))
            for name, value in reference_point.items()
        }
        assert float(legacy_logp(point)) == float(declared_logp(point))


def test_declared_priors_are_reported_back() -> None:
    line = (
        NoobLine("Ha", obs=40000.0, component="broad")
        .center(delta_v_kms=Normal(0.0, 120.0))
        .fwhm(override=TruncatedNormal(1200.0, 400.0, low=800.0, high=5000.0))
        .flux(override=Uniform(0.0, 5.0))
    )
    priors = _spectrum(1600.0).prepare([line]).model().priors["Ha"]

    assert priors["delta_v_kms"].family == "normal"
    assert priors["delta_v_kms"].location == pytest.approx(0.0)
    assert priors["delta_v_kms"].scale == pytest.approx(120.0)
    assert priors["fwhm"].family == "truncatednormal"
    assert priors["fwhm"].bounds == pytest.approx((800.0, 5000.0))
    assert priors["flux"].family == "uniform"
    assert priors["flux"].bounds == pytest.approx((0.0, 5.0))
    # the wavelength centre is an affine image of the declared velocity prior
    assert priors["center"].family == "normal"
    assert priors["center"].location == pytest.approx(40000.0)
