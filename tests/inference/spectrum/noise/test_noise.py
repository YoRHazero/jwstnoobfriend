"""Tests for frame noise models against dense Gaussian references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.noise import AR1, Autocorrelation, FrameNoise, Noise
from noobfriend.inference.spectrum.priors import Uniform

XP = numpy_namespace()
#: Native pixel indices with masked gaps of 2 and 3 pixels.
INDEX = np.array([0, 1, 2, 4, 5, 6, 9, 10, 11, 12])
SIGMA = np.linspace(0.8, 1.6, INDEX.size)
RNG = np.random.default_rng(7)
RESID = RNG.normal(size=(3, INDEX.size))
SCALE = np.array([0.9, 1.0, 1.3])


def dense_covariance(rho_of_lag: np.ndarray, scale: float) -> np.ndarray:
    lags = np.abs(INDEX[:, None] - INDEX[None, :])
    return scale**2 * np.outer(SIGMA, SIGMA) * rho_of_lag[lags]


def reference(
    covariance: np.ndarray, resid: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Return (log likelihood, LOO conditionals, chi-square) by brute force."""
    log_likelihood = stats.multivariate_normal(cov=covariance).logpdf(resid)
    precision = np.linalg.inv(covariance)
    conditionals = np.empty(resid.size)
    for i in range(resid.size):
        keep = np.arange(resid.size) != i
        gain = covariance[i, keep] @ np.linalg.inv(covariance[np.ix_(keep, keep)])
        mean = gain @ resid[keep]
        variance = covariance[i, i] - gain @ covariance[keep, i]
        conditionals[i] = stats.norm(mean, np.sqrt(variance)).logpdf(resid[i])
    return log_likelihood, conditionals, float(resid @ precision @ resid)


@pytest.mark.parametrize("rho", [0.6, -0.45])
def test_ar1_matches_dense_gaussian(rho: float) -> None:
    frame = FrameNoise.build(
        Noise(scale=Uniform(0.1, 5.0), correlation=AR1(Uniform())), SIGMA, INDEX
    )
    rho_of_lag = rho ** np.arange(INDEX[-1] + 1)
    log_likelihood = frame.log_likelihood(XP, RESID, SCALE, np.full(3, rho))
    pointwise = frame.pointwise(XP, RESID, SCALE, np.full(3, rho))
    chi2 = np.sum(frame.whiten(XP, RESID, SCALE, np.full(3, rho)) ** 2, axis=-1)
    for draw in range(3):
        expected = reference(dense_covariance(rho_of_lag, SCALE[draw]), RESID[draw])
        assert log_likelihood[draw] == pytest.approx(expected[0])
        np.testing.assert_allclose(pointwise[draw], expected[1])
        assert chi2[draw] == pytest.approx(expected[2])


def test_fixed_autocorrelation_matches_dense_gaussian() -> None:
    correlation = Autocorrelation((0.4, 0.15))
    frame = FrameNoise.build(
        Noise(scale=Uniform(0.1, 5.0), correlation=correlation), SIGMA, INDEX
    )
    rho_of_lag = correlation.correlations(np.arange(INDEX[-1] + 1))
    log_likelihood = frame.log_likelihood(XP, RESID, SCALE)
    pointwise = frame.pointwise(XP, RESID, SCALE)
    chi2 = np.sum(frame.whiten(XP, RESID, SCALE) ** 2, axis=-1)
    for draw in range(3):
        expected = reference(dense_covariance(rho_of_lag, SCALE[draw]), RESID[draw])
        assert log_likelihood[draw] == pytest.approx(expected[0])
        np.testing.assert_allclose(pointwise[draw], expected[1])
        assert chi2[draw] == pytest.approx(expected[2])


def test_white_noise_factorizes() -> None:
    frame = FrameNoise.build(Noise(), SIGMA, INDEX)
    expected = stats.norm(0.0, 1.3 * SIGMA).logpdf(RESID[2])
    np.testing.assert_allclose(frame.pointwise(XP, RESID[2], 1.3), expected)
    assert frame.log_likelihood(XP, RESID[2], 1.3) == pytest.approx(expected.sum())


def test_numpyro_ar1_matches_frame_likelihood() -> None:
    pytest.importorskip("numpyro")
    from noobfriend.core.jaxruntime import enable_x64

    enable_x64()
    frame = FrameNoise.build(Noise(correlation=AR1(Uniform())), SIGMA, INDEX)
    distribution = frame.distribution(np.zeros(INDEX.size), 1.3, -0.45)
    expected = frame.log_likelihood(XP, RESID[0], 1.3, -0.45)
    assert float(distribution.log_prob(RESID[0])) == pytest.approx(float(expected))


class TestDeclarations:
    """Validation of the noise value objects."""

    def test_autocorrelation_rejects_negative_spectral_density(self) -> None:
        with pytest.raises(ValueError, match="spectral density"):
            Autocorrelation((0.9, 0.9, 0.9))

    def test_ar1_rejects_unit_correlation(self) -> None:
        with pytest.raises(ValueError):
            AR1(1.0)

    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Noise(scale=0.0)

    def test_free_rho_needs_a_value(self) -> None:
        frame = FrameNoise.build(Noise(correlation=AR1(Uniform())), SIGMA, INDEX)
        with pytest.raises(ValueError, match="free"):
            frame.log_likelihood(XP, RESID[0], 1.0)
