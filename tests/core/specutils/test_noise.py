"""Synthetic ground-truth tests for the spectrum noise estimators."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.core.specutils import background_noise, continuum_noise


def ar1_rows(
    rng: np.random.Generator, shape: tuple[int, int], rho: float
) -> np.ndarray:
    """Unit-variance AR(1) along the last axis, independent rows."""
    noise = rng.normal(size=shape)
    out = np.empty(shape)
    out[:, 0] = noise[:, 0]
    width = np.sqrt(1.0 - rho**2)
    for column in range(1, shape[1]):
        out[:, column] = rho * out[:, column - 1] + width * noise[:, column]
    return out


class TestContinuumNoise:
    """1-D estimate of scale and dispersion-axis correlation."""

    def test_recovers_scale_and_ar1_correlation(self) -> None:
        rng = np.random.default_rng(1)
        n = 20_000
        wavelength = np.linspace(1.0, 2.0, n)
        error = np.full(n, 0.5)
        noise = 1.4 * error * ar1_rows(rng, (1, n), 0.5)[0]
        flux = 3.0 + 0.2 * wavelength + noise
        exclude = np.zeros(n, dtype=bool)
        exclude[5000:5100] = True
        scale, rho = continuum_noise(
            flux, error, wavelength=wavelength, exclude=exclude, lags=3
        )
        assert scale == pytest.approx(1.4, rel=0.03)
        np.testing.assert_allclose(rho, [0.5, 0.25, 0.125], atol=0.03)

    def test_too_few_pixels_raises(self) -> None:
        wavelength = np.linspace(1.0, 2.0, 8)
        with pytest.raises(ValueError, match="usable pixels"):
            continuum_noise(np.ones(8), np.ones(8), wavelength=wavelength)


class TestBackgroundNoise:
    """2-D sky autocovariance summed over the collapse window."""

    def test_error_miscalibration_sets_scale(self) -> None:
        rng = np.random.default_rng(2)
        data = 1.3 * rng.normal(size=(40, 3000))
        scale, rho = background_noise(
            data,
            np.ones_like(data),
            window=(18, 23),
            dispersion_axis=1,
            source_mask=np.zeros(data.shape, dtype=bool),
            lags=2,
        )
        assert scale == pytest.approx(1.3, rel=0.03)
        np.testing.assert_allclose(rho, [0.0, 0.0], atol=0.02)

    def test_cross_dispersion_correlation_inflates_scale(self) -> None:
        rng = np.random.default_rng(3)
        white = rng.normal(size=(41, 3000))
        data = (white[:-1] + white[1:]) / np.sqrt(2.0)
        n_rows = 5
        scale, _ = background_noise(
            data,
            np.ones_like(data),
            window=(15, 15 + n_rows),
            dispersion_axis=1,
            source_mask=np.zeros(data.shape, dtype=bool),
        )
        assert scale**2 == pytest.approx((2 * n_rows - 1) / n_rows, rel=0.05)

    def test_dispersion_correlation_with_transposed_input(self) -> None:
        rng = np.random.default_rng(4)
        data = ar1_rows(rng, (40, 3000), 0.6)
        scale, rho = background_noise(
            data.T,
            np.ones_like(data.T),
            window=(18, 22),
            dispersion_axis=0,
            source_mask=np.zeros(data.T.shape, dtype=bool),
            lags=2,
        )
        assert scale == pytest.approx(1.0, rel=0.04)
        np.testing.assert_allclose(rho, [0.6, 0.36], atol=0.04)

    def test_too_little_sky_raises(self) -> None:
        data = np.ones((6, 10))
        with pytest.raises(ValueError, match="background pixels"):
            background_noise(
                data,
                data,
                window=(2, 4),
                dispersion_axis=1,
                source_mask=np.zeros(data.shape, dtype=bool),
            )
