"""Tests for NoobContinuum declarations and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.continuum import NoobContinuum
from noobfriend.inference.spectrum.priors import Normal

XP = numpy_namespace()


def test_direct_construction_is_rejected() -> None:
    with pytest.raises(TypeError):
        NoobContinuum()


def test_lambda_0_is_required() -> None:
    with pytest.raises(TypeError):
        NoobContinuum.polynomial(1)  # type: ignore[call-arg]


def test_polynomial_evaluates_monomials_about_lambda_0() -> None:
    continuum = NoobContinuum.polynomial(2, lambda_0=6500.0)
    wavelength = np.array([6400.0, 6500.0, 6610.0])
    x = wavelength - 6500.0
    expected = 1.5 + 0.01 * x - 2e-5 * x**2
    np.testing.assert_allclose(
        continuum.evaluate(XP, wavelength, [1.5, 0.01, -2e-5]), expected
    )


def test_power_law_evaluates_about_lambda_0() -> None:
    continuum = NoobContinuum.power_law(lambda_0=6500.0)
    wavelength = np.array([6000.0, 6500.0, 7000.0])
    expected = 2.0 * (wavelength / 6500.0) ** -1.7
    np.testing.assert_allclose(
        continuum.evaluate(XP, wavelength, [2.0, -1.7]), expected
    )


def test_coefficient_declarations() -> None:
    continuum = NoobContinuum.polynomial(1, lambda_0=6500.0).coefficient(
        1, Normal(0.0, 1.0)
    )
    assert dict(continuum.declarations) == {"c1": Normal(0.0, 1.0)}
    with pytest.raises(ValueError, match="degree"):
        continuum.coefficient(2, 0.0)
    with pytest.raises(ValueError, match="level"):
        continuum.level(1.0)


def test_power_law_level_is_positive() -> None:
    with pytest.raises(ValueError):
        NoobContinuum.power_law(lambda_0=6500.0).level(-1.0)


def test_sharing_is_validated() -> None:
    with pytest.raises(ValueError, match="sharing"):
        NoobContinuum.polynomial(1, lambda_0=6500.0, sharing="tied")  # type: ignore[arg-type]
