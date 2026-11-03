"""Tests for the closed-form line profiles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from noobfriend.inference.spectrum._units import C_KMS
from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.profiles import (
    GAUSSIAN_FWHM_TO_SIGMA,
    LAPLACE_FWHM_TO_SCALE,
    check_kernels,
    line_profile,
)

XP = numpy_namespace()
CENTER = 6563.0
GRID = np.linspace(CENTER - 400.0, CENTER + 400.0, 400_001)
STEP = GRID[1] - GRID[0]


def width(fwhm_kms: float) -> float:
    return CENTER * fwhm_kms / C_KMS


def numerical_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return fftconvolve(a, b, mode="same") * STEP


@pytest.mark.parametrize(
    ("profile", "kernels", "lsf"),
    [
        ("gaussian", (), None),
        ("gaussian", (), 187.0),
        ("lorentzian", (), None),
        ("lorentzian", (), 187.0),
        ("exponential", (), None),
        ("exponential", (), 187.0),
        ("gaussian", (("laplace", 900.0, 0.4),), 187.0),
        ("gaussian", (("gaussian", 600.0, 0.3), ("laplace", 900.0, 1.0)), None),
    ],
)
def test_unit_area(profile: str, kernels: tuple, lsf: float | None) -> None:
    values = line_profile(
        XP,
        GRID,
        profile=profile,
        center=CENTER,
        fwhm_kms=300.0,
        kernels=kernels,
        lsf_fwhm_kms=lsf,
    )
    expected = 1.0
    if profile == "lorentzian":
        # The grid truncates the lorentzian wings; compare with its exact mass.
        expected = 2.0 / np.pi * np.arctan(400.0 / (0.5 * width(300.0)))
    assert np.sum(values) * STEP == pytest.approx(expected, abs=2e-4 if lsf else 1e-6)


def test_gaussian_kernel_adds_in_quadrature() -> None:
    convolved = line_profile(
        XP,
        GRID,
        profile="gaussian",
        center=CENTER,
        fwhm_kms=300.0,
        kernels=(("gaussian", 400.0, 1.0),),
    )
    plain = line_profile(XP, GRID, profile="gaussian", center=CENTER, fwhm_kms=500.0)
    np.testing.assert_allclose(convolved, plain, rtol=1e-12)


def test_exponential_lsf_is_numerical_convolution() -> None:
    lsf = 187.0
    sigma = width(lsf) / GAUSSIAN_FWHM_TO_SIGMA
    scale = width(500.0) / LAPLACE_FWHM_TO_SCALE
    offset = GRID - CENTER
    laplace = np.exp(-np.abs(offset) / scale) / (2.0 * scale)
    gaussian = np.exp(-0.5 * (offset / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    expected = numerical_convolution(laplace, gaussian)
    actual = line_profile(
        XP, GRID, profile="exponential", center=CENTER, fwhm_kms=500.0, lsf_fwhm_kms=lsf
    )
    window = np.abs(offset) < 60.0
    np.testing.assert_allclose(actual[window], expected[window], rtol=1e-6, atol=1e-12)


def test_lorentzian_lsf_matches_scipy_voigt() -> None:
    lsf = 187.0
    sigma = width(lsf) / GAUSSIAN_FWHM_TO_SIGMA
    gamma = 0.5 * width(800.0)
    expected = voigt_profile(GRID - CENTER, sigma, gamma)
    actual = line_profile(
        XP, GRID, profile="lorentzian", center=CENTER, fwhm_kms=800.0, lsf_fwhm_kms=lsf
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_draw_axis_broadcasts() -> None:
    fwhm = np.array([[200.0], [400.0]])
    center = np.array([[CENTER], [CENTER + 1.0]])
    stacked = line_profile(
        XP, GRID[::1000], profile="gaussian", center=center, fwhm_kms=fwhm
    )
    single = line_profile(
        XP, GRID[::1000], profile="gaussian", center=CENTER + 1.0, fwhm_kms=400.0
    )
    assert stacked.shape == (2, GRID[::1000].size)
    np.testing.assert_allclose(stacked[1], single)


def test_jax_matches_numpy() -> None:
    pytest.importorskip("jax")
    from noobfriend.core.jaxruntime import enable_x64
    from noobfriend.inference.spectrum._xp import jax_namespace

    enable_x64()
    grid = GRID[::997]
    for profile, kernels in (
        ("lorentzian", ()),
        ("exponential", ()),
        ("gaussian", (("laplace", 900.0, 0.4),)),
    ):
        kwargs = {
            "profile": profile,
            "center": CENTER,
            "fwhm_kms": 300.0,
            "kernels": kernels,
        }
        expected = line_profile(XP, grid, lsf_fwhm_kms=187.0, **kwargs)
        actual = line_profile(jax_namespace(), grid, lsf_fwhm_kms=187.0, **kwargs)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize(
    ("profile", "kinds", "error"),
    [
        ("lorentzian", ("laplace",), NotImplementedError),
        ("gaussian", ("laplace", "laplace"), ValueError),
        ("gaussian", ("voigt",), ValueError),
    ],
)
def test_check_kernels_rejects(profile: str, kinds: tuple, error: type) -> None:
    with pytest.raises(error):
        check_kernels(profile, kinds)
