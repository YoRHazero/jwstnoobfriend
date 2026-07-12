"""Tests for NoobSpectrum data preparation."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum import NoobSpectrum


def test_spectrum_prepares_1d_arrays_and_valid_mask() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    flux = np.array([10.0, np.nan, 30.0, 40.0])
    error = np.array([1.0, 1.0, 0.0, 2.0])
    mask = np.array([False, False, False, True])

    spectrum = NoobSpectrum(
        flux,
        error,
        obs=obs,
        unit="um",
        z=7.0,
        resolving_power=1000.0,
        mask_excluded=mask,
    )

    assert spectrum.unit == "um"
    assert spectrum.z == 7.0
    assert spectrum.resolving_power == 1000.0
    assert spectrum.valid_mask.tolist() == [True, False, False, False]
    assert spectrum.valid_wavelength.tolist() == [1.0]
    assert not spectrum.obs.flags.writeable


def test_spectrum_rejects_invalid_1d_contracts() -> None:
    with pytest.raises(ValueError, match="equal length"):
        NoobSpectrum([1.0], [0.1, 0.1], obs=[1.0, 2.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        NoobSpectrum([1.0, 1.0], [0.1, 0.1], obs=[2.0, 1.0])

    with pytest.raises(ValueError, match="unit"):
        NoobSpectrum([1.0], [0.1], obs=[1.0], unit="A")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="strictly positive"):
        NoobSpectrum([1.0], [0.1], obs=[1.0], z=0.0)

    with pytest.raises(ValueError, match="resolving_power"):
        NoobSpectrum([1.0], [0.1], obs=[1.0], resolving_power=0.0)

    with pytest.raises(ValueError, match="mask_excluded shape"):
        NoobSpectrum([1.0, 1.0], [0.1, 0.1], obs=[1.0, 2.0], mask_excluded=[False])


def test_spectrum_converts_rest_axis_to_observed_frame() -> None:
    spectrum = NoobSpectrum([1.0, 1.0], [0.1, 0.1], rest=[1.0, 2.0], z=2.0)

    assert spectrum.obs.tolist() == [3.0, 6.0]
    assert spectrum.rest.tolist() == [1.0, 2.0]

    with pytest.raises(ValueError, match="z is required"):
        NoobSpectrum([1.0, 1.0], [0.1, 0.1], rest=[1.0, 2.0])

    with pytest.raises(ValueError, match="inconsistent"):
        NoobSpectrum([1.0, 1.0], [0.1, 0.1], obs=[3.0, 6.1], rest=[1.0, 2.0], z=2.0)


def test_spectrum_mask_actions_return_copies() -> None:
    spectrum = NoobSpectrum([1.0, 1.0, 1.0], [0.1, 0.1, 0.1], obs=[1.0, 2.0, 3.0])

    masked = spectrum.exclude([False, True, False])
    replaced = masked.replace_mask([True, False, False])
    cleared = replaced.clear_mask()

    assert spectrum.mask_excluded is None
    assert masked.mask_excluded.tolist() == [False, True, False]
    assert replaced.mask_excluded.tolist() == [True, False, False]
    assert cleared.mask_excluded is None


def test_spectrum_from_2d_collapses_flux_and_error() -> None:
    wavelength = np.linspace(1.0, 2.0, 5)
    flux = np.ones((4, 5))
    error = np.full((4, 5), 0.2)

    spectrum = NoobSpectrum.from_2d(
        flux,
        error,
        obs=wavelength,
        collapse_window=(1, 4),
        dispersion="row",
        error_boost=2.0,
    )

    assert np.allclose(spectrum.flux, 3.0)
    assert np.allclose(spectrum.error, np.sqrt(3 * 0.2**2) * 2.0)
    assert spectrum.source_2d is not None
    assert spectrum.source_2d.flux.shape == (4, 5)
    assert spectrum.source_2d.flux.flags.writeable is False
    assert spectrum.source_2d.error.flags.writeable is False
    assert spectrum.source_2d.spatial.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert spectrum.source_2d.spatial_window == pytest.approx((0.5, 3.5))


def test_spectrum_from_2d_retains_column_dispersion_in_canonical_order() -> None:
    wavelength = np.linspace(1.0, 2.0, 5)
    flux = np.arange(20.0).reshape(5, 4)
    error = np.full((5, 4), 0.2)

    spectrum = NoobSpectrum.from_2d(
        flux,
        error,
        obs=wavelength,
        collapse_window=(1, 4),
        dispersion="column",
    )

    assert spectrum.source_2d is not None
    assert spectrum.source_2d.original_dispersion == "column"
    assert spectrum.source_2d.flux.shape == (4, 5)
    assert np.array_equal(spectrum.source_2d.flux, flux.T)


def test_spectrum_from_2d_validates_geometry_and_noise_contract() -> None:
    wavelength = np.linspace(1.0, 2.0, 5)
    flux = np.ones((4, 5))
    error = np.full((4, 5), 0.2)

    with pytest.raises(ValueError, match="dispersion"):
        NoobSpectrum.from_2d(
            flux, error, obs=wavelength, collapse_window=(1, 3), dispersion="diagonal"
        )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="obs length"):
        NoobSpectrum.from_2d(
            flux, error, obs=np.linspace(1.0, 2.0, 4), collapse_window=(1, 3)
        )

    with pytest.raises(ValueError, match="out of range"):
        NoobSpectrum.from_2d(flux, error, obs=wavelength, collapse_window=(3, 5))

    with pytest.raises(ValueError, match="only used"):
        NoobSpectrum.from_2d(
            flux,
            error,
            obs=wavelength,
            collapse_window=(1, 3),
            noise="continuum",
            error_boost=2.0,
        )


def test_spectrum_calibrate_error_rescales_underquoted_errors() -> None:
    rng = np.random.default_rng(0)
    wavelength = np.linspace(1.0, 2.0, 300)
    flux = 1.0 + rng.normal(0.0, 0.2, wavelength.size)
    error = np.full(wavelength.size, 0.1)

    calibrated = NoobSpectrum(flux, error, obs=wavelength).calibrate_error()

    assert np.median(calibrated.error) == pytest.approx(0.2, rel=0.15)
