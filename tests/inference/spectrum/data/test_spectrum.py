"""Tests for NoobSpectrum construction and transformations."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.spectrum.data import NoobSpectrum
from noobfriend.inference.spectrum.noise import AR1, Autocorrelation, Noise

OBS = np.linspace(6500.0, 6600.0, 101)


def spectrum(**kwargs: object) -> NoobSpectrum:
    flux = np.ones(OBS.size)
    error = np.full(OBS.size, 0.1)
    options = {"unit": "angstrom", "resolving_power": 1600.0, "z": 0.0} | kwargs
    return NoobSpectrum(OBS, flux, error, **options)  # type: ignore[arg-type]


class TestConstruction:
    """Required arguments and validation."""

    def test_unit_and_resolving_power_are_required(self) -> None:
        with pytest.raises(TypeError):
            NoobSpectrum(OBS, np.ones(OBS.size), np.ones(OBS.size), unit="angstrom")  # type: ignore[call-arg]

    def test_single_frame_id(self) -> None:
        assert spectrum().frame_ids == ("frame_0",)

    def test_invalid_pixels_stay_in_place(self) -> None:
        flux = np.ones(OBS.size)
        flux[3] = np.nan
        frame = NoobSpectrum(
            OBS, flux, np.ones(OBS.size), unit="nm", resolving_power=None
        ).frames["frame_0"]
        assert frame.size == OBS.size
        assert not frame.usable[3]

    def test_rejects_decreasing_obs(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            NoobSpectrum(
                OBS[::-1],
                np.ones(OBS.size),
                np.ones(OBS.size),
                unit="nm",
                resolving_power=None,
            )

    def test_rejects_resolving_power_function(self) -> None:
        with pytest.raises(TypeError, match="functions"):
            spectrum(resolving_power=lambda wavelength: 1600.0)

    def test_resolving_power_table_interpolates(self) -> None:
        table = (np.array([6400.0, 6700.0]), np.array([1000.0, 1600.0]))
        frame = spectrum(resolving_power=table).frames["frame_0"]
        assert frame.resolving_power_at(6550.0) == pytest.approx(1300.0)
        with pytest.raises(ValueError, match="outside"):
            frame.resolving_power_at(6800.0)


class TestJoin:
    """Frames side by side."""

    def test_mapping_keys_become_ids(self) -> None:
        joined = NoobSpectrum.join(
            {"v1": spectrum(), "v2": spectrum(resolving_power=1200.0)}
        )
        assert joined.frame_ids == ("v1", "v2")
        assert joined.frames["v2"].resolving_power == 1200.0

    def test_sequence_gets_default_ids(self) -> None:
        assert NoobSpectrum.join([spectrum(), spectrum()]).frame_ids == (
            "frame_0",
            "frame_1",
        )

    def test_rejects_multi_frame_input(self) -> None:
        joined = NoobSpectrum.join([spectrum(), spectrum()])
        with pytest.raises(ValueError, match="single-frame"):
            NoobSpectrum.join([joined])

    def test_rejects_redshift_mismatch(self) -> None:
        with pytest.raises(ValueError, match="z="):
            NoobSpectrum.join([spectrum(z=1.0), spectrum(z=1.1)])

    def test_rejects_dotted_ids(self) -> None:
        with pytest.raises(ValueError, match="'.'"):
            NoobSpectrum.join({"v.1": spectrum()})

    def test_select_and_drop(self) -> None:
        joined = NoobSpectrum.join({"a": spectrum(), "b": spectrum(), "c": spectrum()})
        assert joined.select("c", "a").frame_ids == ("c", "a")
        assert joined.drop("b").frame_ids == ("a", "c")
        with pytest.raises(KeyError):
            joined.select("x")


class TestTransformations:
    """Exclusion masks and noise attachment."""

    def test_exclude_rest_needs_z_and_masks_every_frame(self) -> None:
        joined = NoobSpectrum.join([spectrum(z=0.01), spectrum(z=0.01)])
        excluded = joined.exclude(rest=(6500.0 / 1.01, 6510.0 / 1.01))
        for frame in excluded.frames.values():
            assert frame.mask.sum() == 11
        with pytest.raises(ValueError, match="z"):
            spectrum(z=None).exclude(rest=(6500.0, 6510.0))

    def test_with_noise_mapping_replaces_named_frames(self) -> None:
        joined = NoobSpectrum.join({"a": spectrum(), "b": spectrum()})
        noisy = joined.with_noise(Noise(scale=1.2)).with_noise(
            {"b": Noise(correlation=AR1(0.3))}
        )
        assert noisy.frames["a"].noise == Noise(scale=1.2)
        assert noisy.frames["b"].noise == Noise(correlation=AR1(0.3))
        with pytest.raises(KeyError):
            joined.with_noise({"z": Noise()})

    def test_noise_from_continuum_returns_fixed_estimates(self) -> None:
        rng = np.random.default_rng(5)
        obs = np.linspace(1.0, 2.0, 4000)
        error = np.full(obs.size, 0.1)
        flux = 1.0 + 0.2 * error * rng.normal(size=obs.size)
        spec = NoobSpectrum(obs, flux, error, unit="um", resolving_power=None)
        estimate = spec.noise_from_continuum(lags=1)["frame_0"]
        assert estimate.scale == pytest.approx(0.2, rel=0.05)
        assert isinstance(estimate.correlation, Autocorrelation)

    def test_noise_from_background_needs_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D source"):
            spectrum().noise_from_background()


class TestFrom2D:
    """Collapse with the 2-D source kept."""

    def test_column_dispersion_is_reoriented(self) -> None:
        rng = np.random.default_rng(6)
        flux_2d = rng.normal(size=(OBS.size, 12))
        spec = NoobSpectrum.from_2d(
            flux_2d,
            np.ones_like(flux_2d),
            obs=OBS,
            collapse_window=(4, 8),
            dispersion_axis=0,
            unit="angstrom",
            resolving_power=None,
        )
        frame = spec.frames["frame_0"]
        np.testing.assert_allclose(frame.flux, flux_2d[:, 4:8].sum(axis=1))
        np.testing.assert_allclose(frame.error, np.full(OBS.size, 2.0))
        assert frame.source_2d is not None
        assert frame.source_2d.flux.shape == (12, OBS.size)
        np.testing.assert_array_equal(frame.source_2d.spatial, np.arange(12.0))

    def test_rejects_window_outside_rows(self) -> None:
        with pytest.raises(ValueError, match="collapse_window"):
            NoobSpectrum.from_2d(
                np.ones((5, OBS.size)),
                np.ones((5, OBS.size)),
                obs=OBS,
                collapse_window=(3, 7),
                unit="angstrom",
                resolving_power=None,
            )
