"""Tests for multi-band aperture photometry and SED plotting."""

from typing import Any, Callable

import numpy as np
import pytest

from noobfriend.extraction.photometry import ApertureSED


class LinearWCS:
    """A small affine WCS stand-in supporting detector/world transforms."""

    available_frames = ["detector", "world"]

    def __init__(
        self,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        ra0: float = 0.0,
        dec0: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.ra0 = ra0
        self.dec0 = dec0
        self.scale = scale

    def get_transform(
        self, from_frame: str, to_frame: str
    ) -> Callable[[Any, Any], tuple[Any, Any]]:
        """Return detector/world transforms for the requested direction."""
        if (from_frame, to_frame) == ("world", "detector"):
            return self._world_to_detector
        if (from_frame, to_frame) == ("detector", "world"):
            return self._detector_to_world
        raise ValueError(f"Unsupported transform: {from_frame!r} -> {to_frame!r}.")

    def _world_to_detector(self, ra: Any, dec: Any) -> tuple[Any, Any]:
        return (
            (ra - self.ra0) / self.scale + self.x0,
            (dec - self.dec0) / self.scale + self.y0,
        )

    def _detector_to_world(self, x: Any, y: Any) -> tuple[Any, Any]:
        return (
            (x - self.x0) * self.scale + self.ra0,
            (y - self.y0) * self.scale + self.dec0,
        )


def _label(shape: tuple[int, int], pixels: set[tuple[int, int]]) -> np.ndarray:
    """Build a segmentation map with label 1 at ``(row, col)`` pixels."""
    labels = np.zeros(shape, dtype=np.int32)
    for row, col in pixels:
        labels[row, col] = 1
    return labels


def _band(
    data: np.ndarray,
    *,
    wcs: LinearWCS | None = None,
    wavelength: float = 1.0,
    wavelength_error: tuple[float, ...] | None = None,
    flux_scale_mjy: float | None = None,
    flux_unit: str | None = None,
    label_pixels: set[tuple[int, int]] | None = None,
    error: np.ndarray | None = None,
) -> dict[str, object]:
    """Build one public band spec for tests."""
    spec: dict[str, object] = {
        "data": np.asarray(data, dtype=float),
        "wcs": wcs or LinearWCS(),
        "wavelength": wavelength,
    }
    if wavelength_error is not None:
        spec["wavelength_error"] = wavelength_error
    if flux_scale_mjy is not None:
        spec["flux_scale_mjy"] = flux_scale_mjy
    if flux_unit is not None:
        spec["flux_unit"] = flux_unit
    if label_pixels is not None:
        spec["label_map"] = _label(np.asarray(data).shape, label_pixels)
    if error is not None:
        spec["error"] = np.asarray(error, dtype=float)
    return spec


def _grow_all_label_kwargs() -> dict[str, object]:
    """Disable stop criteria so noobase fills the allowed label region."""
    return {
        "snr_threshold": None,
        "gradient_ratio_threshold": 1e99,
    }


class TestApertureSEDInputs:
    """Public input normalization and validation."""

    def test_wavelength_error_must_be_tuple(self) -> None:
        with pytest.raises(ValueError, match="wavelength_error must be a tuple"):
            ApertureSED(
                bands={
                    "F090W": {
                        "data": np.ones((3, 3)),
                        "wcs": LinearWCS(),
                        "wavelength": 0.9,
                        "wavelength_error": 0.05,
                    }
                }
            )

    def test_wavelength_error_len_one_is_symmetric(self) -> None:
        data = np.zeros((3, 3))
        data[1, 1] = 10.0
        sed = ApertureSED(
            bands={
                "F090W": _band(
                    data,
                    wavelength=0.9,
                    wavelength_error=(0.05,),
                    label_pixels={(1, 1)},
                )
            }
        )

        result = sed.measure(
            seed_xy_by_band={"F090W": (1.0, 1.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        )

        assert result.measurements[0].wavelength_error == (0.05, 0.05)

    def test_wavelength_error_other_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length 1 or 2"):
            ApertureSED(
                bands={
                    "F090W": _band(
                        np.ones((3, 3)),
                        wavelength=0.9,
                        wavelength_error=(0.01, 0.02, 0.03),
                    )
                }
            )


class TestApertureSEDMeasure:
    """Union-aperture measurement behaviour."""

    def test_per_band_apertures_union_then_flag_bad_pixels(self) -> None:
        a = np.ones((5, 5))
        b = np.ones((5, 5))
        b[2, 1] = 0.0  # In the union aperture via band A, invalid for band B.
        err = np.ones((5, 5))

        sed = ApertureSED(
            bands={
                "F090W": _band(
                    a,
                    wavelength=0.9,
                    flux_scale_mjy=0.01,
                    flux_unit="test-unit",
                    label_pixels={(2, 1), (2, 2)},
                    error=err,
                ),
                "F115W": _band(
                    b,
                    wavelength=1.15,
                    label_pixels={(2, 2), (2, 3)},
                    error=err,
                ),
            },
            reference="F090W",
            flag_bad_fraction=0.10,
        )

        result = sed.measure(
            seed_xy_by_band={"F090W": (2.0, 2.0), "F115W": (2.0, 2.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        )

        assert result.reference_band == "F090W"
        assert int(result.union_mask.sum()) == 3
        by_band = {m.band: m for m in result.measurements}
        assert by_band["F090W"].flux == pytest.approx(3.0)
        assert by_band["F090W"].error == pytest.approx(np.sqrt(3.0))
        assert by_band["F090W"].flux_mjy == pytest.approx(0.03)
        assert by_band["F090W"].error_mjy == pytest.approx(0.01 * np.sqrt(3.0))
        assert by_band["F090W"].flux_unit == "test-unit"
        assert by_band["F090W"].bad_fraction == pytest.approx(0.0)
        assert not by_band["F090W"].flagged
        assert by_band["F115W"].flux == pytest.approx(2.0)
        assert by_band["F115W"].n_valid_pixels == 2
        assert by_band["F115W"].bad_fraction == pytest.approx(1 / 3)
        assert by_band["F115W"].flagged
        assert by_band["F115W"].flag_reason == "bad_pixels"

    def test_reference_finest_selects_smallest_pixel_scale_band(self) -> None:
        fine = np.zeros((5, 5))
        fine[2, 2] = 10.0
        coarse = np.zeros((5, 5))
        coarse[1, 1] = 10.0
        sed = ApertureSED(
            bands={
                "coarse": _band(
                    coarse,
                    wcs=LinearWCS(scale=2.0),
                    wavelength=2.0,
                    label_pixels={(1, 1)},
                ),
                "fine": _band(
                    fine,
                    wcs=LinearWCS(scale=1.0),
                    wavelength=1.0,
                    label_pixels={(2, 2)},
                ),
            },
            reference="finest",
        )

        result = sed.measure(
            seed_world=(2.0, 2.0),
            grow_kwargs=_grow_all_label_kwargs(),
        )

        assert result.reference_band == "fine"
        assert result.union_mask.shape == (5, 5)

    def test_seed_keys_must_match_bands(self) -> None:
        sed = ApertureSED(
            bands={"F090W": _band(np.ones((3, 3)), label_pixels={(1, 1)})}
        )

        with pytest.raises(ValueError, match="keys must match band names"):
            sed.measure(
                seed_xy_by_band={"F115W": (1.0, 1.0)},
                grow_kwargs=_grow_all_label_kwargs(),
            )


class TestApertureSEDResult:
    """Result export and plot helpers."""

    def test_to_table_and_plot_smoke(self) -> None:
        data = np.zeros((3, 3))
        data[1, 1] = 5.0
        sed = ApertureSED(
            bands={
                "F090W": _band(
                    data,
                    wavelength=0.9,
                    wavelength_error=(0.05, 0.06),
                    flux_scale_mjy=0.01,
                    label_pixels={(1, 1)},
                )
            }
        )
        result = sed.measure(
            seed_xy_by_band={"F090W": (1.0, 1.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        )

        table = result.to_table()
        assert table["band"][0] == "F090W"
        assert table["flux"][0] == pytest.approx(5.0)
        assert "flux_mjy" in table.colnames
        assert "error_mjy" in table.colnames

        fig = result.plot(display_plot=False)
        assert fig.title.text == "Aperture SED"
        assert fig.yaxis[0].axis_label == "Flux (mJy)"
        assert {r.glyph.__class__.__name__ for r in fig.renderers} == {"Scatter"}
        assert fig.renderers[0].glyph.y == "flux_mjy"
