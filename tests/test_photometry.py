"""Tests for multi-band aperture photometry and SED plotting."""

from io import BytesIO
from typing import Any, Callable

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from noobfriend.extraction.photometry import ApertureSED
from noobfriend.extraction.photometry import _grizli


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


def _grizli_hdu(
    filter_name: str,
    data: np.ndarray,
    *,
    primary: bool = False,
    extra_header: dict[str, object] | None = None,
) -> fits.PrimaryHDU | fits.ImageHDU:
    """Build one grizli-like image HDU with a simple celestial WCS."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [2.0, 2.0]  # Pixel (1, 1) maps to crval in astropy's 0-based API.
    wcs.wcs.cdelt = [-0.001, 0.001]
    header = wcs.to_header()
    header["FILTER"] = filter_name.upper()
    header["EXTNAME"] = filter_name.upper()
    for key, value in (extra_header or {}).items():
        header[key] = value
    if primary:
        return fits.PrimaryHDU(data=np.asarray(data, dtype=float), header=header)
    return fits.ImageHDU(
        data=np.asarray(data, dtype=float), header=header, name=filter_name.upper()
    )


def _grizli_payload(
    filter_name: str = "f090w",
    *,
    data: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    extra_header: dict[str, object] | None = None,
) -> bytes:
    """Build an in-memory grizli ``fits_weight`` payload."""
    if data is None:
        data = np.zeros((3, 3), dtype=float)
        data[1, 1] = 5.0
    if weight is None:
        weight = np.ones_like(data, dtype=float)
    hdul = fits.HDUList(
        [
            _grizli_hdu(filter_name, data, primary=True, extra_header=extra_header),
            _grizli_hdu(filter_name, weight, extra_header=extra_header),
        ]
    )
    buffer = BytesIO()
    hdul.writeto(buffer)
    return buffer.getvalue()


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


class TestGrizliCutout:
    """grizli-cutout filter selection and FITS parsing."""

    def test_sed_default_selects_all_jwst_and_only_hst_blue(self) -> None:
        overlap = {
            "f090w": ["x", 1],
            "f444w": ["x", 2],
            "f770w": ["x", 1],
            "f435w": ["x", 1],
            "f606w": ["x", 1],
            "f814w": ["x", 1],
            "g141": ["x", 1],
            "f115w": ["x", 0],
        }

        selected = _grizli.select_overlap_filters(overlap, filter_set="sed-default")

        assert selected == ("f090w", "f444w", "f770w", "f435w", "f606w")

    def test_read_grizli_cutout_bands_pairs_science_and_weight(self) -> None:
        data = np.full((3, 3), 2.0)
        weight = np.ones((3, 3)) * 4.0
        weight[0, 0] = 0.0

        bands = _grizli.read_grizli_cutout_bands(
            _grizli_payload(
                "f090w",
                data=data,
                weight=weight,
                extra_header={"BUNIT": "10.0*nanoJansky"},
            )
        )

        band = bands["f090w"]
        assert band["wavelength"] == pytest.approx(0.90)
        assert band["flux_scale_mjy"] == pytest.approx(1e-5)
        assert band["flux_unit"] == "10.0*nanoJansky"
        assert np.isnan(band["data"][0, 0])
        assert np.isnan(band["error"][0, 0])
        np.testing.assert_allclose(band["error"][1, 1], 0.5)
        assert band["wcs"].available_frames == ("detector", "world")

    def test_grizli_mjy_scale_falls_back_to_photflam_photplam(self) -> None:
        bands = _grizli.read_grizli_cutout_bands(
            _grizli_payload(
                "f606w",
                extra_header={
                    "PHOTFLAM": 7.7424564e-20,
                    "PHOTPLAM": 5919.8506,
                },
            )
        )

        assert bands["f606w"]["flux_scale_mjy"] == pytest.approx(9.05064558e-5)
        assert bands["f606w"]["flux_unit"] == "PHOTFLAM/PHOTPLAM"

    def test_from_grizli_cutout_defaults_to_memory_and_source_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _grizli_payload("f090w")
        calls: list[str] = []

        overlap_payload = (
            '{"f090w": ["x", 1], "f435w": ["x", 1], '
            '"f814w": ["x", 1], "g141": ["x", 1]}'
        ).encode()

        class FakeHTTPSession:
            """Fake noobfriend HTTP session for grizli tests."""

            def __init__(self, *, timeout: float | None = None) -> None:
                self.timeout = timeout

            async def fetch_json(self, url: str):
                calls.append(f"json:{url}")
                return {
                    "f090w": ["x", 1],
                    "f435w": ["x", 1],
                    "f814w": ["x", 1],
                    "g141": ["x", 1],
                }

            async def fetch_content(self, url: str):
                calls.append(f"content:{url}")
                if "/overlap?" in url:
                    return overlap_payload
                return payload

        monkeypatch.setattr(_grizli, "HTTPSession", FakeHTTPSession)

        sed = ApertureSED.from_grizli_cutout(
            10.0,
            20.0,
            size=1.0,
            filters="sed-default",
        )
        result = sed.measure(grow_kwargs=_grow_all_label_kwargs())

        assert sed.source_metadata["cache_path"] is None
        assert sed.source_metadata["selected_filters"] == ("f090w", "f435w")
        assert sed.source_metadata["missing_filters"] == ("f435w",)
        assert result.source_metadata["source"] == "grizli-cutout"
        assert result.reference_band == "f090w"
        assert result.measurements[0].band == "f090w"
        assert result.measurements[0].flux == pytest.approx(5.0)
        thumb_calls = [call for call in calls if "/thumb?" in call]
        assert len(thumb_calls) == 1
