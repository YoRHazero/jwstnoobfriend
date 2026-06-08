"""Tests for the grizli-cutout data-source client in :mod:`noobfriend.core.io`."""

from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from noobfriend.core.io import grizli
from noobfriend.extraction.photometry import ApertureSED


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


class TestFilterSelection:
    """Overlap-driven filter selection."""

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

        selected = grizli.select_overlap_filters(overlap, filter_set="sed-default")

        assert selected == ("f090w", "f444w", "f770w", "f435w", "f606w")


class TestReadCutoutBands:
    """Parsing grizli ``fits_weight`` payloads into band specs."""

    def test_read_grizli_cutout_bands_pairs_science_and_weight(self) -> None:
        data = np.full((3, 3), 2.0)
        weight = np.ones((3, 3)) * 4.0
        weight[0, 0] = 0.0

        bands = grizli.read_grizli_cutout_bands(
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
        bands = grizli.read_grizli_cutout_bands(
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


class TestFromGrizliCutout:
    """The ApertureSED convenience shim over the core grizli client."""

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

        monkeypatch.setattr(grizli, "HTTPSession", FakeHTTPSession)

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
