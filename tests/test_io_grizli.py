"""Tests for the grizli-cutout data-source client in :mod:`noobfriend.core.io`."""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from noobfriend.core.io import grizli


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


class TestPerFilterCache:
    """Per-filter ``/thumb`` fetching and one-file-per-filter caching."""

    def _fake_session(self, thumb_calls: list[str]) -> type:
        """Build a fake session that returns a payload matching the queried filter."""

        class FakeHTTPSession:
            def __init__(self, *, timeout: float | None = None) -> None:
                self.timeout = timeout

            @asynccontextmanager
            async def acquire(self):
                yield self

            async def fetch_content(self, url: str, **_kwargs: object):
                thumb_calls.append(url)
                requested = parse_qs(urlparse(url).query)["filters"][0]
                return _grizli_payload(requested)

        return FakeHTTPSession

    def test_caches_one_file_per_filter_and_reuses_them(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thumb_calls: list[str] = []
        monkeypatch.setattr(grizli, "HTTPSession", self._fake_session(thumb_calls))

        cutout = asyncio.run(
            grizli.load_grizli_cutout(
                10.0,
                20.0,
                size_arcsec=1.0,
                filters=("f090w", "f150w"),
                cache=True,
                cache_dir=tmp_path,
            )
        )

        assert set(cutout.bands) == {"f090w", "f150w"}
        files = sorted(p.name for p in tmp_path.iterdir())
        assert len(files) == 2
        assert any("_f090w_" in name for name in files)
        assert any("_f150w_" in name for name in files)
        assert set(cutout.metadata["cache_paths"]) == {"f090w", "f150w"}
        assert len(thumb_calls) == 2

        # A second identical request must reuse the cached files, fetching nothing.
        reused = asyncio.run(
            grizli.load_grizli_cutout(
                10.0,
                20.0,
                size_arcsec=1.0,
                filters=("f090w", "f150w"),
                cache=True,
                cache_dir=tmp_path,
            )
        )

        assert set(reused.bands) == {"f090w", "f150w"}
        assert len(thumb_calls) == 2

    def test_partial_overlap_only_fetches_missing_filter(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thumb_calls: list[str] = []
        monkeypatch.setattr(grizli, "HTTPSession", self._fake_session(thumb_calls))

        asyncio.run(
            grizli.load_grizli_cutout(
                10.0,
                20.0,
                size_arcsec=1.0,
                filters=("f090w", "f150w"),
                cache=True,
                cache_dir=tmp_path,
            )
        )
        thumb_calls.clear()

        cutout = asyncio.run(
            grizli.load_grizli_cutout(
                10.0,
                20.0,
                size_arcsec=1.0,
                filters=("f150w", "f200w"),
                cache=True,
                cache_dir=tmp_path,
            )
        )

        assert set(cutout.bands) == {"f150w", "f200w"}
        # f150w is reused from the first request; only f200w is fetched.
        assert len(thumb_calls) == 1
        assert "f200w" in thumb_calls[0]
