"""Tests for the grizli-cutout data-source client in :mod:`noobfriend.core.io`."""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from noobfriend.core.io import grizli_cutout as grizli


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("f090w-clear", True),  # JWST imaging through the CLEAR pupil
            ("f435w", True),  # bare HST filter
            ("f1000w", True),  # bare MIRI filter
            ("f150w2-f162m", False),  # NIRCam dual-filter combo (pupil F162M)
            ("f150w-gr150c", False),  # NIRISS WFSS grism
            ("f356w-grismr", False),  # NIRCam WFSS grism
            ("g102", False),  # HST grism
            ("g141", False),  # HST grism
        ],
    )
    def test_is_imaging_filter(self, name: str, expected: bool) -> None:
        assert grizli._is_imaging_filter(name) is expected

    def test_sed_default_drops_dual_filter_and_grism_names(self) -> None:
        # The live service names JWST bands ``FILTER-PUPIL``; only CLEAR-pupil
        # (and bare HST/MIRI) names are clean single bandpasses.
        overlap = {
            "filters": [
                "f090w-clear",
                "f150w2-f162m",
                "f150w-gr150c",
                "f356w-grismr",
                "g141",
                "f1000w",
            ],
            "f090w-clear": [100.0, 5],
            "f150w2-f162m": [100.0, 3],
            "f150w-gr150c": [100.0, 6],
            "f356w-grismr": [100.0, 8],
            "g141": [100.0, 4],
            "f1000w": [100.0, 5],
        }

        selected = grizli.select_overlap_filters(overlap, filter_set="sed-default")

        assert selected == ("f090w-clear", "f1000w")


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


class TestNoCutoutSentinel:
    """A 200 non-FITS 'No cutout found' body is an unavailable band, not a crash.

    The grizli service returns each band at its own URL and answers a filter with
    no covering subtile with a plain-text ``No cutout found ...`` body (HTTP 200),
    not FITS. That must be treated as a missing band rather than fed to
    ``fits.open`` (which raises ``OSError: No SIMPLE card found``).
    """

    _SENTINEL = b"No cutout found for ra=10.0, dec=20.0, filters=['f162m-clear']"

    def _session_with_sentinel(self, bad: set[str]) -> type:
        """Fake session returning the sentinel for ``bad`` filters, FITS otherwise."""
        sentinel = self._SENTINEL

        class FakeHTTPSession:
            def __init__(self, *, timeout: float | None = None) -> None:
                self.timeout = timeout

            @asynccontextmanager
            async def acquire(self):
                yield self

            async def fetch_content(self, url: str, **_kwargs: object):
                requested = parse_qs(urlparse(url).query)["filters"][0]
                return sentinel if requested in bad else _grizli_payload(requested)

        return FakeHTTPSession

    def test_read_rejects_non_fits_payload(self) -> None:
        with pytest.raises(ValueError, match="not a FITS file"):
            grizli.read_grizli_cutout_bands(self._SENTINEL)

    def test_load_drops_no_cutout_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            grizli, "HTTPSession", self._session_with_sentinel({"f150w2-f162m"})
        )

        cutout = asyncio.run(
            grizli.load_grizli_cutout(
                10.0, 20.0, size_arcsec=1.0, filters=("f090w", "f150w2-f162m")
            )
        )

        assert set(cutout.bands) == {"f090w"}
        assert cutout.metadata["missing_filters"] == ("f150w2-f162m",)

    def test_load_strict_raises_on_no_cutout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            grizli, "HTTPSession", self._session_with_sentinel({"f150w2-f162m"})
        )

        with pytest.raises(ValueError, match="no FITS cutout"):
            asyncio.run(
                grizli.load_grizli_cutout(
                    10.0,
                    20.0,
                    size_arcsec=1.0,
                    filters=("f090w", "f150w2-f162m"),
                    allow_missing=False,
                )
            )

    def test_no_cutout_band_is_not_cached(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            grizli, "HTTPSession", self._session_with_sentinel({"f150w2-f162m"})
        )

        asyncio.run(
            grizli.load_grizli_cutout(
                10.0,
                20.0,
                size_arcsec=1.0,
                filters=("f090w", "f150w2-f162m"),
                cache=True,
                cache_dir=tmp_path,
            )
        )

        files = sorted(p.name for p in tmp_path.iterdir())
        # Only the valid band is cached; the sentinel is never written to disk.
        assert len(files) == 1
        assert any("_f090w_" in name for name in files)
