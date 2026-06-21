"""Unit tests for the internal FITS layout walker and minimal-FITS rebuild.

All fixtures are synthesised in memory with astropy -- no real FITS files and no
network -- so the tests verify only the offset arithmetic and the byte-for-byte
faithfulness of the reconstruction against astropy's own reading.
"""

from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits

from noobfriend.core.io._layout import data_span, parse_front, wrap_image


def _multi_extension_fits() -> tuple[bytes, dict[str, np.ndarray]]:
    """Build an in-memory FITS with non-square SCI/ERR/DQ extensions."""
    rng = np.random.default_rng(0)
    sci = rng.standard_normal((8, 5)).astype(">f4")
    err = np.abs(rng.standard_normal((8, 5))).astype(">f4")
    dq = rng.integers(0, 7, size=(8, 5)).astype(">i4")
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=sci, name="SCI"),
            fits.ImageHDU(data=err, name="ERR"),
            fits.ImageHDU(data=dq, name="DQ"),
        ]
    )
    buf = BytesIO()
    hdul.writeto(buf)
    return buf.getvalue(), {"SCI": sci, "ERR": err, "DQ": dq}


class TestParseFront:
    """Offsets from the header walk must match astropy's own ``fileinfo``."""

    def test_offsets_match_astropy(self) -> None:
        raw, _ = _multi_extension_fits()
        spans = parse_front(raw)
        with fits.open(BytesIO(raw)) as hdul:
            assert [s.name for s in spans] == ["PRIMARY", "SCI", "ERR", "DQ"]
            for i, span in enumerate(spans):
                info = hdul.fileinfo(i)
                assert span.hdr_loc == info["hdrLoc"]
                assert span.data_loc == info["datLoc"]
                assert span.data_span == info["datSpan"]

    def test_stops_at_incomplete_trailing_header(self) -> None:
        raw, _ = _multi_extension_fits()
        spans = parse_front(raw)
        err_hdr_loc = next(s.hdr_loc for s in spans if s.name == "ERR")
        # Truncate inside the ERR header: only PRIMARY and SCI are fully present.
        partial = parse_front(raw[: err_hdr_loc + 80])
        assert [s.name for s in partial] == ["PRIMARY", "SCI"]

    def test_primary_has_no_data(self) -> None:
        raw, _ = _multi_extension_fits()
        primary = parse_front(raw)[0]
        assert primary.data_span == 0


class TestDataSpan:
    """The padded data size must match astropy for image and table HDUs."""

    @pytest.mark.parametrize("ext", ["SCI", "ERR", "DQ"])
    def test_image_span_matches_astropy(self, ext: str) -> None:
        raw, _ = _multi_extension_fits()
        with fits.open(BytesIO(raw)) as hdul:
            idx = {"SCI": 1, "ERR": 2, "DQ": 3}[ext]
            assert data_span(hdul[idx].header) == hdul.fileinfo(idx)["datSpan"]

    def test_table_heap_counts_via_pcount(self) -> None:
        header = fits.Header(
            {
                "XTENSION": "BINTABLE",
                "BITPIX": 8,
                "NAXIS": 2,
                "NAXIS1": 10,
                "NAXIS2": 3,
                "PCOUNT": 4321,
                "GCOUNT": 1,
            }
        )
        # 10*3 + 4321 = 4351 bytes -> padded to two 2880 blocks.
        assert data_span(header) == 5760


class TestWrapImage:
    """A wrapped extension must decode back to the original array."""

    @pytest.mark.parametrize("ext", ["SCI", "ERR", "DQ"])
    def test_round_trips_through_astropy(self, ext: str) -> None:
        raw, arrays = _multi_extension_fits()
        span = next(s for s in parse_front(raw) if s.name == ext)
        data = raw[span.data_loc : span.data_loc + span.data_span]
        wrapped = wrap_image(span.header, data)
        with fits.open(BytesIO(wrapped)) as hdul:
            assert np.array_equal(hdul[1].data, arrays[ext])

    def test_short_data_is_zero_padded(self) -> None:
        raw, _ = _multi_extension_fits()
        span = next(s for s in parse_front(raw) if s.name == "SCI")
        wrapped = wrap_image(span.header, b"")  # no data at all
        with fits.open(BytesIO(wrapped)) as hdul:
            assert hdul[1].data.shape == (8, 5)
            assert np.all(hdul[1].data == 0)
