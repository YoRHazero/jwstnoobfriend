"""Unit tests for the byte-frugal FITS readers driven by a ByteAccessor.

Fixtures are synthesised in memory with astropy + stdatamodels (a real
ASDF-in-FITS layout) -- no example data, no network. Reads go through both a
:class:`LocalAccessor` (a temp file) and a :class:`BytesAccessor`.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from stdatamodels import asdf_in_fits

from noobfriend.core.io.accessor import BytesAccessor, LocalAccessor
from noobfriend.core.io.fits import (
    read_data,
    read_dq,
    read_err,
    read_gwcs,
    read_layout,
    read_meta,
    read_meta_and_gwcs,
)


def _write_asdf_fits(
    path: Path,
    *,
    exts: tuple[str, ...] = ("SCI", "ERR", "DQ"),
    wcs: object | None = None,
    pupil: str = "CLEAR",
    filt: str = "F200W",
) -> dict[str, np.ndarray]:
    """Write a synthetic ASDF-in-FITS file; return the extension arrays by name."""
    rng = np.random.default_rng(1)
    arrays: dict[str, np.ndarray] = {}
    hdus = [fits.PrimaryHDU()]
    for name in exts:
        dtype = ">i4" if name == "DQ" else ">f4"
        arr = (rng.standard_normal((8, 5)) * 10).astype(dtype)
        arrays[name] = arr
        hdus.append(fits.ImageHDU(data=arr, name=name))
    meta: dict[str, object] = {"instrument": {"pupil": pupil, "filter": filt}}
    if wcs is not None:
        meta["wcs"] = wcs
    buf = BytesIO()
    asdf_in_fits.write(buf, {"meta": meta}, hdulist=fits.HDUList(hdus))
    path.write_bytes(buf.getvalue())
    return arrays


@pytest.fixture
def cal_file(tmp_path: Path):
    """Build a synthetic SCI/ERR/DQ ASDF-in-FITS file and its extension arrays."""
    path = tmp_path / "synthetic_cal.fits"
    arrays = _write_asdf_fits(path)
    return path, arrays


class TestReadLayout:
    """The layout resolves SCI shape/offset from the leading header."""

    def test_sci_shape_and_span(self, cal_file) -> None:
        path, _ = cal_file
        layout = read_layout(LocalAccessor(path))
        assert layout.shape == (8, 5)
        assert layout.sci is not None and layout.sci.name == "SCI"

    def test_no_sci_yields_none(self, tmp_path: Path) -> None:
        path = tmp_path / "no_sci.fits"
        _write_asdf_fits(path, exts=("DQ",))
        layout = read_layout(LocalAccessor(path))
        assert layout.sci is None and layout.shape is None


class TestReadExtensions:
    """Each pixel extension decodes to the original array via ranged reads."""

    def test_data_err_dq(self, cal_file) -> None:
        path, arrays = cal_file
        acc = LocalAccessor(path)
        layout = read_layout(acc)
        assert np.array_equal(read_data(acc, layout), arrays["SCI"])
        assert np.array_equal(read_err(acc, layout), arrays["ERR"])
        assert np.array_equal(read_dq(acc, layout), arrays["DQ"])

    def test_via_bytes_accessor(self, cal_file) -> None:
        path, arrays = cal_file
        acc = BytesAccessor(path.read_bytes())
        layout = read_layout(acc)
        assert np.array_equal(read_data(acc, layout), arrays["SCI"])
        assert np.array_equal(read_dq(acc, layout), arrays["DQ"])

    def test_missing_extension_raises_keyerror(self, tmp_path: Path) -> None:
        path = tmp_path / "sci_only.fits"
        _write_asdf_fits(path, exts=("SCI",))
        acc = LocalAccessor(path)
        layout = read_layout(acc)
        with pytest.raises(KeyError):
            read_err(acc, layout)
        with pytest.raises(KeyError):
            read_dq(acc, layout)

    def test_no_sci_data_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no_sci.fits"
        _write_asdf_fits(path, exts=("DQ",))
        acc = LocalAccessor(path)
        with pytest.raises(KeyError):
            read_data(acc, read_layout(acc))


class _CountingAccessor:
    """Wrap a BytesAccessor, tallying ranged reads (to observe re-walking)."""

    def __init__(self, data: bytes) -> None:
        self._inner = BytesAccessor(data)
        self.range_calls = 0

    def open(self):
        return self._inner.open()

    def read_range(self, offset: int, length: int) -> bytes:
        self.range_calls += 1
        return self._inner.read_range(offset, length)

    def read_tail(self, length: int) -> bytes:
        return self._inner.read_tail(length)


class TestExtensionMemo:
    """ERR / DQ offsets are discovered once per layout, not on every access."""

    def test_err_walk_happens_once(self, cal_file) -> None:
        path, _ = cal_file
        acc = _CountingAccessor(path.read_bytes())
        layout = read_layout(acc)
        acc.range_calls = 0
        read_err(acc, layout)  # walk to ERR header (1) + read its data (1)
        first = acc.range_calls
        acc.range_calls = 0
        read_err(acc, layout)  # memoised: only the data read remains
        assert first == 2
        assert acc.range_calls == 1

    def test_absent_extension_not_researched(self, tmp_path: Path) -> None:
        path = tmp_path / "sci_only.fits"
        _write_asdf_fits(path, exts=("SCI",))
        acc = _CountingAccessor(path.read_bytes())
        layout = read_layout(acc)
        with pytest.raises(KeyError):
            read_err(acc, layout)
        acc.range_calls = 0
        with pytest.raises(KeyError):
            read_err(acc, layout)  # cached absence: no further header walk
        assert acc.range_calls == 0


class TestReadMeta:
    """Metadata and GWCS come from the tail without reading pixel extensions."""

    def test_meta_scalars(self, cal_file) -> None:
        path, _ = cal_file
        meta = read_meta(LocalAccessor(path))
        assert meta.instrument.pupil == "CLEAR"
        assert meta.instrument.filter == "F200W"

    def test_no_wcs_raises_keyerror(self, cal_file) -> None:
        path, _ = cal_file
        with pytest.raises(KeyError):
            read_gwcs(LocalAccessor(path))

    def test_meta_and_gwcs_returns_none_wcs(self, cal_file) -> None:
        path, _ = cal_file
        meta, wcs = read_meta_and_gwcs(LocalAccessor(path))
        assert meta.instrument.pupil == "CLEAR"
        assert wcs is None

    def test_wcs_present_is_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "with_wcs.fits"
        _write_asdf_fits(path, wcs="PLACEHOLDER-WCS")
        assert read_gwcs(LocalAccessor(path)) == "PLACEHOLDER-WCS"
        _, wcs = read_meta_and_gwcs(LocalAccessor(path))
        assert wcs == "PLACEHOLDER-WCS"
