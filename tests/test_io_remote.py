"""Unit tests for the remote-read spec parsing and local byte fetching."""

from pathlib import Path

import pytest

from noobfriend.core.io.remote import _parse_spec, fetch_bytes


class TestParseSpec:
    """scp-style local-versus-remote disambiguation for a single file."""

    def test_remote_with_absolute_path(self) -> None:
        assert _parse_spec("icrhome08:/data/x_cal.fits") == (
            "icrhome08",
            "/data/x_cal.fits",
        )

    def test_remote_with_user_and_relative_path(self) -> None:
        assert _parse_spec("zchao@icrhome08:rel/x.fits") == (
            "zchao@icrhome08",
            "rel/x.fits",
        )

    @pytest.mark.parametrize("spec", ["/data/x.fits", "./x.fits", "x.fits"])
    def test_local_string_paths(self, spec: str) -> None:
        assert _parse_spec(spec) == (None, spec)

    def test_path_object_is_always_local(self) -> None:
        assert _parse_spec(Path("/data/x.fits")) == (None, "/data/x.fits")

    def test_colon_after_slash_is_local(self) -> None:
        # A ':' that appears after a '/' belongs to the path, not a host.
        assert _parse_spec("/weird/a:b") == (None, "/weird/a:b")

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host:path"):
            _parse_spec(":/data/x.fits")

    def test_empty_remote_path_raises(self) -> None:
        # Unlike a download destination, a read spec must name a file.
        with pytest.raises(ValueError, match="file path is required"):
            _parse_spec("icrhome08:")


class TestFetchBytesLocal:
    """Local specs are read straight off disk, as str or Path."""

    def test_reads_local_string_path(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        payload = b"\x00\x01noobfriend\xff"
        target.write_bytes(payload)
        assert fetch_bytes(str(target)) == payload

    def test_reads_local_path_object(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        payload = b"payload"
        target.write_bytes(payload)
        assert fetch_bytes(target) == payload

    def test_missing_local_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fetch_bytes(tmp_path / "does-not-exist.fits")
