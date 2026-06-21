"""Unit tests for the location-transparent byte accessor.

Local accessors exercise a real temp file; remote accessors mock the transport
to assert delegation and spec propagation -- no network is touched.
"""

from pathlib import Path

from noobfriend.core.io.accessor import (
    BytesAccessor,
    LocalAccessor,
    RemoteAccessor,
    open_accessor,
)


class TestBytesAccessor:
    """An accessor over in-memory bytes (the from_file ``raw=`` path)."""

    def test_open_range_tail(self) -> None:
        acc = BytesAccessor(b"0123456789abcdef")
        assert acc.open().read() == b"0123456789abcdef"
        assert acc.read_range(3, 4) == b"3456"
        assert acc.read_tail(5) == b"bcdef"

    def test_tail_longer_than_data(self) -> None:
        assert BytesAccessor(b"012").read_tail(100) == b"012"


class TestOpenAccessorRouting:
    """The single local-versus-remote branch lives in ``open_accessor``."""

    def test_local_string_path(self) -> None:
        assert isinstance(open_accessor("/data/x.fits"), LocalAccessor)

    def test_local_path_object(self) -> None:
        assert isinstance(open_accessor(Path("/data/x.fits")), LocalAccessor)

    def test_remote_spec(self) -> None:
        acc = open_accessor("icrhome08:/data/x.fits")
        assert isinstance(acc, RemoteAccessor)


class TestLocalAccessor:
    """Local reads go straight to the filesystem (lazily, never whole-loaded)."""

    def test_open_reads_whole_file(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        payload = b"0123456789abcdef"
        target.write_bytes(payload)
        with LocalAccessor(target).open() as handle:
            assert handle.read() == payload

    def test_read_range(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"0123456789abcdef")
        assert LocalAccessor(target).read_range(3, 4) == b"3456"

    def test_read_range_past_eof_is_short(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"012")
        assert LocalAccessor(target).read_range(1, 100) == b"12"

    def test_read_tail(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"0123456789abcdef")
        assert LocalAccessor(target).read_tail(5) == b"bcdef"

    def test_read_tail_longer_than_file_returns_all(self, tmp_path: Path) -> None:
        target = tmp_path / "blob.bin"
        target.write_bytes(b"012")
        assert LocalAccessor(target).read_tail(100) == b"012"


class TestRemoteAccessor:
    """Remote reads delegate to the transport, passing the spec verbatim."""

    def test_read_range_delegates(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake(spec, offset, length):
            seen.update(spec=spec, offset=offset, length=length)
            return b"ranged"

        monkeypatch.setattr("noobfriend.core.io.accessor.fetch_range", fake)
        assert RemoteAccessor("h:/p.fits").read_range(10, 5) == b"ranged"
        assert seen == {"spec": "h:/p.fits", "offset": 10, "length": 5}

    def test_read_tail_delegates(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake(spec, length):
            seen.update(spec=spec, length=length)
            return b"tail"

        monkeypatch.setattr("noobfriend.core.io.accessor.fetch_tail", fake)
        assert RemoteAccessor("h:/p.fits").read_tail(7) == b"tail"
        assert seen == {"spec": "h:/p.fits", "length": 7}

    def test_open_wraps_whole_bytes(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "noobfriend.core.io.accessor.fetch_bytes", lambda spec: b"whole-file"
        )
        with RemoteAccessor("h:/p.fits").open() as handle:
            assert handle.read() == b"whole-file"
