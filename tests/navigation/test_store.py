"""Unit tests for the LRU byte store: caching, ranged/tail reads, fallback."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from noobfriend.core.io.remote import (
    RemoteReadError,
    _reset_range_capability,
    range_capable,
)
from noobfriend.navigation._store import LruByteStore


@pytest.fixture(autouse=True)
def _clean_capability():
    """Keep the process-global host capability registry isolated per test."""
    _reset_range_capability()
    yield
    _reset_range_capability()


class TestLocalReads:
    """Whole / ranged / tail reads against a real local file, cached by key."""

    def test_fetch_whole(self, tmp_path: Path) -> None:
        target = tmp_path / "x.bin"
        target.write_bytes(b"0123456789")
        store = LruByteStore()
        assert store.fetch("id", str(target)) == b"0123456789"

    def test_fetch_range_and_tail(self, tmp_path: Path) -> None:
        target = tmp_path / "x.bin"
        target.write_bytes(b"0123456789")
        store = LruByteStore()
        assert store.fetch_range("id", str(target), 2, 3) == b"234"
        assert store.fetch_tail("id", str(target), 4) == b"6789"

    def test_range_is_cached(self, tmp_path: Path) -> None:
        target = tmp_path / "x.bin"
        target.write_bytes(b"0123456789")
        store = LruByteStore()
        store.fetch_range("id", str(target), 2, 3)
        # The file changing underneath must not be re-read: the range is cached.
        target.write_bytes(b"XXXXXXXXXX")
        assert store.fetch_range("id", str(target), 2, 3) == b"234"


class TestCapabilityFallback:
    """A remote host whose ranged read fails falls back to whole-file-and-slice."""

    def test_falls_back_and_marks_host(self, monkeypatch) -> None:
        whole = b"abcdefghij"

        def no_range(*_a, **_k):
            raise RemoteReadError("dd: unsupported")

        monkeypatch.setattr(
            "noobfriend.navigation._store.open_accessor",
            lambda loc: SimpleNamespace(read_range=no_range, read_tail=no_range),
        )
        monkeypatch.setattr(
            "noobfriend.navigation._store.fetch_bytes", lambda loc: whole
        )

        store = LruByteStore()
        assert store.fetch_range("id", "h:/p.fits", 2, 3) == b"cde"
        assert range_capable("h") is False  # remembered, so no retry next time

    def test_known_incapable_host_skips_ranged_attempt(self, monkeypatch) -> None:
        whole = b"abcdefghij"
        called = {"range": 0}

        def no_range(*_a, **_k):
            called["range"] += 1
            raise RemoteReadError("should not be called")

        monkeypatch.setattr(
            "noobfriend.navigation._store.open_accessor",
            lambda loc: SimpleNamespace(read_range=no_range, read_tail=no_range),
        )
        monkeypatch.setattr(
            "noobfriend.navigation._store.fetch_bytes", lambda loc: whole
        )
        from noobfriend.core.io.remote import set_range_capable

        set_range_capable("h", False)

        store = LruByteStore()
        assert store.fetch_tail("id", "h:/p.fits", 4) == b"ghij"
        assert called["range"] == 0  # ranged read never attempted
