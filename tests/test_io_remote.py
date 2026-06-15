"""Unit tests for the remote-read spec parsing and local byte fetching."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from noobfriend.core.io.remote import (
    RemoteReadError,
    RemoteWriteError,
    _parse_spec,
    fetch_bytes,
    list_remote_dir,
    write_bytes,
)


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


class TestListRemoteDir:
    """Remote one-level file listing over SSH (the subprocess is mocked)."""

    def test_parses_find_output_into_names(self, monkeypatch) -> None:
        captured: dict[str, list[str]] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(
                returncode=0, stdout=b"a_cal.fits\nb_cal.fits\n", stderr=b""
            )

        monkeypatch.setattr("noobfriend.core.io.remote.subprocess.run", fake_run)

        assert list_remote_dir("icrhome08:/data/stage2") == ["a_cal.fits", "b_cal.fits"]
        assert captured["cmd"][0] == "ssh"
        assert "icrhome08" in captured["cmd"]
        assert any("find" in part for part in captured["cmd"])

    def test_local_spec_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Not a remote spec"):
            list_remote_dir("/data/stage2")

    def test_failure_raises_remote_read_error(self, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            return SimpleNamespace(
                returncode=1, stdout=b"", stderr=b"no such directory"
            )

        monkeypatch.setattr("noobfriend.core.io.remote.subprocess.run", fake_run)

        with pytest.raises(RemoteReadError, match="no such directory"):
            list_remote_dir("icrhome08:/missing")


class TestWriteBytes:
    """Writing bytes to a local path (real) or a remote host:path (mocked SSH)."""

    def test_writes_local_creating_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "deep" / "blob.bin"
        payload = b"\x00written\xff"
        write_bytes(str(target), payload)
        assert target.read_bytes() == payload

    def test_remote_pipes_bytes_over_ssh(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr("noobfriend.core.io.remote.subprocess.run", fake_run)

        write_bytes("icrhome08:/data/2b/x_cal.fits", b"remote-bytes")

        assert captured["cmd"][0] == "ssh"
        assert "icrhome08" in captured["cmd"]
        assert any("mkdir -p" in part and "cat >" in part for part in captured["cmd"])
        assert captured["input"] == b"remote-bytes"

    def test_remote_failure_raises_write_error(self, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            return SimpleNamespace(
                returncode=1, stdout=b"", stderr=b"permission denied"
            )

        monkeypatch.setattr("noobfriend.core.io.remote.subprocess.run", fake_run)

        with pytest.raises(RemoteWriteError, match="permission denied"):
            write_bytes("icrhome08:/data/x.fits", b"x")
