"""Unit tests for env mount/unmount: rewrite planning, sidecar, gitignore, flow.

The sshfs / unmount / git subprocesses are mocked, so no real mount is made.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import noobfriend.cli.env.mount as mount_cli
from noobfriend.cli.env._io import read_env_file
from noobfriend.cli.env._mount import (
    default_mountpoint,
    ensure_gitignored,
    is_remote_server,
    plan_rewrite,
    save_state,
)
from noobfriend.core.env._mount import MountState, load_state, sidecar_path


class TestIsRemoteServer:
    """Local aliases are not remote; a real host is."""

    @pytest.mark.parametrize("value", [None, "", "localhost", "LOCAL", " local "])
    def test_local(self, value: str | None) -> None:
        assert is_remote_server(value) is False

    def test_remote(self) -> None:
        assert is_remote_server("icrhome08") is True


class TestDefaultMountpoint:
    """The default mountpoint lives in the user cache, keyed by host + root."""

    def test_uses_xdg_cache(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        mp = default_mountpoint("icrhome08", "/disk/cos/data")
        assert mp == tmp_path / "noobfriend" / "mounts" / "icrhome08" / "disk_cos_data"


class TestPlanRewrite:
    """The rewrite makes a mounted root look local; strays stay remote."""

    def test_rewrites_server_root_and_stage(self) -> None:
        values = {
            "NOOB_SERVER": "icrhome08",
            "DATA_ROOT_PATH": "/data",
            "STAGE_2B_PATH": "/data/stage2/2b",
        }
        saved, new, uncovered = plan_rewrite(values, Path("/mnt/x"))
        assert new["NOOB_SERVER"] == "localhost"
        assert new["DATA_ROOT_PATH"] == "/mnt/x"
        assert new["STAGE_2B_PATH"] == "/mnt/x/stage2/2b"
        assert saved == values
        assert uncovered == []

    def test_stage_outside_root_is_uncovered(self) -> None:
        values = {
            "NOOB_SERVER": "icrhome08",
            "DATA_ROOT_PATH": "/data",
            "STAGE_X_PATH": "/elsewhere/x",
        }
        saved, new, uncovered = plan_rewrite(values, Path("/mnt/x"))
        assert uncovered == ["STAGE_X_PATH"]
        assert "STAGE_X_PATH" not in new
        assert "STAGE_X_PATH" not in saved


class TestSidecar:
    """The sidecar round-trips the restore state."""

    def test_save_load_remove(self, tmp_path: Path) -> None:
        state = MountState(
            host="icrhome08",
            data_root="/data",
            mountpoint="/mnt/x",
            saved={"NOOB_SERVER": "icrhome08"},
        )
        save_state(tmp_path, state)
        assert sidecar_path(tmp_path).exists()
        assert load_state(tmp_path) == state

    def test_load_absent_is_none(self, tmp_path: Path) -> None:
        assert load_state(tmp_path) is None


class TestEnsureGitignored:
    """Entries are appended once, idempotently."""

    def test_adds_then_idempotent(self, tmp_path: Path) -> None:
        assert ensure_gitignored(tmp_path, "noob_server/") is True
        assert ensure_gitignored(tmp_path, "noob_server/") is False
        assert "noob_server/" in (tmp_path / ".gitignore").read_text()


def _mock_mount_io(monkeypatch) -> None:
    """Make the mount command run without touching sshfs / git / the network."""
    ok = SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr(mount_cli, "sshfs_available", lambda: True)
    monkeypatch.setattr(mount_cli, "is_mounted", lambda p: False)
    monkeypatch.setattr(mount_cli, "run_sshfs", lambda h, r, mp: ok)
    monkeypatch.setattr(mount_cli, "run_unmount", lambda mp, force: ok)
    monkeypatch.setattr(mount_cli, "git_work_tree", lambda p: None)
    monkeypatch.setattr(mount_cli, "is_tracked", lambda p: False)


class TestMountFlow:
    """End-to-end .env rewrite and restore (sshfs mocked)."""

    def test_mount_then_unmount_restores(self, tmp_path: Path, monkeypatch) -> None:
        _mock_mount_io(monkeypatch)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "NOOB_SERVER=icrhome08\nDATA_ROOT_PATH=/data\nSTAGE_2B_PATH=/data/stage2/2b\n"
        )
        mnt = tmp_path / "mnt"

        mount_cli.cli_mount(env_file=env_file, path=mnt, force=False)
        after = read_env_file(env_file)
        assert after["NOOB_SERVER"] == "localhost"
        assert after["DATA_ROOT_PATH"] == str(mnt)
        assert after["STAGE_2B_PATH"] == str(mnt / "stage2" / "2b")
        assert sidecar_path(tmp_path).exists()

        mount_cli.cli_unmount(env_file=env_file, force=False)
        restored = read_env_file(env_file)
        assert restored["NOOB_SERVER"] == "icrhome08"
        assert restored["DATA_ROOT_PATH"] == "/data"
        assert restored["STAGE_2B_PATH"] == "/data/stage2/2b"
        assert not sidecar_path(tmp_path).exists()

    def test_mount_local_server_aborts(self, tmp_path: Path, monkeypatch) -> None:
        _mock_mount_io(monkeypatch)
        env_file = tmp_path / ".env"
        env_file.write_text("NOOB_SERVER=localhost\nDATA_ROOT_PATH=/data\n")
        with pytest.raises(typer.Exit):
            mount_cli.cli_mount(env_file=env_file)

    def test_double_mount_aborts_without_force(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _mock_mount_io(monkeypatch)
        env_file = tmp_path / ".env"
        env_file.write_text("NOOB_SERVER=icrhome08\nDATA_ROOT_PATH=/data\n")
        mount_cli.cli_mount(env_file=env_file, path=tmp_path / "mnt")
        with pytest.raises(typer.Exit):
            mount_cli.cli_mount(env_file=env_file, path=tmp_path / "mnt")

    def test_unmount_when_not_mounted_is_noop(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("NOOB_SERVER=localhost\n")
        mount_cli.cli_unmount(env_file=env_file)  # no sidecar -> returns cleanly
