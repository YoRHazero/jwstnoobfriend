"""Tests for the runtime mount resolver (canonical <-> local path resolution)."""

import json
from pathlib import Path

import pytest

from noobfriend.core.env import _mount as mount_mod
from noobfriend.core.env import find_mount_state, to_canonical, to_local


def _write_sidecar(env_dir: Path, mountpoint: Path) -> None:
    (env_dir / ".noob_mount.json").write_text(
        json.dumps(
            {
                "host": "user@server",
                "data_root": "/disk/data",
                "mountpoint": str(mountpoint),
                "saved": {},
            }
        )
    )


@pytest.fixture
def mounted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mount_mod, "find_project_root", lambda: None)
    mnt = tmp_path / "mnt"
    _write_sidecar(tmp_path, mnt)
    return mnt


@pytest.fixture
def unmounted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mount_mod, "find_project_root", lambda: None)
    return tmp_path


def test_no_mount_is_a_noop(unmounted: Path) -> None:
    assert find_mount_state() is None
    assert to_canonical("/local/2b/x.fits") == "/local/2b/x.fits"
    assert to_local("/local/2b/x.fits") == Path("/local/2b/x.fits")
    assert to_local("host:/disk/2b/x.fits") is None  # remote, nothing mounted


def test_to_canonical_rewrites_mounted_path(mounted: Path) -> None:
    p = mounted / "stage2" / "2b" / "x.fits"
    assert to_canonical(str(p)) == "user@server:/disk/data/stage2/2b/x.fits"
    assert to_canonical("/elsewhere/x.fits") == "/elsewhere/x.fits"  # outside mount


def test_to_local_resolves_canonical_and_local(mounted: Path) -> None:
    assert (
        to_local("user@server:/disk/data/stage2/2b/x.fits")
        == mounted / "stage2" / "2b" / "x.fits"
    )
    assert to_local(str(mounted / "x.fits")) == mounted / "x.fits"  # plain local
    assert to_local("other@host:/disk/data/x.fits") is None  # host mismatch
    assert to_local("user@server:/elsewhere/x.fits") is None  # outside data root


def test_canonical_local_roundtrip(mounted: Path) -> None:
    p = mounted / "stage2" / "2b" / "x.fits"
    assert to_local(to_canonical(str(p))) == p
