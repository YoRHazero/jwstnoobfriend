"""Runtime read of the mount sidecar and canonical <-> local path resolution.

``env mount`` sshfs-mounts a remote data root locally and rewrites the ``.env``
to local paths, so most library code sees a plain local filesystem and needs no
mount awareness. The reduction *runner* is the deliberate exception: the NooBox
manifest must record *canonical* ``host:server_path`` locations (portable off the
reduce machine, not this machine's mount point), and file IO can skip the remote
round-trip by opening the mounted path directly. Both read the sidecar
(``.noob_mount.json``) that ``env mount`` leaves next to the ``.env``.

This read-side model + resolver live in ``core`` so the (core-only) runner can use
them without importing ``cli``; ``cli.env`` re-imports them for the mount command
and keeps the write-side (planning, sshfs, save/remove).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from noobfriend.core.env._loader import find_project_root

#: Sidecar file ``env mount`` writes next to the ``.env`` it rewrote.
SIDECAR_NAME: str = ".noob_mount.json"


@dataclass
class MountState:
    """The mount sidecar: what ``env mount`` mounted and the values to restore.

    Attributes
    ----------
    host : str
        The ssh host (the original ``NOOB_SERVER``) that was mounted.
    data_root : str
        The remote ``DATA_ROOT_PATH`` that was mounted.
    mountpoint : str
        The local directory the remote root was mounted onto.
    saved : dict of str to str
        Original ``.env`` values, keyed by variable name, to restore on unmount.
    """

    host: str
    data_root: str
    mountpoint: str
    saved: dict[str, str]


def sidecar_path(env_dir: Path) -> Path:
    """Return the sidecar path for the ``.env`` directory ``env_dir``."""
    return env_dir / SIDECAR_NAME


def load_state(env_dir: Path) -> MountState | None:
    """Load the mount sidecar from ``env_dir``, or ``None`` if not mounted."""
    path = sidecar_path(env_dir)
    if not path.exists():
        return None
    return MountState(**json.loads(path.read_text()))


def find_mount_state() -> MountState | None:
    """Load the mount sidecar next to the active ``.env``, or ``None``.

    Mirrors the ``.env`` search of
    :func:`~noobfriend.core.env.load_environment` -- the sidecar sits beside the
    ``.env`` that ``env mount`` rewrote (project root, then cwd). Returns ``None``
    when nothing is mounted.
    """
    for env_dir in _env_dirs():
        state = load_state(env_dir)
        if state is not None:
            return state
    return None


def to_canonical(path: str | Path) -> str:
    """Return the canonical ``host:server_path`` for a locally-mounted ``path``.

    A ``path`` under the active mountpoint is rewritten to
    ``<host>:<data_root>/<relative>`` -- the portable location the NooBox manifest
    should record. Anything not under a mountpoint (nothing mounted, an
    already-remote ``host:path``, or a plain local path) is returned unchanged.
    """
    state = find_mount_state()
    if state is None:
        return str(path)
    rel = _relative_to(str(path), state.mountpoint)
    if rel is None:
        return str(path)
    base = state.data_root.rstrip("/")
    return f"{state.host}:{base}" if rel == "." else f"{state.host}:{base}/{rel}"


def to_local(location: str) -> Path | None:
    """Return the local path for ``location`` when it resolves on this machine.

    - A plain local path (no ``host:`` prefix) is returned as a :class:`Path`.
    - A ``host:server_path`` whose host and root match the active mount is
      rewritten to the mounted local path.
    - Anything else (a genuinely remote host, or nothing mounted) returns
      ``None``, signalling the caller to fall back to a remote fetch.
    """
    host, sep, remote = str(location).partition(":")
    if not sep or not host or host.startswith("/") or host.startswith("."):
        return Path(location)  # already a local path
    state = find_mount_state()
    if state is None or host != state.host:
        return None
    rel = _relative_to(remote, state.data_root)
    if rel is None:
        return None
    mount = Path(state.mountpoint)
    return mount if rel == "." else mount / rel


def _env_dirs() -> list[Path]:
    """Directories to look for the sidecar in, mirroring the ``.env`` search."""
    root = find_project_root()
    cwd = Path.cwd()
    dirs = [cwd]
    if root is not None and root != cwd:
        dirs.insert(0, root)
    return dirs


def _relative_to(path: str, root: str) -> str | None:
    """Return ``path`` relative to ``root`` (``"."`` when equal), or ``None``."""
    try:
        rel = Path(path).relative_to(Path(root))
    except ValueError:
        return None
    return str(rel)
