"""Pure helpers for ``env mount`` / ``env unmount``: planning, sidecar, sshfs.

Mounting is a one-time configuration transform: the remote ``DATA_ROOT_PATH`` is
sshfs-mounted locally and the ``.env`` is rewritten so the library sees a local
``localhost`` server with local paths. Most library code stays mount-unaware; the
reduction runner is the deliberate exception, reading the *sidecar* to record
canonical server paths (see :mod:`noobfriend.core.env._mount`). The sidecar also
lets :func:`env unmount` restore the original ``.env`` verbatim. By default the
mount lives outside any git repository (under the user cache), so the mount tree
never lands in a working copy.
"""

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from noobfriend.core.env._mount import MountState, sidecar_path

#: Values of ``NOOB_SERVER`` that mean "already local -- nothing to mount".
_LOCAL_ALIASES: frozenset[str] = frozenset({"", "localhost", "local"})
#: Tuned sshfs options (benchmarked): reconnect, cache, no compression, fast cipher.
_SSHFS_OPTS: str = (
    "reconnect,cache=yes,kernel_cache,Compression=no,Ciphers=aes128-gcm@openssh.com"
)


def is_remote_server(value: str | None) -> bool:
    """Whether ``value`` names a remote host (vs the local machine)."""
    return value is not None and value.strip().lower() not in _LOCAL_ALIASES


def default_mountpoint(host: str, data_root: str) -> Path:
    """Return the default out-of-repo mountpoint for ``host:data_root``.

    Lives under the user cache (``$XDG_CACHE_HOME`` or ``~/.cache``), keyed by
    host and data root, so it is never inside a git working copy and is shared
    by every ``.env`` that points at the same remote root.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    key = data_root.strip("/").replace("/", "_") or "root"
    return cache / "noobfriend" / "mounts" / host / key


def _relative_to(path: str, root: str) -> str | None:
    """Return ``path`` relative to ``root``, or ``None`` if it is not under it."""
    try:
        rel = Path(path).relative_to(Path(root))
    except ValueError:
        return None
    return str(rel)


def plan_rewrite(
    values: dict[str, str | None], mountpoint: Path
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Plan the ``.env`` rewrite that makes a mounted root look local.

    Parameters
    ----------
    values : dict of str to (str or None)
        The current ``.env`` values.
    mountpoint : Path
        Where the remote ``DATA_ROOT_PATH`` is mounted.

    Returns
    -------
    saved : dict of str to str
        Original values of every variable being changed (to restore later).
    new : dict of str to str
        The replacement values: ``NOOB_SERVER`` -> ``localhost``,
        ``DATA_ROOT_PATH`` -> ``mountpoint``, and each ``STAGE_*_PATH`` under the
        old root rewritten under ``mountpoint``.
    uncovered : list of str
        Names of ``STAGE_*_PATH`` variables that are *not* under
        ``DATA_ROOT_PATH`` and so cannot be served by this single mount; they are
        left untouched (still remote).
    """
    data_root = values.get("DATA_ROOT_PATH") or ""
    saved: dict[str, str] = {}
    new: dict[str, str] = {}
    uncovered: list[str] = []

    saved["NOOB_SERVER"] = values.get("NOOB_SERVER") or ""
    new["NOOB_SERVER"] = "localhost"
    saved["DATA_ROOT_PATH"] = data_root
    new["DATA_ROOT_PATH"] = str(mountpoint)

    for name, value in values.items():
        if not (name.startswith("STAGE_") and name.endswith("_PATH") and value):
            continue
        rel = _relative_to(value, data_root)
        if rel is None:
            uncovered.append(name)
            continue
        saved[name] = value
        new[name] = str(mountpoint if rel == "." else mountpoint / rel)
    return saved, new, uncovered


def save_state(env_dir: Path, state: MountState) -> None:
    """Write the mount sidecar into ``env_dir``."""
    sidecar_path(env_dir).write_text(json.dumps(asdict(state), indent=2))


def remove_state(env_dir: Path) -> None:
    """Delete the mount sidecar from ``env_dir`` if present."""
    sidecar_path(env_dir).unlink(missing_ok=True)


def sshfs_available() -> bool:
    """Whether the ``sshfs`` binary is on ``PATH``."""
    return shutil.which("sshfs") is not None


def is_mounted(mountpoint: Path) -> bool:
    """Whether ``mountpoint`` is currently a mount (treating a stale one as such)."""
    try:
        return mountpoint.is_dir() and os.path.ismount(mountpoint)
    except OSError:
        return True  # a stale/broken FUSE mount still needs unmounting


def run_sshfs(
    host: str, remote_path: str, mountpoint: Path
) -> subprocess.CompletedProcess:
    """Mount ``host:remote_path`` at ``mountpoint`` with the tuned options."""
    mountpoint.mkdir(parents=True, exist_ok=True)
    return subprocess.run(  # noqa: S603
        ["sshfs", f"{host}:{remote_path}", str(mountpoint), "-o", _SSHFS_OPTS],  # noqa: S607
        capture_output=True,
    )


def run_unmount(mountpoint: Path, *, force: bool) -> subprocess.CompletedProcess:
    """Unmount ``mountpoint``; ``force`` escalates to a lazy/forced unmount.

    On Linux ``force`` uses ``fusermount -u -z`` (lazy: detach now, free when the
    last handle closes); on macOS it uses ``umount -f`` (forced). Without
    ``force`` a busy mount fails cleanly rather than disrupting an open reader.
    """
    if platform.system() == "Darwin":
        cmd = (
            ["umount", "-f", str(mountpoint)] if force else ["umount", str(mountpoint)]
        )
    else:
        cmd = (
            ["fusermount", "-u", "-z", str(mountpoint)]
            if force
            else ["fusermount", "-u", str(mountpoint)]
        )
    return subprocess.run(cmd, capture_output=True)  # noqa: S603


def git_work_tree(path: Path) -> Path | None:
    """Return the git work-tree root that contains ``path``, or ``None``."""
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],  # noqa: S607
        capture_output=True,
        text=True,
    )
    return Path(proc.stdout.strip()) if proc.returncode == 0 else None


def is_tracked(path: Path) -> bool:
    """Whether ``path`` is tracked by git (so a rewrite would dirty the repo)."""
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],  # noqa: S607
        capture_output=True,
    )
    return proc.returncode == 0


def ensure_gitignored(directory: Path, entry: str) -> bool:
    """Idempotently add ``entry`` to ``directory/.gitignore``; return if added.

    Returns ``True`` when the entry was newly appended, ``False`` when it was
    already present.
    """
    gitignore = directory / ".gitignore"
    lines = gitignore.read_text().splitlines() if gitignore.exists() else []
    if entry in {line.strip() for line in lines}:
        return False
    with gitignore.open("a") as handle:
        if lines and lines[-1].strip():
            handle.write("\n")
        handle.write(f"# noobfriend mount artifact\n{entry}\n")
    return True
