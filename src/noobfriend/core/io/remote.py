"""Fetch the raw bytes of a file that may live on a remote host.

:func:`fetch_bytes` resolves a *spec* — a local path or an ``[user@]host:path``
string — into the file's full byte content. A local spec is read straight off
disk; a remote spec is streamed in one shot with ``ssh <host> cat`` so the
bytes land in memory without ever touching the local disk. As with the rest of
:mod:`noobfriend.core.io`, nothing is cached: each call performs one transfer.
Memoisation is left to the caller, which is the natural place to keep a
whole-file cache (store the returned ``bytes`` and wrap a fresh
:class:`io.BytesIO` around them for every reader call).

Remote access delegates entirely to the system ``ssh`` so the user's
``~/.ssh/config`` aliases, keys and agent are used as-is. A dedicated
``ControlMaster`` socket (distinct from the one the fetch CLI uses) multiplexes
repeated reads of the same host onto a single authenticated connection, so a
notebook that reads many files only pays the SSH handshake once.
"""

import shlex
import subprocess
from pathlib import Path

#: ``ssh`` connection-multiplexing socket, kept separate from the fetch CLI's
#: own ControlPath so the long-lived notebook read session and the short-lived
#: download command never share — and thus never tear down — each other's
#: master connection. ``%C`` hashes host/port/user so each target gets its own.
_CONTROL_PATH: str = "~/.ssh/noobfriend-io-cm-%C"
_CONNECT_TIMEOUT: int = 10
_CONTROL_PERSIST: int = 600


class RemoteReadError(RuntimeError):
    """A remote ``ssh`` read failed (unreachable host, missing file, ...)."""


def _parse_spec(spec: str | Path) -> tuple[str | None, str]:
    """Split a spec into ``(host, path)``, with ``host`` ``None`` when local.

    A :class:`~pathlib.Path` is always local. A :class:`str` is parsed with the
    ``scp`` disambiguation rule: it is remote only when a ``:`` appears before
    any ``/`` (so ``host:/data/x.fits`` and ``user@host:rel/x.fits`` are remote,
    while ``./x.fits`` and ``/weird/a:b`` stay local). Unlike the fetch CLI's
    directory-destination parser, an empty remote path is rejected here: this
    resolves a single *file*, not a directory.

    Parameters
    ----------
    spec : str or Path
        The file location.

    Returns
    -------
    host : str or None
        The ``[user@]host`` for a remote spec, or ``None`` when local.
    path : str
        The (remote or local) file path.

    Raises
    ------
    ValueError
        The spec looks remote but the host part or the path part is empty.
    """
    if isinstance(spec, Path):
        return None, str(spec)

    colon = spec.find(":")
    slash = spec.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        host, _, path = spec.partition(":")
        if not host:
            raise ValueError(
                f"Invalid remote spec {spec!r}: expected '[user@]host:path'."
            )
        if not path:
            raise ValueError(
                f"Invalid remote spec {spec!r}: a file path is required after ':'."
            )
        return host, path
    return None, spec


def _ssh_opts() -> list[str]:
    """Build the ``ssh`` options: non-interactive, timed, connection-multiplexed.

    ``BatchMode=yes`` turns a password requirement into a fast failure instead
    of an interactive prompt, and the ``ControlMaster`` settings reuse a single
    authenticated connection across repeated reads of the same host.
    """
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={_CONNECT_TIMEOUT}",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={_CONTROL_PATH}",
        "-o",
        f"ControlPersist={_CONTROL_PERSIST}",
    ]


def fetch_bytes(spec: str | Path) -> bytes:
    """Return the full byte content of a local or remote file.

    A local spec is read directly off disk. A remote ``[user@]host:path`` spec
    is streamed into memory with ``ssh <host> cat`` — the bytes never touch the
    local disk. The result is suitable for handing to the
    :mod:`noobfriend.core.io.fits` readers via :class:`io.BytesIO`::

        from io import BytesIO

        from noobfriend.core.io import fetch_bytes, read_data

        data = read_data(BytesIO(fetch_bytes("icrhome08:/data/x_cal.fits")))

    Parameters
    ----------
    spec : str or Path
        A local path, or an ``[user@]host:path`` string naming a file on a host
        resolvable through the user's ``~/.ssh/config`` (see :func:`_parse_spec`
        for the local-versus-remote rule).

    Returns
    -------
    bytes
        The file's full content, loaded into memory.

    Raises
    ------
    ValueError
        The spec is malformed (see :func:`_parse_spec`).
    FileNotFoundError
        A local ``spec`` does not exist.
    RemoteReadError
        The remote ``ssh``/``cat`` failed (unreachable host, missing file,
        authentication declined under ``BatchMode``, ...).
    """
    host, path = _parse_spec(spec)
    if host is None:
        return Path(path).read_bytes()

    remote_cmd = f"cat -- {shlex.quote(path)}"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteReadError(
            detail or f"ssh cat {host}:{path} exited with status {proc.returncode}"
        )
    return proc.stdout
