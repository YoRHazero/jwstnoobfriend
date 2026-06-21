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

import posixpath
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


#: Per-host record of whether ranged reads (``dd`` / ``tail``) work on a host.
#: Optimistically assumed ``True`` (a GNU/Linux remote, as the ``find``-based
#: helpers already assume). A caller that finds a ranged read fails where a
#: whole-file read succeeds marks the host ``False`` via :func:`set_range_capable`
#: so it stops paying for a doomed ranged attempt on every later file.
_RANGE_CAPABLE: dict[str, bool] = {}


def range_capable(host: str) -> bool:
    """Whether ``host`` is (still) believed to support ranged ``dd`` / ``tail``."""
    return _RANGE_CAPABLE.get(host, True)


def set_range_capable(host: str, value: bool) -> None:
    """Record whether ``host`` supports ranged reads (see :data:`_RANGE_CAPABLE`)."""
    _RANGE_CAPABLE[host] = value


def _reset_range_capability() -> None:
    """Forget all recorded host capabilities (for tests)."""
    _RANGE_CAPABLE.clear()


class RemoteReadError(RuntimeError):
    """A remote ``ssh`` read failed (unreachable host, missing file, ...)."""


class RemoteWriteError(RuntimeError):
    """A remote ``ssh`` write failed (unreachable host, permission denied, ...)."""


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
    local disk. This pulls the *whole* file; to read only part of a FITS product
    (its metadata tail or a single extension) prefer the byte-frugal readers in
    :mod:`noobfriend.core.io.fits` driven by an accessor::

        from noobfriend.core.io import open_accessor, read_meta

        meta = read_meta(open_accessor("icrhome08:/data/x_cal.fits"))

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


def fetch_range(spec: str | Path, offset: int, length: int) -> bytes:
    """Return ``length`` bytes of a file starting at ``offset``.

    The partial-read counterpart of :func:`fetch_bytes`: only the requested byte
    range crosses the network (or is read off the local disk), so a caller that
    knows a FITS extension's position can pull just that extension instead of the
    whole file. A local spec is served with ``seek`` + ``read``; a remote
    ``[user@]host:path`` spec is served with ``dd ... iflag=skip_bytes,count_bytes``
    so ``offset`` and ``length`` are interpreted in bytes regardless of block size.

    Parameters
    ----------
    spec : str or Path
        A local path, or an ``[user@]host:path`` string (see :func:`_parse_spec`
        for the local-versus-remote rule).
    offset : int
        Byte offset of the first byte to read (``0``-based).
    length : int
        Number of bytes to read. A range that runs past end-of-file yields the
        bytes that exist (a short read), never an error.

    Returns
    -------
    bytes
        The requested range, possibly shorter than ``length`` at end-of-file.

    Raises
    ------
    ValueError
        ``spec`` is malformed, or ``offset`` / ``length`` is negative.
    FileNotFoundError
        A local ``spec`` does not exist.
    RemoteReadError
        The remote ``ssh``/``dd`` failed (unreachable host, missing file,
        authentication declined under ``BatchMode``, ...).

    Notes
    -----
    Assumes a GNU/Linux remote: it relies on ``dd``'s ``iflag=skip_bytes,count_bytes``
    (as the other remote helpers rely on GNU ``find``/``tail``).
    """
    if offset < 0 or length < 0:
        raise ValueError(
            f"offset and length must be non-negative, got {offset=}, {length=}."
        )
    host, path = _parse_spec(spec)
    if host is None:
        with Path(path).open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)

    remote_cmd = (
        f"dd if={shlex.quote(path)} bs=4M skip={offset} count={length} "
        "iflag=skip_bytes,count_bytes status=none"
    )
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteReadError(
            detail or f"ssh dd {host}:{path} exited with status {proc.returncode}"
        )
    return proc.stdout


def fetch_tail(spec: str | Path, length: int) -> bytes:
    """Return the last ``length`` bytes of a file.

    The trailing-range counterpart of :func:`fetch_bytes`: useful for a format
    whose index lives at the end of the file (a JWST product's ASDF metadata HDU
    is the last HDU), so the metadata can be pulled without reading the file's
    large leading data extensions. A local spec seeks from end-of-file; a remote
    ``[user@]host:path`` spec is served with ``tail -c``.

    Parameters
    ----------
    spec : str or Path
        A local path, or an ``[user@]host:path`` string (see :func:`_parse_spec`
        for the local-versus-remote rule).
    length : int
        Number of trailing bytes to read. A file shorter than ``length`` yields
        its whole content, never an error.

    Returns
    -------
    bytes
        The file's last ``length`` bytes (or the whole file if it is shorter).

    Raises
    ------
    ValueError
        ``spec`` is malformed, or ``length`` is negative.
    FileNotFoundError
        A local ``spec`` does not exist.
    RemoteReadError
        The remote ``ssh``/``tail`` failed (unreachable host, missing file,
        authentication declined under ``BatchMode``, ...).

    Notes
    -----
    Assumes a GNU/Linux remote: it relies on ``tail -c`` (as the other remote
    helpers rely on GNU ``find``/``dd``).
    """
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length=}.")
    host, path = _parse_spec(spec)
    if host is None:
        target = Path(path)
        with target.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - length))
            return handle.read()

    remote_cmd = f"tail -c {length} -- {shlex.quote(path)}"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteReadError(
            detail or f"ssh tail {host}:{path} exited with status {proc.returncode}"
        )
    return proc.stdout


def write_bytes(spec: str | Path, data: bytes) -> None:
    """Write ``data`` to a local path or a remote ``host:path``, making parents.

    The write counterpart of :func:`fetch_bytes`: a local spec is written to disk
    (its parent directories are created first); a remote ``[user@]host:path`` spec
    is streamed over ``ssh`` -- ``mkdir -p`` then ``cat`` from stdin -- so the
    bytes are uploaded in one shot. This is how a reduction script run off the
    data server stores a product back into a remote stage directory.

    Parameters
    ----------
    spec : str or Path
        A local path, or an ``[user@]host:path`` string naming the destination
        file on a host resolvable through the user's ``~/.ssh/config`` (see
        :func:`_parse_spec` for the local-versus-remote rule).
    data : bytes
        The content to write.

    Raises
    ------
    ValueError
        The spec is malformed (see :func:`_parse_spec`).
    RemoteWriteError
        The remote ``ssh`` upload failed (unreachable host, permission denied,
        authentication declined under ``BatchMode``, ...).
    """
    host, path = _parse_spec(spec)
    if host is None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return

    directory = posixpath.dirname(path) or "."
    remote_cmd = f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteWriteError(
            detail or f"ssh write {host}:{path} exited with status {proc.returncode}"
        )


def list_remote_dir(spec: str) -> list[str]:
    """List the regular-file names in a remote directory, one level deep.

    Runs ``ssh <host> find <path> -maxdepth 1 -type f`` and returns the bare
    file names (no directory part), so the caller can match them against a glob
    and rebuild full ``host:path`` specs. Like :func:`fetch_bytes`, it reuses
    the shared ``ControlMaster`` connection and delegates authentication and
    host resolution to the system ``ssh``.

    Parameters
    ----------
    spec : str
        A remote ``[user@]host:path`` directory spec.

    Returns
    -------
    list of str
        The names of the regular files in the directory (not full paths), in
        the order ``find`` reports them.

    Raises
    ------
    ValueError
        ``spec`` is local (has no ``host:`` part), or is otherwise malformed
        (see :func:`_parse_spec`).
    RemoteReadError
        The remote ``ssh``/``find`` failed (unreachable host, missing
        directory, authentication declined under ``BatchMode``, ...).

    Notes
    -----
    Assumes a GNU/Linux remote: it relies on ``find``'s ``-printf`` (as the
    fetch CLI's remote helpers do). Listing is non-recursive (``-maxdepth 1``)
    and excludes directories.
    """
    host, path = _parse_spec(spec)
    if host is None:
        raise ValueError(f"Not a remote spec: {spec!r}. Expected '[user@]host:path'.")

    remote_cmd = f"find {shlex.quote(path)} -maxdepth 1 -type f -printf '%f\\n'"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteReadError(
            detail or f"ssh find {host}:{path} exited with status {proc.returncode}"
        )
    return [name for name in proc.stdout.decode(errors="replace").splitlines() if name]


def remote_exists(host: str, path: str) -> bool:
    """Whether ``path`` exists on ``host``, via ``ssh <host> test -e``.

    Mirrors :meth:`pathlib.Path.exists` (any file type counts) for a path that
    lives on a remote host. Reuses the shared ``ControlMaster`` connection and
    delegates authentication and host resolution to the system ``ssh``.

    Parameters
    ----------
    host : str
        The ``[user@]host`` to test on (e.g. a ``~/.ssh/config`` alias).
    path : str
        The remote path to test.

    Returns
    -------
    bool
        ``True`` if the path exists, ``False`` if it does not.

    Raises
    ------
    RemoteReadError
        The ``ssh`` connection itself failed (unreachable host, authentication
        declined under ``BatchMode``, ...) — distinct from the path simply not
        existing, which returns ``False``.
    """
    remote_cmd = f"test -e {shlex.quote(path)}"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    detail = proc.stderr.decode(errors="replace").strip()
    raise RemoteReadError(
        detail or f"ssh test -e {host}:{path} exited with status {proc.returncode}"
    )


def remote_makedirs(host: str, path: str) -> None:
    """Create ``path`` (and parents) on ``host``, via ``ssh <host> mkdir -p``.

    The remote analogue of ``Path.mkdir(parents=True, exist_ok=True)``: existing
    directories are not an error. Reuses the shared ``ControlMaster`` connection
    and delegates authentication and host resolution to the system ``ssh``.

    Parameters
    ----------
    host : str
        The ``[user@]host`` to create the directory on.
    path : str
        The remote directory path to create.

    Raises
    ------
    RemoteReadError
        The remote ``ssh``/``mkdir`` failed (unreachable host, permission
        denied, authentication declined under ``BatchMode``, ...).
    """
    remote_cmd = f"mkdir -p -- {shlex.quote(path)}"
    proc = subprocess.run(  # noqa: S603
        ["ssh", *_ssh_opts(), host, remote_cmd],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RemoteReadError(
            detail or f"ssh mkdir -p {host}:{path} exited with status {proc.returncode}"
        )
