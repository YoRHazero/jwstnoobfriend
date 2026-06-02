"""Remote download support for the fetch CLI.

Two destinations besides the local disk are supported, both delegating
authentication and host resolution entirely to the system ``ssh`` (so the
user's ``~/.ssh/config`` aliases, keys and agent are used as-is):

- **remote-direct** (:func:`remote_download`): the *remote* host downloads
  straight from MAST via ``curl``; the bytes never touch the local machine.
- **relay** (:func:`relay_download`): a fallback for hosts that cannot reach
  MAST. Each product is streamed to a local temp file, ``scp``-ed to the
  remote, then deleted, so local disk use stays bounded by the concurrency.

The choice between them is made by the caller after probing the remote with
:func:`probe_ssh` and :func:`probe_remote_mast`.
"""

import asyncio
import logging
import os
import posixpath
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from noobfriend.cli.fetch._options import (
    MAST_JWST_BASE_URL,
    MAST_JWST_DOWNLOAD_URL,
)
from noobfriend.core.env import get_settings
from noobfriend.core.io import HTTPSession

logger = logging.getLogger(__name__)

REMOTE_CONCURRENCY: int = 5
_LOCAL_SENTINELS: frozenset[str] = frozenset({"", "localhost", "local"})
_CONTROL_PATH: str = "~/.ssh/noobfriend-cm-%C"


class RemoteTransferError(RuntimeError):
    """A remote transfer failed in a way that aborts the whole command."""


@dataclass(frozen=True)
class DownloadTarget:
    """Resolved download destination.

    Parameters
    ----------
    ssh_dest : str or None
        The ``[user@]host`` string passed verbatim to ``ssh``/``scp`` for a
        remote target, or ``None`` for a local target.
    path : str
        The destination directory: a path on ``ssh_dest`` when remote, or a
        local path otherwise.
    """

    ssh_dest: str | None
    path: str

    @property
    def is_remote(self) -> bool:
        """Whether this target lives on a remote host."""
        return self.ssh_dest is not None


def parse_destination(spec: str) -> DownloadTarget:
    """Parse a ``[user@]host:path`` (remote) or plain path (local) specifier.

    Uses the ``scp`` disambiguation rule: the specifier is remote only when a
    ``:`` appears before any ``/``. This keeps local paths such as
    ``./downloads`` or ``/data/raw`` local while treating ``host:/data/raw``
    and ``user@host:rel`` as remote.

    Parameters
    ----------
    spec : str
        Destination specifier.

    Returns
    -------
    DownloadTarget
        Parsed target. Remote targets with an empty path default to ``"."``
        (the remote login directory).

    Raises
    ------
    ValueError
        If the specifier looks remote but the ``[user@]host`` part is empty
        (e.g. ``":/data"``).
    """
    colon = spec.find(":")
    slash = spec.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        ssh_dest, _, path = spec.partition(":")
        if not ssh_dest:
            raise ValueError(
                f"Invalid remote destination {spec!r}: expected '[user@]host:path'."
            )
        return DownloadTarget(ssh_dest=ssh_dest, path=path or ".")
    return DownloadTarget(ssh_dest=None, path=spec)


def resolve_target(output: str | None, current_folder: bool) -> DownloadTarget:
    """Resolve the download destination from arguments, then the environment.

    Priority: ``current_folder`` > ``output`` > environment. The environment
    layer reads :class:`~noobfriend.core.env.NoobSettings`: ``NOOB_SERVER`` (the
    destination host; unset / ``localhost`` / ``local`` mean local) and the
    ``START_STAGE`` / ``STAGE_<STAGE>_PATH`` mechanism for the path on that host.

    Parameters
    ----------
    output : str or None
        Value of ``-o/--output`` (a ``[user@]host:path`` or local path), or
        ``None`` when the option was not supplied.
    current_folder : bool
        Whether ``-c/--current-folder`` was supplied.

    Returns
    -------
    DownloadTarget
        The resolved destination.
    """
    if current_folder:
        return DownloadTarget(ssh_dest=None, path=str(Path.cwd()))
    if output is not None:
        return parse_destination(output)

    settings = get_settings()  # also loads the layered .env files
    stage_path = settings.start_stage_path()
    server = settings.noob_server.strip()
    if server.lower() in _LOCAL_SENTINELS:
        return DownloadTarget(
            ssh_dest=None, path=stage_path or str(Path.cwd() / "downloads")
        )
    return DownloadTarget(ssh_dest=server, path=stage_path or "downloads")


def _ssh_opts() -> list[str]:
    """Build the common ``ssh``/``scp`` options: non-interactive, timed, multiplexed.

    ``BatchMode=yes`` turns a password requirement into a fast failure instead
    of an interactive prompt, and ``ControlMaster`` reuses a single
    authenticated connection across the probe and every transfer.
    """
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={_CONTROL_PATH}",
        "-o",
        "ControlPersist=120",
    ]


async def probe_ssh(ssh_dest: str) -> str | None:
    """Check that ``ssh_dest`` is reachable non-interactively.

    Returns
    -------
    str or None
        ``None`` if a key/agent-authenticated connection succeeds; otherwise
        the captured ``ssh`` stderr (or a generic message) explaining why not.
    """
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        *_ssh_opts(),
        ssh_dest,
        "true",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode == 0:
        return None
    return stderr.decode().strip() or f"ssh exited with status {proc.returncode}"


async def probe_remote_mast(ssh_dest: str) -> bool:
    """Return whether ``ssh_dest`` has ``curl`` and can reach MAST."""
    remote_cmd = (
        "command -v curl >/dev/null 2>&1 && "
        f"curl -sI --max-time 10 {shlex.quote(MAST_JWST_BASE_URL)} >/dev/null 2>&1"
    )
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        *_ssh_opts(),
        ssh_dest,
        remote_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode == 0


def build_remote_script(
    products: list[dict],
    remote_dir: str,
    *,
    skip_exist: bool,
    concurrency: int = REMOTE_CONCURRENCY,
) -> str:
    """Build the server-side bash script for a remote-direct download.

    The script creates ``remote_dir`` and downloads every product with a
    bounded ``xargs -P`` pool of ``curl`` calls. Each worker prints exactly one
    status line — ``DONE``, ``SKIP`` or ``FAIL`` followed by the filename — so
    the local side can drive a progress bar (see :func:`parse_progress_line`).
    Downloads go to a ``.<name>.part`` temp and are atomically renamed on
    success. A failure to create ``remote_dir`` prints ``FATAL`` to stderr and
    exits non-zero.

    Parameters
    ----------
    products : list of dict
        Products with ``filename`` and ``size`` keys.
    remote_dir : str
        Destination directory on the remote host.
    skip_exist : bool
        When true, skip a file that already exists with the expected size.
    concurrency : int, default :data:`REMOTE_CONCURRENCY`
        Number of parallel ``curl`` workers on the remote.

    Returns
    -------
    str
        The script, suitable for ``ssh <dest> bash -s`` on stdin.

    Notes
    -----
    Assumes a GNU/Linux remote (``stat -c``). JWST product filenames contain
    no whitespace or shell metacharacters, so they are safe both in the
    whitespace-delimited work list and when appended to the download URL.
    """
    listing = "\n".join(f"{p['filename']} {p['size']}" for p in products)
    skip_flag = "1" if skip_exist else ""
    worker = (
        'name="$0"; size="$1"; dest="$DIR/$name"\n'
        'if [ -n "$SKIP" ] && [ -f "$dest" ] && '
        '[ "$(stat -c %s "$dest" 2>/dev/null)" = "$size" ]; then\n'
        '  echo "SKIP $name"; exit 0\n'
        "fi\n"
        'tmp="$DIR/.$name.part"\n'
        'if curl -fSL --retry 3 -o "$tmp" "$URL?product_name=$name" '
        '>/dev/null 2>&1 && mv -f "$tmp" "$dest"; then\n'
        '  echo "DONE $name"\n'
        "else\n"
        '  rm -f "$tmp" 2>/dev/null; echo "FAIL $name"\n'
        "fi"
    )
    return (
        "set -u\n"
        f"DIR={shlex.quote(remote_dir)}\n"
        f"URL={shlex.quote(MAST_JWST_DOWNLOAD_URL)}\n"
        f"SKIP={shlex.quote(skip_flag)}\n"
        "export DIR URL SKIP\n"
        'mkdir -p "$DIR" || { echo "FATAL: cannot create $DIR" >&2; exit 3; }\n'
        f"cat <<'__NOOB_LIST__' | xargs -P {int(concurrency)} -n2 sh -c {shlex.quote(worker)}\n"
        f"{listing}\n"
        "__NOOB_LIST__\n"
    )


def parse_progress_line(
    line: str,
) -> tuple[Literal["DONE", "SKIP", "FAIL"], str] | None:
    """Parse one ``"<STATUS> <filename>"`` line from the remote script.

    Returns
    -------
    tuple or None
        ``(status, filename)`` for a recognised status, else ``None``.
    """
    stripped = line.strip()
    for status in ("DONE", "SKIP", "FAIL"):
        prefix = f"{status} "
        if stripped.startswith(prefix):
            return status, stripped[len(prefix) :]  # type: ignore[return-value]
    return None


async def remote_download(
    ssh_dest: str,
    remote_dir: str,
    products: list[dict],
    progress: Progress,
    *,
    skip_exist: bool,
    concurrency: int = REMOTE_CONCURRENCY,
) -> tuple[int, list[str]]:
    """Run a remote-direct download and report per-file outcomes.

    Parameters
    ----------
    ssh_dest : str
        ``[user@]host`` to run the download on.
    remote_dir : str
        Destination directory on the remote host.
    products : list of dict
        Products to download.
    progress : Progress
        Live display; a single ``M/N`` task is advanced as files finish.
    skip_exist : bool
        Forwarded into the remote script.
    concurrency : int, default :data:`REMOTE_CONCURRENCY`
        Parallel ``curl`` workers on the remote.

    Returns
    -------
    succeeded : int
        Number of files downloaded or skipped.
    failed : list of str
        Filenames the remote reported as ``FAIL``.

    Raises
    ------
    RemoteTransferError
        If the remote shell aborts (e.g. cannot create ``remote_dir``).
    """
    script = build_remote_script(
        products, remote_dir, skip_exist=skip_exist, concurrency=concurrency
    )
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        *_ssh_opts(),
        ssh_dest,
        "bash",
        "-s",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert (
        proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    )

    async def _feed() -> None:
        proc.stdin.write(script.encode())
        await proc.stdin.drain()
        proc.stdin.close()

    feed_task = asyncio.create_task(_feed())
    task_id = progress.add_task("Downloading on remote", total=len(products))
    succeeded = 0
    failed: list[str] = []
    async for raw in proc.stdout:
        parsed = parse_progress_line(raw.decode(errors="replace"))
        if parsed is None:
            continue
        status, name = parsed
        progress.advance(task_id)
        if status == "FAIL":
            failed.append(name)
        else:
            succeeded += 1

    stderr = (await proc.stderr.read()).decode(errors="replace").strip()
    await feed_task
    await proc.wait()
    if proc.returncode != 0:
        raise RemoteTransferError(stderr or f"remote shell exited {proc.returncode}")
    return succeeded, failed


async def _remote_existing_sizes(ssh_dest: str, remote_dir: str) -> dict[str, int]:
    """Return ``{filename: size}`` for files already in ``remote_dir`` (one round trip)."""
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} 2>/dev/null && "
        "find . -maxdepth 1 -type f -printf '%f\\t%s\\n' 2>/dev/null"
    )
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        *_ssh_opts(),
        ssh_dest,
        remote_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    sizes: dict[str, int] = {}
    for line in out.decode(errors="replace").splitlines():
        name, _, size = line.partition("\t")
        if name and size.isdigit():
            sizes[name] = int(size)
    return sizes


async def _relay_one(
    http: HTTPSession,
    ssh_dest: str,
    remote_dir: str,
    product: dict,
    progress: Progress,
    total_task_id: int,
    existing: dict[str, int],
) -> Literal["done", "skip", "fail"]:
    """Download one product locally, ``scp`` it to the remote, then clean up."""
    filename = product["filename"]
    if existing.get(filename) == product["size"]:
        logger.info("Remote already has %s, skipping.", filename)
        progress.advance(total_task_id)
        return "skip"

    task_id = progress.add_task(
        f"Downloading {filename}", total=product["size"] or None
    )
    tmp = Path(tempfile.gettempdir()) / f".noobfriend-{os.getpid()}-{filename}"
    try:
        await http.download_to_file(
            MAST_JWST_DOWNLOAD_URL,
            tmp,
            method="GET",
            params={"product_name": filename},
            progress_callback=lambda downloaded, total: progress.update(
                task_id, completed=downloaded, total=total or None
            ),
        )
        progress.update(task_id, description=f"Uploading {filename}")
        scp = await asyncio.create_subprocess_exec(
            "scp",
            *_ssh_opts(),
            str(tmp),
            f"{ssh_dest}:{shlex.quote(posixpath.join(remote_dir, filename))}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await scp.communicate()
        if scp.returncode != 0:
            logger.error(
                "scp of %s failed: %s",
                filename,
                stderr.decode(errors="replace").strip(),
            )
            return "fail"
        return "done"
    except Exception as error:
        logger.error("Failed to relay %s: %s", filename, error)
        return "fail"
    finally:
        tmp.unlink(missing_ok=True)
        progress.remove_task(task_id)
        progress.advance(total_task_id)


async def relay_download(
    ssh_dest: str,
    remote_dir: str,
    products: list[dict],
    progress: Progress,
    *,
    skip_exist: bool,
    concurrency: int = REMOTE_CONCURRENCY,
) -> tuple[int, list[str]]:
    """Relay products through the local machine to the remote (fallback path).

    Each product is downloaded to a local temp file, ``scp``-ed to the remote
    and then deleted, so local disk use peaks at roughly
    ``concurrency × file_size`` rather than the whole dataset.

    Parameters
    ----------
    ssh_dest, remote_dir, products, progress
        As in :func:`remote_download`. ``progress`` shows per-file byte
        progress plus an overall total.
    skip_exist : bool
        When true, files already present on the remote with the expected size
        are skipped (resolved with a single upfront listing).
    concurrency : int, default :data:`REMOTE_CONCURRENCY`
        Maximum simultaneous relays.

    Returns
    -------
    succeeded : int
        Number of files transferred or skipped.
    failed : list of str
        Filenames that failed to download or upload.

    Raises
    ------
    RemoteTransferError
        If the remote directory cannot be created.
    """
    mk = await asyncio.create_subprocess_exec(
        "ssh",
        *_ssh_opts(),
        ssh_dest,
        "mkdir",
        "-p",
        remote_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await mk.communicate()
    if mk.returncode != 0:
        raise RemoteTransferError(
            stderr.decode(errors="replace").strip()
            or f"cannot create remote directory {remote_dir}"
        )

    existing = await _remote_existing_sizes(ssh_dest, remote_dir) if skip_exist else {}
    http = HTTPSession(max_connections=concurrency)
    total_task_id = progress.add_task("Total Progress:", total=len(products))
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(product: dict) -> Literal["done", "skip", "fail"]:
        async with semaphore:
            return await _relay_one(
                http, ssh_dest, remote_dir, product, progress, total_task_id, existing
            )

    async with http.acquire():
        outcomes = await asyncio.gather(*(_guarded(p) for p in products))

    succeeded = sum(1 for outcome in outcomes if outcome != "fail")
    failed = [
        product["filename"]
        for product, outcome in zip(products, outcomes, strict=True)
        if outcome == "fail"
    ]
    return succeeded, failed
