"""The ``env check`` command: report which configured paths exist on disk."""

import os
import posixpath
from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.env._io import find_overrides, read_env_file
from noobfriend.cli.env._presenters import render_path_status
from noobfriend.core.console import console
from noobfriend.core.env import EnvGroup, EnvPathKind, env_fields
from noobfriend.core.io import RemoteReadError, remote_exists, remote_makedirs

_FIELDS = env_fields()
_SCHEMA_PATH_VARS: frozenset[str] = frozenset(
    field.name for field in _FIELDS if field.is_path
)
_SCHEMA_GROUPS: dict[str, EnvGroup] = {field.name: field.group for field in _FIELDS}
_SCHEMA_PATH_KINDS: dict[str, EnvPathKind | None] = {
    field.name: field.path_kind for field in _FIELDS
}

#: ``NOOB_SERVER`` values that denote the local machine rather than an ssh host.
_LOCAL_SERVER_ALIASES: frozenset[str] = frozenset({"", "localhost", "local"})


def _is_path_var(name: str) -> bool:
    """Whether ``name`` denotes a path: a schema path field or a ``*_PATH`` var."""
    upper = name.upper()
    return upper in _SCHEMA_PATH_VARS or upper.endswith("_PATH")


def _group_of(name: str) -> EnvGroup:
    """Return the configuration group ``name`` belongs to.

    Schema fields use their declared group; any other ``*_PATH`` variable (the
    dynamic ``STAGE_<STAGE>_PATH`` family) defaults to
    :attr:`~noobfriend.core.env.EnvGroup.storage`, i.e. it lives wherever
    ``NOOB_SERVER`` points.
    """
    return _SCHEMA_GROUPS.get(name.upper(), EnvGroup.storage)


def _path_kind_of(name: str) -> EnvPathKind:
    """Return whether ``name`` identifies a file or directory path.

    Dynamic ``*_PATH`` variables are stage/data directories by convention.
    """
    return _SCHEMA_PATH_KINDS.get(name.upper()) or EnvPathKind.directory


def _local_mkdir_path(name: str, path: Path) -> Path:
    """Return the local directory to create for ``name``'s configured path."""
    return path.parent if _path_kind_of(name) is EnvPathKind.file else path


def _remote_mkdir_path(name: str, path: str) -> str:
    """Return the remote directory to create for ``name``'s configured path."""
    if _path_kind_of(name) is not EnvPathKind.file:
        return path
    parent = posixpath.dirname(path.rstrip("/"))
    return parent or "."


def _resolve_server(raw: str | None) -> str | None:
    """Normalise a ``NOOB_SERVER`` value to an ssh host, or ``None`` when local.

    ``None``, the empty string and the ``localhost`` / ``local`` aliases all mean
    the local machine and yield ``None``; any other value is returned stripped,
    ready to use as the ``ssh`` destination.

    Parameters
    ----------
    raw : str or None
        The effective ``NOOB_SERVER`` value.

    Returns
    -------
    str or None
        The remote host, or ``None`` when the configuration points at the local
        machine.
    """
    if raw is None:
        return None
    host = raw.strip()
    return None if host.lower() in _LOCAL_SERVER_ALIASES else host


def _is_remote_path(name: str, server_host: str | None) -> bool:
    """Whether ``name``'s path should be checked on ``server_host``.

    A path is remote only when a remote ``NOOB_SERVER`` is configured *and* the
    variable belongs to the ``storage`` group; every other group stays local.
    """
    return server_host is not None and _group_of(name) is EnvGroup.storage


def cli_check(
    mkdir: Annotated[
        bool,
        typer.Option(
            "-m",
            "--mkdir",
            help="Create the directories that do not yet exist.",
        ),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            "-f",
            "--env-file",
            help="The .env file to check. Defaults to ./.env.",
            resolve_path=True,
        ),
    ] = Path(".env"),
) -> None:
    """Check whether the path variables in the .env file exist.

    Existing paths are printed in green, missing components in red. ``storage``
    paths are checked on ``NOOB_SERVER`` over ``ssh`` when it names a remote host
    (so the data directories are verified where they actually live); every other
    path is checked on the local filesystem. Variables whose effective value is
    overridden by the shell environment have their key shown in red and are
    listed in a warning at the end.
    """
    if not env_file.exists():
        console.print(
            f"[bold red].env not found at [yellow]{env_file}[/yellow]. Run [cyan]env init[/cyan] first.[/bold red]"
        )
        raise typer.Exit(code=1)

    shell_env = dict(os.environ)  # captured before any .env layer is applied
    declared = read_env_file(env_file)
    overrides = find_overrides(declared, shell_env)
    overridden = {name for name, _, _ in overrides}

    # Effective value: the shell wins over the file (see load_environment).
    path_vars = {
        name: shell_env.get(name, value)
        for name, value in declared.items()
        if value and _is_path_var(name)
    }
    server_host = _resolve_server(
        shell_env.get("NOOB_SERVER", declared.get("NOOB_SERVER"))
    )

    console.print(
        f"[bold blue]Checking paths in[/bold blue] [yellow]{env_file}[/yellow]"
    )
    if not path_vars:
        console.print("[dim]No path variables are set.[/dim]")
    else:
        if server_host is not None:
            console.print(
                f"[dim]Storage paths checked on [cyan]{server_host}[/cyan] (NOOB_SERVER).[/dim]"
            )
        missing_local: list[tuple[str, Path]] = []
        missing_remote: list[tuple[str, str, str]] = []
        for name, value in path_vars.items():
            key = (
                f"[red]{name}[/red]"
                if name in overridden
                else f"[bold cyan]{name}[/bold cyan]"
            )
            if _is_remote_path(name, server_host):
                assert server_host is not None
                try:
                    exists = remote_exists(server_host, value)
                except RemoteReadError as error:
                    console.print(
                        f"{key}: [cyan]{server_host}[/cyan]:[yellow]{value}[/yellow] "
                        f"[bold red](remote check failed: {error})[/bold red]"
                    )
                    continue
                status = "[green]exists[/green]" if exists else "[red]missing[/red]"
                console.print(f"{key}: [cyan]{server_host}[/cyan]:{value} {status}")
                if not exists:
                    missing_remote.append((name, server_host, value))
            else:
                path = Path(value).resolve()
                console.print(f"{key}: {render_path_status(path)}")
                if not path.exists():
                    missing_local.append((name, path))

        total_missing = len(missing_local) + len(missing_remote)
        if total_missing and mkdir:
            console.print("[bold yellow]Creating missing directories...[/bold yellow]")
            for name, local_path in missing_local:
                mkdir_path = _local_mkdir_path(name, local_path)
                try:
                    mkdir_path.mkdir(parents=True, exist_ok=True)
                    label = "Created parent" if mkdir_path != local_path else "Created"
                    console.print(f"[green]{label}:[/green] {mkdir_path}")
                except OSError as error:
                    console.print(
                        f"[bold red]Failed to create {mkdir_path}:[/bold red] {error}"
                    )
            for name, host, remote_path in missing_remote:
                mkdir_path = _remote_mkdir_path(name, remote_path)
                try:
                    remote_makedirs(host, mkdir_path)
                    label = "Created parent" if mkdir_path != remote_path else "Created"
                    console.print(f"[green]{label}:[/green] {host}:{mkdir_path}")
                except RemoteReadError as error:
                    console.print(
                        f"[bold red]Failed to create {host}:{mkdir_path}:[/bold red] {error}"
                    )
        elif total_missing:
            console.print(
                f"[yellow]{total_missing} path(s) missing. Re-run with [cyan]-m/--mkdir[/cyan] to create them.[/yellow]"
            )

    if overrides:
        console.print(
            f"\n[bold red]⚠ {len(overrides)} value(s) overridden by the shell environment:[/bold red]"
        )
        for name, declared_value, shell_value in overrides:
            console.print(
                f"  [red]{name}[/red]: .env has [yellow]{declared_value}[/yellow], "
                f"shell sets [yellow]{shell_value}[/yellow] (in effect)"
            )
