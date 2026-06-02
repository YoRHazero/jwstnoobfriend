"""The ``env check`` command: report which configured paths exist on disk."""

import os
from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.env._io import find_overrides, read_env_file
from noobfriend.cli.env._presenters import render_path_status
from noobfriend.core.console import console
from noobfriend.core.env import env_fields

_SCHEMA_PATH_VARS: frozenset[str] = frozenset(
    field.name for field in env_fields() if field.is_path
)


def _is_path_var(name: str) -> bool:
    """Whether ``name`` denotes a path: a schema path field or a ``*_PATH`` var."""
    upper = name.upper()
    return upper in _SCHEMA_PATH_VARS or upper.endswith("_PATH")


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

    Existing paths are printed in green, missing components in red. Variables
    whose effective value is overridden by the shell environment have their key
    shown in red and are listed in a warning at the end.
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

    console.print(
        f"[bold blue]Checking paths in[/bold blue] [yellow]{env_file}[/yellow]"
    )
    if not path_vars:
        console.print("[dim]No path variables are set.[/dim]")
    else:
        missing: list[Path] = []
        for name, value in path_vars.items():
            path = Path(value).resolve()
            key = (
                f"[red]{name}[/red]"
                if name in overridden
                else f"[bold cyan]{name}[/bold cyan]"
            )
            console.print(f"{key}: {render_path_status(path)}")
            if not path.exists():
                missing.append(path)

        if missing and mkdir:
            console.print("[bold yellow]Creating missing directories...[/bold yellow]")
            for path in missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]Created:[/green] {path}")
                except OSError as error:
                    console.print(
                        f"[bold red]Failed to create {path}:[/bold red] {error}"
                    )
        elif missing:
            console.print(
                f"[yellow]{len(missing)} path(s) missing. Re-run with [cyan]-m/--mkdir[/cyan] to create them.[/yellow]"
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
