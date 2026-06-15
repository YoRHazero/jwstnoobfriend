"""The ``reduce init`` command: scaffold a starter reduction recipe."""

from pathlib import Path
from typing import Annotated

import typer
from rich.syntax import Syntax

from noobfriend.cli.reduce._io import DEFAULT_RECIPE, resolve_stage
from noobfriend.cli.reduce._recipe import scaffold
from noobfriend.core.console import console


def cli_init(
    stage: Annotated[
        str | None,
        typer.Argument(
            help="Input stage to reduce from. Defaults to [yellow]START_STAGE[/yellow] from .env.",
        ),
    ] = None,
    path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--path",
            help="Recipe file to write.",
        ),
    ] = DEFAULT_RECIPE,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing recipe without prompting."),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("-v", "--show", help="Print the recipe after writing."),
    ] = False,
) -> None:
    """Write a starter recipe listing every reduction step in order, none skipped."""
    resolved_stage = resolve_stage(stage)
    if path.exists() and not force:
        typer.confirm(f"{path} already exists. Overwrite?", abort=True)

    text = scaffold(resolved_stage)
    path.write_text(text)
    console.print(f"[bold green]Wrote recipe to [yellow]{path}[/yellow][/bold green]")
    if show:
        console.print(Syntax(text, "toml", background_color="default"))
