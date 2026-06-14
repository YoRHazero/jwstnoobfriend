"""The ``box init`` command: bootstrap a single-stage NooBox manifest."""

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from noobfriend.cli.box._io import resolve_box_path
from noobfriend.cli.box._options import resolve_stage
from noobfriend.cli.box._presenters import make_box_summary_table
from noobfriend.core.console import console
from noobfriend.core.io import RemoteReadError
from noobfriend.navigation import NooBox


def cli_init(
    stage: Annotated[
        str | None,
        typer.Argument(
            help="Stage label assigned to every discovered file. Defaults to [yellow]START_STAGE[/yellow] from .env.",
        ),
    ] = None,
    root: Annotated[
        str | None,
        typer.Option(
            "-r",
            "--root",
            help="Directory to scan: a local path, or a remote 'host:path'. Defaults to [yellow]NOOB_SERVER[/yellow] + [yellow]STAGE_<STAGE>_PATH[/yellow] from .env.",
            rich_help_panel="Source",
            metavar="[HOST:]PATH",
        ),
    ] = None,
    wildcard: Annotated[
        str,
        typer.Option(
            "-w",
            "--wildcard",
            help="Glob selecting product files in the directory.",
            rich_help_panel="Source",
        ),
    ] = "*.fits",
    probe: Annotated[
        bool,
        typer.Option(
            "-p",
            "--probe",
            help="Read each file's header now to fill pupil / filter / shape / footprint (slower). Otherwise books stay thin.",
            rich_help_panel="Source",
        ),
    ] = False,
    path: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--path",
            help="Manifest destination. Defaults to [yellow]NOOBOX_PATH[/yellow] from .env.",
            resolve_path=True,
            rich_help_panel="Output",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing manifest without prompting.",
            rich_help_panel="Output",
        ),
    ] = False,
    show: Annotated[
        bool,
        typer.Option(
            "-v",
            "--show",
            help="Print the resulting box summary after writing.",
            rich_help_panel="Output",
        ),
    ] = False,
) -> None:
    """Initialize a single-stage NooBox from a directory of products."""
    resolved_stage = resolve_stage(stage)
    target = resolve_box_path(path)

    if target.exists() and not force:
        typer.confirm(f"{target} already exists. Overwrite?", abort=True)

    console.print(f"[dim]Scanning for stage {resolved_stage} products ...[/dim]")
    try:
        box = NooBox.from_directory(
            resolved_stage, root, wildcard=wildcard, probe=probe
        )
    except (ValueError, RemoteReadError) as error:
        console.print(Panel(str(error), title="Discovery failed", border_style="red"))
        raise typer.Exit(code=1) from error

    if len(box) == 0:
        console.print(
            f"[yellow]No files matched {wildcard!r} for stage {resolved_stage}; "
            "nothing written.[/yellow]"
        )
        raise typer.Exit(code=1)

    box.save(target)
    console.print(
        f"[bold green]Wrote NooBox ({len(box)} books @ {resolved_stage}) to "
        f"[yellow]{target}[/yellow][/bold green]"
    )
    if show:
        console.print(make_box_summary_table(target, box))
