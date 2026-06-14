"""The ``box show`` command: summarise a saved NooBox manifest."""

from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.box._io import resolve_box_path
from noobfriend.cli.box._presenters import (
    make_box_listing_table,
    make_box_summary_table,
)
from noobfriend.core.console import console
from noobfriend.navigation import NooBox


def cli_show(
    path: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--path",
            help="Manifest to read. Defaults to [yellow]NOOBOX_PATH[/yellow] from .env.",
            resolve_path=True,
        ),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option(
            "-s",
            "--stage",
            help="Restrict the view to a single stage before showing.",
        ),
    ] = None,
    list_products: Annotated[
        bool,
        typer.Option(
            "-l",
            "--list",
            help="Also print the full per-product table.",
        ),
    ] = False,
) -> None:
    """Show a summary of a saved NooBox manifest."""
    target = resolve_box_path(path)
    if not target.exists():
        raise typer.BadParameter(f"no NooBox manifest at {target}.")

    box = NooBox.load(target)
    if stage is not None:
        box = box.select(stage=stage)

    if len(box) == 0:
        console.print("[yellow]Box is empty.[/yellow]")
        return

    console.print(make_box_summary_table(target, box))
    if list_products:
        console.print(make_box_listing_table(list(box)))
