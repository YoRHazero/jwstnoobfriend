"""Assemble the ``box`` command group (init / show)."""

import typer

from noobfriend.cli.box.init import cli_init
from noobfriend.cli.box.show import cli_show

box_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Build and inspect the NooBox product index.",
)

box_app.command("init")(cli_init)
box_app.command("show")(cli_show)
