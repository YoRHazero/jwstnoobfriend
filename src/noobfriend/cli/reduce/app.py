"""Assemble the ``reduce`` command group (init / gen)."""

import typer

from noobfriend.cli.reduce.gen import cli_gen
from noobfriend.cli.reduce.init import cli_init

reduce_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Generate reduction scripts from a NooBox selection and a recipe.",
)

reduce_app.command("init")(cli_init)
reduce_app.command("gen")(cli_gen)
