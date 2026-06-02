"""Assemble the ``env`` command group (init / add-stage / check)."""

import typer

from noobfriend.cli.env.add_stage import cli_add_stage
from noobfriend.cli.env.check import cli_check
from noobfriend.cli.env.init import cli_init

env_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Initialize, extend, and check the .env configuration file.",
)

env_app.command("init")(cli_init)
env_app.command("add-stage")(cli_add_stage)
env_app.command("check")(cli_check)
