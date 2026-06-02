"""Assemble the ``fetch`` command group (search / download / manifest)."""

import typer

from noobfriend.cli.fetch.download import register_download_command
from noobfriend.cli.fetch.manifest import register_manifest_commands
from noobfriend.cli.fetch.search import register_search_command

fetch_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Search, download, and inspect JWST data products from MAST.",
)

register_search_command(fetch_app)
register_download_command(fetch_app)
register_manifest_commands(fetch_app)
