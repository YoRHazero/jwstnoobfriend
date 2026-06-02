"""Assemble the ``fetch`` command group (search / download / manifest)."""

import typer

from noobfriend.cli.fetch.download import cli_download
from noobfriend.cli.fetch.manifest import manifest_app
from noobfriend.cli.fetch.search import cli_search

fetch_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Search, download, and inspect JWST data products from MAST.",
)

fetch_app.command("search")(cli_search)
fetch_app.command("download")(cli_download)
fetch_app.add_typer(manifest_app, name="manifest")
