"""Root Typer application wiring together the noobfriend CLI command groups."""

from typing import Annotated

import typer

from noobfriend.cli.box import box_app
from noobfriend.cli.env import env_app
from noobfriend.cli.fetch import fetch_app
from noobfriend.cli.reduce import reduce_app
from noobfriend.core.log import setup_logging

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="noobfriend command-line tools.",
)

app.add_typer(env_app, name="env")
app.add_typer(fetch_app, name="fetch")
app.add_typer(box_app, name="box")
app.add_typer(reduce_app, name="reduce")


@app.callback()
def configure(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level for the noobfriend logger (DEBUG, INFO, WARNING, ...).",
        ),
    ] = "INFO",
) -> None:
    """Configure logging for the whole CLI invocation."""
    setup_logging(log_level)


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()
