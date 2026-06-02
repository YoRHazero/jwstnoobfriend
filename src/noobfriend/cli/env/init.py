"""The ``env init`` command: write a fresh ``.env`` from the schema."""

from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.env._io import write_env_file
from noobfriend.cli.env._options import start_stage_callback
from noobfriend.cli.env._render import render_base
from noobfriend.core.console import console


def cli_init(
    start_stage: Annotated[
        str | None,
        typer.Option(
            "-s",
            "--start-stage",
            help="Stage to start the reduction from, e.g. '1b'. The 1st character [red]MUST[/red] be a digit.",
            callback=start_stage_callback,
            rich_help_panel="Setup",
        ),
    ] = None,
    crds_path: Annotated[
        Path | None,
        typer.Option(
            "-c",
            "--crds-path",
            help="Path to the CRDS cache directory.",
            resolve_path=True,
            rich_help_panel="Setup",
        ),
    ] = None,
    data_root_path: Annotated[
        Path | None,
        typer.Option(
            "-d",
            "--data-root-path",
            help="Root directory of the data; base for auto-named stage dirs.",
            resolve_path=True,
            rich_help_panel="Storage",
        ),
    ] = None,
    noob_server: Annotated[
        str | None,
        typer.Option(
            "-S",
            "--noob-server",
            help="Default download host (an ~/.ssh/config alias); 'localhost' means local.",
            rich_help_panel="Remote",
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-path",
            help="Parent directory for the [purple].env[/purple] file. Defaults to the current directory.",
            resolve_path=True,
            rich_help_panel="Output",
        ),
    ] = Path.cwd(),
    show_content: Annotated[
        bool,
        typer.Option(
            "-v",
            "--show-content",
            help="Print the rendered .env content after writing it.",
            rich_help_panel="Output",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing .env without prompting.",
            rich_help_panel="Output",
        ),
    ] = False,
) -> None:
    """Initialize the .env file with the base configuration variables."""
    env_file = output_path / ".env"
    if env_file.exists() and not force:
        typer.confirm(f"{env_file} already exists. Overwrite?", abort=True)

    values: dict[str, str | None] = {
        "START_STAGE": start_stage,
        "CRDS_PATH": str(crds_path) if crds_path else None,
        "DATA_ROOT_PATH": str(data_root_path) if data_root_path else None,
        "NOOB_SERVER": noob_server,
    }
    content = render_base(values)
    write_env_file(env_file, content)
    console.print(f"[bold green]Wrote .env to [yellow]{env_file}[/yellow][/bold green]")
    if show_content:
        console.print(content)
