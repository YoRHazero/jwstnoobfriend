"""The ``env add-stage`` command: add per-stage path variables to a ``.env``."""

from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.env._io import read_env_file, upsert_var
from noobfriend.cli.env._options import stage_list_callback
from noobfriend.core.console import console
from noobfriend.core.env import stage_path_var


def stage_dir(data_root: str, stage: str) -> str:
    """Default directory for ``stage``: ``<data_root>/stage<N>/<stage>``.

    The leading digit of ``stage`` groups its variants, e.g. ``2bi`` and ``2bii``
    both live under ``<data_root>/stage2/``.
    """
    return str(Path(data_root) / f"stage{stage[0]}" / stage)


def cli_add_stage(
    stage_list: Annotated[
        list[str],
        typer.Argument(
            help="Stages whose path variables to add, e.g. '1b 2a 2bi'.",
            callback=stage_list_callback,
            metavar="STAGES...",
        ),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "-f",
            "--env-file",
            help="The .env file to extend. Defaults to ./.env.",
            resolve_path=True,
        ),
    ] = Path(".env"),
) -> None:
    """Add per-stage path variables, auto-named under DATA_ROOT_PATH.

    Each STAGE_<STAGE>_PATH is upserted in place (existing values are updated,
    new ones appended), leaving the rest of the file untouched. Running it again
    with the same stages is a no-op.
    """
    if not env_file.exists():
        console.print(
            f"[bold red].env not found at [yellow]{env_file}[/yellow]. Run [cyan]env init[/cyan] first.[/bold red]"
        )
        raise typer.Exit(code=1)

    data_root = read_env_file(env_file).get("DATA_ROOT_PATH")
    if not data_root:
        console.print(
            "[bold red]DATA_ROOT_PATH is not set in the .env file. Run [cyan]env init[/cyan] first.[/bold red]"
        )
        raise typer.Exit(code=1)

    for stage in stage_list:
        upsert_var(env_file, stage_path_var(stage), stage_dir(data_root, stage))

    console.print(
        f"[bold green]Set {len(stage_list)} stage path(s) in [yellow]{env_file}[/yellow].[/bold green]"
    )
