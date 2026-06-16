"""The ``reduce init`` command: scaffold starter reduction recipe(s)."""

from pathlib import Path
from typing import Annotated

import typer
from rich.syntax import Syntax

from noobfriend.cli.reduce._registry import select_pipelines
from noobfriend.core.console import console


def cli_init(
    pupil: Annotated[
        str | None,
        typer.Option(
            "-p",
            "--pupil",
            help="Pipeline to scaffold: [yellow]clear[/yellow] or [yellow]grism[/yellow]. "
            "Omit to scaffold all.",
        ),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option(
            "-s",
            "--stage",
            help="Broad stage to scaffold, e.g. [yellow]2[/yellow]. Omit for all.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing recipes without prompting."),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("-v", "--show", help="Print each recipe after writing."),
    ] = False,
) -> None:
    """Write starter recipe(s) -- one per selected pipeline -- listing every step."""
    pipelines = select_pipelines(pupil, stage)
    if not pipelines:
        console.print(
            f"[red]No pipeline matches pupil={pupil!r} stage={stage!r}.[/red]"
        )
        raise typer.Exit(code=1)

    for pipeline in pipelines:
        path = Path(pipeline.recipe_name)
        if path.exists() and not force:
            typer.confirm(f"{path} already exists. Overwrite?", abort=True)
        text = pipeline.scaffold(pipeline.input_stage)
        path.write_text(text)
        console.print(
            f"[bold green]Wrote recipe to [yellow]{path}[/yellow][/bold green]"
        )
        if show:
            console.print(Syntax(text, "toml", background_color="default"))
