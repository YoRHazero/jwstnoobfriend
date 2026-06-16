"""The ``reduce gen`` command: render runner script(s) from existing recipe(s)."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.syntax import Syntax

from noobfriend.cli.reduce._io import validate_stages
from noobfriend.cli.reduce._recipe import load_recipe
from noobfriend.cli.reduce._registry import select_pipelines
from noobfriend.core.console import console
from noobfriend.core.env import get_settings


def cli_gen(
    pupil: Annotated[
        str | None,
        typer.Option(
            "-p",
            "--pupil",
            help="Pipeline to render: [yellow]clear[/yellow] or [yellow]grism[/yellow]. "
            "Omit to render every existing recipe.",
        ),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option(
            "-s", "--stage", help="Broad stage to render, e.g. [yellow]2[/yellow]."
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing scripts without prompting."),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("-v", "--show", help="Print each generated script."),
    ] = False,
) -> None:
    """Generate editable runner script(s) from the matching recipe file(s)."""
    get_settings()  # load .env so the STAGE_<X>_PATH checks see the configured stages
    pipelines = select_pipelines(pupil, stage)
    if not pipelines:
        console.print(
            f"[red]No pipeline matches pupil={pupil!r} stage={stage!r}.[/red]"
        )
        raise typer.Exit(code=1)

    explicit = pupil is not None or stage is not None
    generated = 0
    for pipeline in pipelines:
        recipe_path = Path(pipeline.recipe_name)
        if not recipe_path.exists():
            if explicit:
                console.print(
                    f"[red]Recipe {recipe_path} not found; run "
                    f"`noobfriend reduce init -p {pipeline.pupil} -s {pipeline.stage}` first.[/red]"
                )
                raise typer.Exit(code=1)
            continue  # all-mode: silently skip pipelines with no recipe yet
        try:
            recipe = load_recipe(recipe_path)
            validate_stages(recipe)
            source = pipeline.render(recipe)
        except (ValidationError, ValueError, OSError) as error:
            console.print(
                Panel(str(error), title=f"{recipe_path} failed", border_style="red")
            )
            raise typer.Exit(code=1) from error

        target = Path(pipeline.script_name)
        if target.exists() and not force:
            typer.confirm(f"{target} already exists. Overwrite?", abort=True)
        target.write_text(source)
        console.print(
            f"[bold green]Wrote runner to [yellow]{target}[/yellow][/bold green] "
            f"(from {recipe_path})"
        )
        if show:
            console.print(Syntax(source, "python", background_color="default"))
        generated += 1

    if generated == 0 and not explicit:
        console.print(
            "[yellow]No recipes found; run `noobfriend reduce init` first.[/yellow]"
        )
