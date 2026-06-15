"""The ``reduce gen`` command: render a runner script from a recipe."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.syntax import Syntax

from noobfriend.cli.reduce._codegen import render
from noobfriend.cli.reduce._io import DEFAULT_RECIPE, validate_stages
from noobfriend.cli.reduce._recipe import load_recipe
from noobfriend.core.console import console
from noobfriend.core.env import get_settings


def cli_gen(
    recipe: Annotated[
        Path,
        typer.Option("-r", "--recipe", help="Recipe TOML to render."),
    ] = DEFAULT_RECIPE,
    path: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--path",
            help="Script to write. Defaults to [yellow]reduce_<stage>.py[/yellow].",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing script without prompting."),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("-v", "--show", help="Print the generated script."),
    ] = False,
) -> None:
    """Generate an editable reduction runner script from a recipe."""
    get_settings()  # load .env so the STAGE_<X>_PATH checks see the configured stages
    try:
        spec = load_recipe(recipe)
        validate_stages(spec)
        source = render(spec)
    except (ValidationError, ValueError, OSError) as error:
        console.print(Panel(str(error), title="reduce gen failed", border_style="red"))
        raise typer.Exit(code=1) from error

    target = path or Path(f"reduce_{spec.select.stage}.py")
    if target.exists() and not force:
        typer.confirm(f"{target} already exists. Overwrite?", abort=True)

    target.write_text(source)
    console.print(
        f"[bold green]Wrote runner to [yellow]{target}[/yellow][/bold green] "
        f"({len(spec.ordered())} steps from {recipe})"
    )
    if show:
        console.print(Syntax(source, "python", background_color="default"))
