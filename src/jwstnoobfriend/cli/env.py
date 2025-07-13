import typer
from typing import Annotated
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import os
import re

from jwstnoobfriend.utils.log import getLogger
from jwstnoobfriend.utils.environment import load_environment
from jwstnoobfriend.utils.display import console

logger = getLogger(__name__)

current_file_path = Path(__file__)
package_root = current_file_path.parent.parent  # from cli/ to jwstnoobfriend/
templates_dir = package_root / "templates"  # from jwstnoobfriend/ to templates/
env = Environment(loader=FileSystemLoader(str(templates_dir)))

env_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Commands for managing the .env file",
)


def start_stage_callback(value: str | None) -> str:
    """
    Callback to validate the start_stage option.
    The first character must be a digit.
    """
    if value and not value[0].isdigit():
        raise typer.BadParameter("The first character of start_stage must be a digit.")
    return value


@env_app.command(name="init", help="Initialize the .env file")
def init_env(
    start_stage: Annotated[
        str,
        typer.Option(
            "-s",
            "--start-stage",
            help="Stage to start the reduction from, e.g., '1b'. The 1st character [red]MUST[/red] be a number.",
            callback=start_stage_callback,
            rich_help_panel="Setup",
        ),
    ] = None,
    crds_path: Annotated[
        Path,
        typer.Option(
            "-c",
            "--crds-path",
            help="Path to the CRDS cache directory.",
            resolve_path=True,
            rich_help_panel="Setup",
        ),
    ] = None,
    data_root_path: Annotated[
        Path,
        typer.Option(
            "-d",
            "--data-root-path",
            help="Path to the root directory of the data.",
            resolve_path=True,
            rich_help_panel="Setup",
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-path",
            help="Parent path for the [purple].env[/purple] file. Defaults to the current working directory.",
            resolve_path=True,
            rich_help_panel="Output",
        ),
    ] = Path.cwd(),
    show_content: Annotated[
        bool,
        typer.Option(
            "-v",
            "--show-content",
            help="Show the content of the .env file after initialization.",
            is_flag=True,
            rich_help_panel="Output",
        ),
    ] = False,
):
    init_template = env.get_template("reduction_init.env.jinja2")
    if show_content:
        console.print("[bold green]Initializing .env file...[/bold green]\n")
        console.print(
            init_template.render(
                start_stage=start_stage.lower() if start_stage else "",
                crds_path=crds_path if crds_path else "",
                data_root_path=data_root_path if data_root_path else "",
            )
        )
    with open(output_path / ".env", "w") as f:
        f.write(
            init_template.render(
                start_stage=start_stage.lower() if start_stage else "",
                crds_path=crds_path if crds_path else "",
                data_root_path=data_root_path if data_root_path else "",
            )
        )
    console.print(
        f"[bold green]The .env file has been written to [yellow]{output_path / '.env'}[/yellow][/bold green]"
    )


def stage_list_callback(value_list: list[str]) -> list[str]:
    """
    Callback to validate the stage_list option.
    Ensures that the first character of each stage is a digit.
    """
    load_environment()

    # If the start_stage is not given, add it to the beginning of the list
    start_stage = os.getenv("START_STAGE", None)
    if start_stage is None or start_stage[0] == "#":
        raise typer.BadParameter(
            "The START_STAGE environment variable is not set in the .env file."
        )
    if start_stage not in value_list:
        value_list.insert(0, start_stage)
    for value in value_list:
        if not value[0].isdigit():
            raise typer.BadParameter(
                f"The first character of stage '{value}' must be a digit."
            )

    return [
        value.lower() for value in value_list
    ]  # Convert to lowercase for consistency


@env_app.command(name="append", help="Append folder information to the .env file")
def append_env(
    stage_list: Annotated[
        list[str],
        typer.Argument(
            help="Stage to append to the .env file. Can be specified multiple times.",
            callback=stage_list_callback,
            metavar="LIST",
        ),
    ],
    auto_name: Annotated[
        bool,
        typer.Option(
            "-n",
            "--auto-name",
            help="Automatically name the paths based on the stage names.",
            is_flag=True,
        ),
    ] = False,
    append_at_end: Annotated[
        bool,
        typer.Option(
            "-a",
            "--append-at-end",
            help="Append the new paths to the end of the .env file.",
            is_flag=True,
        ),
    ] = False,
):
    init_template = env.get_template("reduction_init.env.jinja2")
    append_template = env.get_template("reduction_storage.env.jinja2")
    init_keys = []
    for line in init_template.render().splitlines():
        re_match = re.match(r"(\w+)\s*=", line)
        if re_match:
            init_keys.append(re_match.group(1))
    init_render_dict = {}
    ## Check if the .env file exists and all required keys are set
    for key in init_keys:
        env_var = os.getenv(key, None)
        if env_var is not None and env_var[0] != "#":
            init_render_dict[key.lower()] = env_var
        else:
            console.print(
                f"[bold red]{key}[/bold red] is not set in the .env file. Please initialize the .env file first."
            )
            raise typer.Exit(code=1)

    ## use the data_root_path to generate the stage paths
    data_root_path = Path(os.environ["DATA_ROOT_PATH"])
    append_render_dict = {}
    for stage in stage_list:
        if auto_name:
            stage_path = data_root_path / f"stage{stage[0]}" / stage
            append_render_dict[stage] = stage_path
        else:
            stage_path = ""
            append_render_dict[stage] = stage_path

    # Append mode
    if append_at_end:
        append_render_dict.pop(
            os.environ["START_STAGE"], None
        )  # Remove the start stage if it exists
        with open(".env", "a") as f:
            f.write(append_template.render(stage_folder_paths=append_render_dict))
            console.print(
                "[bold green]The .env file has been appended with the new stage paths.[/bold green]"
            )
    # Replace mode
    else:
        rendered_content = (
            init_template.render(**init_render_dict)
            + "\n"
            + append_template.render(
                replace=True,
                file_box_path=data_root_path / "noobox.json",
                stage_folder_paths=append_render_dict,
            )
        )
        with open(".env", "w") as f:
            f.write(rendered_content)
            console.print(
                "[bold green]The .env file has been updated with the new stage paths.[/bold green]"
            )


def check_directory_exists(path: Path):
    path = path.resolve()
    if path.exists():
        return f"[green]{path}[/green]"
    parts = path.parts
    for i in range(1, len(parts) + 1):
        sub_path = Path(*parts[:i])
        if not sub_path.exists():
            break
    return f"[green]/{'/'.join(parts[1 : i - 1])}[/green]/[red]{'/'.join(parts[i - 1 :])}[/red]"


@env_app.command(
    name="check",
    help="Check whether the file paths in the .env file exist. The existing paths will be printed in green, and the non-existing paths will be printed in red.",
)
def check_env(
    mkdir: Annotated[
        bool,
        typer.Option(
            "-m",
            "--mkdir",
            help="Create the directories if they do not exist.",
            is_flag=True,
        ),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            "-f",
            "--env-file",
            help="Path to the .env file to check. Defaults to the current working directory.",
            resolve_path=True,
        ),
    ] = Path(".env"),
):
    if not env_file.exists():
        console.print(
            f"[bold red]The .env file does not exist at [yellow]{env_file}[/yellow]. Please initialize the .env file first.[/bold red]"
        )
        raise typer.Exit(code=1)

    path_vars = {}
    with open(env_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("#"):
                console.print(f"[bold blue]{key}[/bold blue]: Not set!")
                continue

            # Remove comments from the value
            if "#" in value:
                value = value.split("#", 1)[0].strip()

            if "PATH" in key.upper():
                path_vars[key] = Path(value)

    console.print(
        f"[bold blue]Checking Paths in[/bold blue] [bold yellow]{env_file}...[/bold yellow]"
    )

    paths_need_to_create = []

    for var_name, path in path_vars.items():
        console.print(f"[bold cyan]{var_name}[/bold cyan]:", end=" ")
        is_file = "." in path.name and not path.name.startswith(
            "."
        )  # Check if the path is a file
        if is_file:
            path = path.parent

        path_status_string = check_directory_exists(path)
        console.print(path_status_string)
        if not path.exists():
            paths_need_to_create.append(path)

    if mkdir and paths_need_to_create:
        console.print("[bold yellow]Creating directories...[/bold yellow]")
        for path in paths_need_to_create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                console.print(f"[bold green]Created:[/bold green] {path}")
            except Exception as e:
                console.print(f"[bold red]Failed to create {path}:[/bold red] {e}")


if __name__ == "__main__":
    env_app()
