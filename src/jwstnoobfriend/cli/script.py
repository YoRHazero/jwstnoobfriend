import typer
from jinja2 import Environment, FileSystemLoader
import os
from typing import Annotated, Literal
from pathlib import Path
from enum import Enum
from jwstnoobfriend.utils.environment import load_environment
from jwstnoobfriend.utils.display import console

load_environment()
LOGGER_NAME_LIST = ["CRDS", "stpipe", "jwst"]

script_app = typer.Typer(
    rich_markup_mode='rich',
    no_args_is_help=True,
    help="Setup configuration files and scripts for different reduction stages.",
)

current_file_path = Path(__file__)
package_root = current_file_path.parent.parent
templates_dir = package_root / "templates"
env = Environment(
    loader=FileSystemLoader(str(templates_dir))
)

def stage_callback(value: str) -> str:
    stage_path = os.getenv(f"STAGE_{value.upper()}_PATH", None)
    # Check if the stage path is defined in the environment variables
    if stage_path is None:
        return typer.BadParameter(
            f"Stage {value} is not defined in the environment variables."
        )
    stage_path = Path(stage_path)

    # Check if the stage path exists
    if not stage_path.exists():
        return typer.BadParameter(
            f"Stage {value} path does not exist: {stage_path}"
        )
    return value.lower()

@script_app.command(
    name="ls",
    help="List available reduction templates.",
)
def list_templates():
    script_templates = [f.name for f in templates_dir.glob("*.py.jinja2")]
    console.print("[bold green]Available reduction templates:[/bold green]")
    for template in script_templates:
        console.print(f"- {template}")
    config_templates = [f.name for f in templates_dir.glob("*.toml.jinja2")]
    console.print("[bold green]Available toml configuration templates:[/bold green]")
    for template in config_templates:
        console.print(f"- {template}")


class FrameType(str, Enum):
    CLEAR = "clear"
    GRISM = "grism"
    ALL = "*"

@script_app.command(
    name="stage2",
    help="Generate reduction scripts and configuration files for reduction steps whose products are at stage 2.",
)
def generate_stage_2(
    stage_output: Annotated[
        str,
        typer.Argument(
            help="The output stage of reduction",
            callback=stage_callback
        )
    ],
    frame_type: Annotated[
        FrameType,
        typer.Option(
            "-t",
            "--frame-type",
            help="The frame type of the input data. Required for certain reduction steps. The available options are 'clear' and 'grism'. If not specified, templates for all frame types will be generated.",
        )
    ] = FrameType.ALL,
    save_path: Annotated[
        Path,
        typer.Option(
            '-s',
            '--save-path',
            help="The folder to save the generated files. If not provided, the files will be saved in the current directory.",
            resolve_path=True
        )
    ] = Path.cwd(),
    name_prefix: Annotated[
        str | None,
        typer.Option(
            '-p',
            '--name-prefix',
            help="The prefix to add to the generated script and config file names.",
        )
    ] = None,
    
    log_record: Annotated[
        bool,
        typer.Option(
            '-n',
            '--no-log-record',
            help="Whether to write log in files.",
        )
    ] = True,
    mute_console_logger: Annotated[
        bool,
        typer.Option(
            '-u',
            '--unmute-console-logger',
            help="Whether to unmute console loggers in the generated script.",
        )
    ] = True,
    process_count: Annotated[
        int,
        typer.Option(
            '-c',
            '--process-count',
            help="The number of processes to use for parallel processing.",
        )
    ] = 1,
):
    # Get available script templates
    file_wildcard = f"{frame_type.lower()}_*_{stage_output}.py.jinja2"
    script_template_paths = list(templates_dir.glob(file_wildcard))
    if len(script_template_paths) == 0:
        console.print(f"[red]No templates found for stage {stage_output} with frame type {frame_type}.[/red]")
    toml_wildcard = f"{frame_type.lower()}_{stage_output}.toml.jinja2"
    toml_template_paths = list(templates_dir.glob(toml_wildcard))
    if len(toml_template_paths) == 0:
        console.print(f"[red]No toml templates found for stage {stage_output}.[/red]")
    config_filename_list = []
    
    for toml_template_path in list(toml_template_paths):
        toml_template = env.get_template(toml_template_path.name)
        
        toml_filename = toml_template_path.name.replace(".toml.jinja2", ".toml")
        if name_prefix:
            toml_filename = f"{name_prefix}_{toml_filename}"
        config_filename_list.append(toml_filename)
        output_path = os.getenv(f"STAGE_{stage_output.upper()}_PATH", "./")
                
        with open(save_path / toml_filename, 'w') as f:
            f.write(
                toml_template.render(
                    output_path=output_path,
                )
            )
        console.print(f"[green]Generated toml configuration file: {toml_filename}[/green]")
    
    for script_template_path in list(script_template_paths):
        script_template = env.get_template(script_template_path.name)
        
        script_filename = script_template_path.name.replace(".py.jinja2", ".py")
        if name_prefix:
            script_filename = f"{name_prefix}_{script_filename}"

        toml_filename_wildcard = f"{name_prefix + '_' if name_prefix else ''}_{frame_type}_{stage_output}.toml"
        ## Todo: Handle multiple toml files
        if len(config_filename_list) != 0:
            toml_filename = config_filename_list[0]
            config_path=save_path / toml_filename
        else:
            config_path=None

        with open(save_path / script_filename, 'w') as f:
            f.write(
                script_template.render(
                    mute_logger_list=LOGGER_NAME_LIST,
                    config_path=config_path,
                    log_record=log_record,
                    process_count=process_count,
                    mute_console=mute_console_logger,
                )
            )
        console.print(f"[green]Generated script: {script_filename} with settings:[/green]")
        console.print(f"  - Mute console logger: {mute_console_logger}")
        console.print(f"  - Log record: {log_record}")
        console.print(f"  - Process count: {process_count}")
        if len(config_filename_list) > 1:
            console.print(f"[red]Warning: Multiple toml configuration files generated. Please confirm the correct one to use in the script.[/red]")

@script_app.command(
    name="config2",
    help="Set up toml configuration files for different reduction step for stage 2 product.",
)
def setup_toml_stage_2(
    stage_output: Annotated[
        str,
        typer.Argument(
            help="The output stage of reduction",
            callback=stage_callback
        )
    ],
    stage_input: Annotated[
        str,
        typer.Option(
            "-i",
            "--input-stage",
            help="The input stage of reduction",
            callback=stage_callback
        )
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option(
            '-p',
            '--config-path',
            help="The path to save the generated toml file. If not provided, the toml file will be saved in the current directory.",
            resolve_path=True
        )
    ] = Path.cwd(),
):
    toml_template = env.get_template(f"reduction_{stage_output}.toml.jinja2")
    output_path = os.getenv(f"STAGE_{stage_output.upper()}_PATH")
    file_name = f"pipeline_setup_{stage_input}_{stage_output}.toml" if stage_input else f"pipeline_setup_{stage_output}.toml"
    with open(config_path / file_name, 'w') as f:
        f.write(
            toml_template.render(
                output_path=output_path,
            )
        )
    
@script_app.command(
    name="config3a",
    help="Set up toml configuration files for step of getting 3a products.",
)
def setup_toml_stage_3a(
    output_path: Annotated[
        Path,
        typer.Option(
            '-p',
            '--save-path',
            help="The path to save the generated toml file. If not provided, the toml file will be saved in the current directory.",
            resolve_path=True
        )
    ] = Path.cwd(),
    stage_3a_path: Annotated[
        Path,
        typer.Option(
            '-a',
            '--stage-3a-path',
            help="The path to the stage 3a products.",
            resolve_path=True
        )
    ] = None,
    stage_2c_path: Annotated[
        Path,
        typer.Option(
            '-c',
            '--stage-2c-path',
            help="The path to the stage 2c products.",
            resolve_path=True
        )
    ] = None
):
    
    toml_template = env.get_template(f"reduction_3a.toml.jinja2")
    if stage_2c_path is None:
        stage_2c_path = os.getenv("STAGE_2C_PATH", None)
        if stage_2c_path is None:
            raise FileNotFoundError("Stage 2c path not found in environment.")
        else:
            stage_2c_path = Path(stage_2c_path)
    if stage_3a_path is None:
        stage_3a_path = os.getenv("STAGE_3A_PATH", None)
        if stage_3a_path is None:
            raise FileNotFoundError("Stage 3a path not found in environment.")
        else:
            stage_3a_path = Path(stage_3a_path)
    with open(output_path / "pipeline_setup_3a.toml", 'w') as f:
        f.write(
            toml_template.render(
                stage_2c_path=stage_2c_path,
                stage_3a_path=stage_3a_path,
            )
        )

if __name__ == "__main__":
    script_app()