from pydantic import validate_call, ConfigDict
import typer
from typing import Annotated, Callable, Iterable, Literal
from pathlib import Path
from collections import Counter
from jwstnoobfriend.utils.display import console, time_footer
from jwstnoobfriend.utils.log import getLogger
from astroquery.mast.missions import MastMissionsClass
from rich.table import Table
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import asyncio
import asyncer
from jwstnoobfriend.utils.network import ConnectionSession
from jwstnoobfriend.utils.environment import load_environment
import json
import sys
import os


## Initialize the logger
logger = getLogger(__name__)

retrieve_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Check and retrieve JWST data from MAST.",
)

mission = MastMissionsClass(mission="jwst")


def util_first_contact(
    program_query: Iterable, contact_key: str = "instrume"
) -> dict[str, int]:
    """Utility function to find the first contact index for each instrument in the program query."""
    contact = {}
    for i, row in enumerate(program_query):
        if row[contact_key] in contact:
            continue
        contact[str(row[contact_key])] = i
    return contact


def product_level_callback(value: str) -> str:
    """Callback function to validate product level input."""
    choices = ["1b", "2a", "2b", "2c"]
    if value not in choices:
        raise typer.BadParameter(
            f"Invalid product level '{value}'. Choose from {choices}."
        )
    return value


### Check command
@retrieve_app.command(
    name="check",
    help="Old version of the check command. In this version, the retrieval is done by astroquery. And for the speed, we assume every dataset of the same instrument has the same suffix, which may not be true in some cases. Use [cyan]retrieve[/cyan] command instead.",
)
@time_footer
def cli_retrieve_check(
    proposal_id: Annotated[
        str, typer.Argument(help="Proposal ID to check, 5 digits, e.g. '01895'.")
    ],
    product_level: Annotated[
        str,
        typer.Option(
            "-l",
            "--product-level",
            callback=product_level_callback,
            help="Product stage to check. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
        ),
    ] = "1b",
    show_example: Annotated[
        bool,
        typer.Option(
            "-s",
            "--show-example",
            help="Show first 5 products of output, default is False.",
        ),
    ] = False,
    show_suffix: Annotated[
        bool,
        typer.Option(
            "-d",
            "--show-suffix",
            help="Show the suffix list of the products for each instrument, default is False.",
        ),
    ] = False,
    include_rateint: Annotated[
        bool,
        typer.Option(
            "-r",
            "--include-rateint",
            help="Include rateint products in the output, default is False, in most cases rateint products are just the same as rate products.",
        ),
    ] = False,
):
    global console
    ## Format of the table of accessibility
    table_access = Table(title=f"Accessibility of {proposal_id}")
    table_access.add_column("Accessibility", justify="left", style="cyan", width=20)
    table_access.add_column("number of dataset", justify="right", style="green")

    ## Format of the table of instruments
    table_instrument = Table(
        title=f"Instrument file numbers of {proposal_id} ({product_level})"
    )
    table_instrument.add_column("Instrument ", justify="left", style="cyan", width=20)
    table_instrument.add_column("number of files", justify="right", style="green")

    ## Get the file set ID for the proposal and product level
    program_query = mission.query_criteria(  # type: ignore
        program=proposal_id, productLevel=product_level
    )

    if len(program_query) == 0:
        console.print(
            f"[red]No data found for proposal ID {proposal_id} with product level {product_level}.[/red]"
        )
        raise typer.Exit(code=1)

    instrument_contact = util_first_contact(program_query)
    products_suffix_dict = {}
    ## Filter the products based on the product level
    for instrument, index in instrument_contact.items():
        products = mission.get_product_list(  # type: ignore
            program_query["fileSetName"][index],
        )
        selected_products = mission.filter_products(products, category=product_level)
        products_suffix_dict[instrument] = [
            p["filename"].split(p["dataset"])[-1] for p in selected_products
        ]

        if not include_rateint:
            products_suffix_dict[instrument] = [
                suffix
                for suffix in products_suffix_dict[instrument]
                if "rateint" not in suffix
            ]

        if show_suffix:
            console.print(f"[yellow]Suffixes for {instrument}:[/yellow]")
            console.print(products_suffix_dict[instrument])

    access_counts = Counter(program_query["access"])
    for access, count in access_counts.items():
        table_access.add_row(access, str(count))
    console.print(table_access)

    fileset_counts = Counter(program_query["instrume"])
    file_counts = {
        instrument: len(products_suffix_dict[instrument]) * fileset_counts[instrument]
        for instrument in products_suffix_dict
    }
    for instrument, count in file_counts.items():
        table_instrument.add_row(str(instrument), str(count))
    console.print(table_instrument)

    if show_example:
        table_example = Table(title="Example Products")
        table_example.add_column(
            "Product File Name", justify="left", style="cyan", width=10, no_wrap=False
        )
        table_example.add_column("Accessibility", justify="left", style="red")
        table_example.add_column("Instrument", justify="left", style="green")
        for product in selected_products[:5]:
            table_example.add_row(
                product["product_key"], product["access"], product["instrument"]
            )


### Retrieve command

mast_jwst_base_url = "https://mast.stsci.edu/search/jwst/api/v0.1"
mast_jwst_search_url = f"{mast_jwst_base_url}/search"
mast_jwst_product_url = f"{mast_jwst_base_url}/list_products"
mast_jwst_post_product_url = f"{mast_jwst_base_url}/post_list_products"


async def search_proposal_id(
    proposal_id: str,
    product_level: str,
):
    """Search for the file sets of a given proposal ID and product level."""
    async with ConnectionSession.session() as session:
        search_json = await ConnectionSession.fetch_json_async(
            mast_jwst_search_url,
            session,
            method="POST",
            body={
                "conditions": [{"program": proposal_id, "productLevel": product_level}]
            },
        )
        return search_json["results"]


async def send_products_request(
    fileset_name: str,
    product_level: str,
    include_rateint: bool,
):
    """Send a request to get the products for a given fileset name and product level."""
    RETRY_LIMIT = 3
    async with ConnectionSession.session() as session:
        for _ in range(RETRY_LIMIT):
            product_json = await ConnectionSession.fetch_json_async(
                mast_jwst_product_url,
                session,
                method="GET",
                params={"dataset_ids": fileset_name},
            )
            products = product_json["products"]
            if len(products) > 0:
                break
            else:
                await asyncio.sleep(1)
        if not include_rateint:
            products = [p for p in products if "rateint" not in p["file_suffix"]]
        products_filtered = [p for p in products if p["category"] == product_level]
        return products_filtered


async def get_products(
    search_results: Iterable[dict],
    product_level: str,
    include_rateint: bool,
):
    """wrapping the send_products_request to get the products for each fileset for runnable in asyncer."""
    tasks = []
    search_results = [
        result for result in search_results if result["access"] != "private"
    ]  # Filter out private access filesets
    async with asyncer.create_task_group() as task_group:
        for result in search_results:
            fileset_name = result["fileSetName"]
            soon_products = task_group.soonify(send_products_request)(
                fileset_name=fileset_name,
                product_level=product_level,
                include_rateint=include_rateint,
            )
            tasks.append(soon_products)
    error_table = Table(title="No products found or an error occurred.")
    error_table.add_column("Fileset", justify="left", style="red", width=30)
    results = []
    for i, task in enumerate(tasks):
        products = task.value
        if not products and error_table is not None:
            error_table.add_row(
                search_results[i]["fileSetName"],
            )
        results.extend(products)
    if error_table.row_count > 0:
        console.print(error_table)
        console.print("This problem can be solved by rerunning the command.")
    return results


async def get_products_combined_request(
    search_results: Iterable[dict],
    product_level: str,
    include_rateint: bool,
):
    """This function is potentially used for chunking the products in a single request, instead of sending one request for only one fileset."""
    async with ConnectionSession.session() as session:
        product_json = await ConnectionSession.fetch_json_async(
            mast_jwst_post_product_url,
            session,
            method="POST",
            body={"dataset_ids": [result["fileSetName"] for result in search_results]},
        )
        products = product_json["products"]
        if not include_rateint:
            products = [p for p in products if "rateint" not in p["file_suffix"]]
        products_filtered = [p for p in products if p["category"] == product_level]
        return products_filtered


@retrieve_app.command(
    name="retrieve", help="Check the JWST data of given proposal id from MAST."
)
@time_footer
def cli_retrieve_check_async(
    proposal_id: Annotated[
        str, typer.Argument(help="Proposal ID to check, 5 digits, e.g. '01895'.")
    ],
    product_level: Annotated[
        str,
        typer.Option(
            "-l",
            "--product-level",
            callback=product_level_callback,
            help="Product stage to check. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
        ),
    ] = "1b",
    show_example: Annotated[
        bool,
        typer.Option(
            "-s",
            "--show-example",
            help="Show first 5 products of output, default is False.",
        ),
    ] = False,
    include_rateint: Annotated[
        bool,
        typer.Option(
            "-r",
            "--include-rateint",
            help="Include rateint products in the output, default is False, in most cases rateint products are just the same as rate products.",
        ),
    ] = False,
    output_file: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-file",
            help="File to save the products, default is 'products.json' in the current directory.",
            rich_help_panel="Output",
            metavar="FILE",
            prompt="Output file is not specified. Use the default path (press [Enter] to confirm or type a new path):\n ",
            prompt_required=False,
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd() / "products.json",
):
    global console
    ## Retrieval part
    search_filesets = asyncer.runnify(search_proposal_id)(
        proposal_id=proposal_id,
        product_level=product_level,
    )

    ## Show the summary of the search results
    fileset_access = Counter([fs["access"] for fs in search_filesets])
    table_access = Table(title=f"Accessibility of {proposal_id}")
    table_access.add_column("Accessibility", justify="left", style="cyan", width=20)
    table_access.add_column("number of dataset", justify="right", style="green")
    for access, count in fileset_access.items():
        table_access.add_row(access, str(count))
    console.print(table_access)

    results = asyncer.runnify(get_products)(
        search_results=search_filesets,
        product_level=product_level,
        include_rateint=include_rateint,
    )

    ## Show the summary of the instruments
    products_instrument = Counter([p["instrument_name"] for p in results])
    table_instrument = Table(
        title=f"Instrument file numbers of {proposal_id} ({product_level})"
    )
    table_instrument.add_column("Instrument ", justify="left", style="cyan", width=20)
    table_instrument.add_column("number of files", justify="right", style="green")
    for instrument, count in products_instrument.items():
        table_instrument.add_row(str(instrument), str(count))
    console.print(table_instrument)

    ## Show the example products
    if show_example:
        table_example = Table(title="Example Products")
        table_example.add_column(
            "Product File Name", justify="left", style="cyan", width=10, no_wrap=False
        )
        table_example.add_column("Accessibility", justify="left", style="red")
        table_example.add_column("Instrument", justify="left", style="green")
        table_example.add_column("File Size", justify="left", style="magenta")
        for product in results[:5]:
            table_example.add_row(
                product["filename"],
                product["access"],
                product["instrument_name"],
                f"{product['size'] / (1024 * 1024):.2f} MB"
                if product["size"]
                else "N/A",
            )
        console.print(table_example)

    ## Save the results to file, if output_file is specified
    output_option_used = "-o" in sys.argv or "--output-file" in sys.argv
    if output_option_used and output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[teal]Opening {output_file} [/teal]...")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            console.print("[green] Saved [/green]")


### Download command
def env_download_folder():
    ## Load environment variables from .env
    load_environment()
    START_STAGE = os.getenv("START_STAGE", None)
    if START_STAGE is None:
        return None
    START_STAGE_PATH = os.environ.get(f"STAGE_{START_STAGE.upper()}_PATH", None)
    if START_STAGE_PATH is None:
        logger.warning(
            f"START_STAGE is set to {START_STAGE}, but STAGE_{START_STAGE.upper()}_PATH is not set. Check whether the .env file is written correctly."
        )
        return None
    if not Path(START_STAGE_PATH).exists():
        Path(START_STAGE_PATH).mkdir(parents=True, exist_ok=True)
    return Path(START_STAGE_PATH)


mast_jwst_download_url = f"{mast_jwst_base_url}/retrieve_product"


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
async def download_single_request(
    product: dict,
    output_dir: Path,
    progress: Progress,
    exist_mode: Literal["skip", "overwrite"] = "skip",
):
    """Download a single product from MAST JWST.
    
    Args:
        product (dict): The product information dictionary containing keys like 'filename', 'size', etc.
        output_dir (Path): The directory where the product will be saved.
        exist_mode (Literal["skip", "overwrite"]): How to handle existing files. Defaults to "skip".
        individual_progress_callback (Callable | None): Optional callback function to report individual file download progress.
        total_progress_callback (Callable | None): Optional callback function to report total download progress.
    
    Returns:
        bool: True if the download was successful, False otherwise.
    """
    filename = product["filename"]
    output_path = output_dir / filename
    if output_path.exists() and exist_mode == "skip":
        # Check if the file size matches the expected size
        if output_path.stat().st_size == product['size']:
            logger.info(f"File {output_path} already exists, skipping download.")
            # Call the progress callback if it exists
            progress.update(0, advance=1)
            return False
        else:
            logger.warning(
                f"File {output_path} exists but size does not match. Will overwrite."
            )

    MAX_CONCURRENT_DOWNLOADS = 5
    async with ConnectionSession.session(
        max_tcp_connector=MAX_CONCURRENT_DOWNLOADS
    ) as session:
        try:
            generator = ConnectionSession.download_and_save_async(
                url=mast_jwst_download_url,
                output_path=output_path,
                session=session,
                method="GET",
                params={
                    "product_name": filename,
                },
                progress_callback=lambda: progress.update(0, advance=1) 
            )
            task_id = progress.add_task(
                f"Downloading {filename}"
                )
            progress.update(task_id, description=f"Downloading File {task_id}", total=product['size'])

            ## for future individual download progress tracking
            async for downloaded_size, total_size in generator:
                progress.update(
                    task_id,
                    advance=downloaded_size,
                    total=total_size,
                    )
            progress.remove_task(task_id)
            return True
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False


async def download_products(
    products: list[dict],
    output_dir: Path,
    exist_mode: Literal["skip", "overwrite"] = "skip",
):
    """Wrapping the download_single_request so that it can be called by runnify."""
    tasks = []
    with Progress(
        TextColumn("[bold blue]{task.description}[/bold blue]", justify="right"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        console=console,
        ) as progress:
        progress.add_task("Total Progress:", total=len(products))
        async with asyncer.create_task_group() as task_group:
            for i, product in enumerate(products):
                task = task_group.soonify(download_single_request)(
                    product=product,
                    output_dir=output_dir,
                    progress=progress,
                    exist_mode=exist_mode,
                )
                tasks.append(task)
    ## wait for all tasks to complete and report the results
    download_status = []
    for task in tasks:
        result = task.value
        download_status.append(result)
    return download_status


@retrieve_app.command(
    name="download", help="Download the JWST data of given products list."
)
def cli_retrieve_download(
    products_file: Annotated[
        Path,
        typer.Argument(
            help="File containing the products list to download, the output of [cyan]retrieve[/cyan] command.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dir",
            help='Directory to save the downloaded products. Default is the path of start stage in "[yellow].env[/yellow]". If not set, "[yellow]/downloads[/yellow]" in the current directory.',
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            rich_help_panel="Output",
            metavar="DIR",
            prompt="Output directory is not specified. Use the default path (press [red]Enter[/red] to confirm or type a new path):\n ",
            prompt_required=False,
        ),
    ] = None,
    current_folder: Annotated[
        bool,
        typer.Option(
            "-c",
            "--current-folder",
            help="Download the products to the current folder, this will override the [blue]--output-dir[/blue] option.",
            rich_help_panel="Output",
        ),
    ] = False,
    skip_exist: Annotated[
        bool,
        typer.Option(
            "-s",
            "--skip-exist",
            is_flag=True,
            help="Skip the existing files in the output directory, default is True. If set to False, will overwrite the existing files.",
            rich_help_panel="Output",
        ),
    ] = False,
):
    global console
    ## Load environment variables from .env
    if output_dir is None:
        env_dir = env_download_folder()
        output_dir = env_dir if env_dir else Path.cwd() / "downloads"
    ## Check if the output directory exists, if not, create it
    if current_folder:
        output_dir = Path.cwd()
    else:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]{output_dir} is created[/green]")

    ## Load the products from the file
    with open(products_file, "r") as f:
        products = json.load(f)
    if not products:
        console.print("[red]No products found in the file.[/red]")
        raise typer.Exit(code=1)

    if skip_exist:
        exist_mode = "skip"
    else:
        exist_mode = "overwrite"
    

    download_status = asyncer.runnify(download_products)(
        products=products,
        output_dir=output_dir,
        exist_mode=exist_mode,
    )

    download_status = [not not status for status in download_status]

    console.print(
        f"[green]Newly downloaded {sum(download_status)} out of {len(products)} products.[/green]"
    )


if __name__ == "__main__":
    retrieve_app()
