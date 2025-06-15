import typer
from typing import Annotated, Iterable
from pathlib import Path
from collections import Counter
from jwstnoobfriend.utils.display import track_func, track
from astroquery.mast.missions import MastMissionsClass
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
import aiohttp
import asyncer
from jwstnoobfriend.utils.network import fetch_json_async

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Check and retrieve JWST data from MAST.",
                  )

mission = MastMissionsClass(mission="jwst")

def util_first_contact(program_query: Iterable, contact_key: str = 'instrume') -> dict[str, int]:
    """Utility function to find the first contact index for each instrument in the program query."""
    contact = {}
    for i, row in enumerate(program_query):
        if row[contact_key] in contact:
            continue
        contact[str(row[contact_key])] = i
    return contact

def product_level_callback(value: str) -> str:
    """Callback function to validate product level input."""
    choices = ['1b', '2a', '2b', '2c']
    if value not in choices:
        raise typer.BadParameter(f"Invalid product level '{value}'. Choose from {choices}.")
    return value

@app.command(name="check", help="Old version of the check command. In this version, \
                                    the retrieval is done by astroquery. And for the speed,\
                                    we assume every dataset of the same instrument has the same \
                                    suffix, which may not be true in some cases. Use [cyan]retrieve[/cyan] command instead.")
def cli_retrieve_check(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID to check, 5 digits, e.g. '01895'.")],
    product_level: Annotated[str, 
                             typer.Option('-l', '--product-level', 
                                          callback=product_level_callback,
                                          help="Product stage to check. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
                                          )] = '1b',
    show_example: Annotated[bool, typer.Option('-s', '--show-example',
                                               help="Show first 5 products of output, default is False.",
                                               )] = False,
    show_suffix: Annotated[bool, typer.Option('-d', '--show-suffix',
                                              help="Show the suffix list of the products for each instrument, default is False.",
                                              )] = False,
    include_rateint: Annotated[bool, typer.Option('-r', '--include-rateint',
                                                  help="Include rateint products in the output, default is False, in most cases rateint products are just the same as rate products.",
                                                  )] = False,
):
    console = Console()
    ## Format of the table of accessibility
    table_access = Table(title=f"Accessibility of {proposal_id}")
    table_access.add_column("Accessibility", justify="left", style="cyan", width=20)
    table_access.add_column("number of dataset", justify="right", style='green')
    
    ## Format of the table of instruments
    table_instrument = Table(title=f"Instrument file numbers of {proposal_id} ({product_level})")
    table_instrument.add_column("Instrument ", justify="left", style="cyan", width=20)
    table_instrument.add_column("number of files", justify="right", style='green')
    
    ## Get the file set ID for the proposal and product level
    program_query = mission.query_criteria( # type: ignore
        program=proposal_id,
        productLevel=product_level
    )
    
    if len(program_query) == 0:
        console.print(f"[red]No data found for proposal ID {proposal_id} with product level {product_level}.[/red]")
        raise typer.Exit(code=1)
    
    instrument_contact = util_first_contact(program_query)
    products_suffix_dict = {}
    ## Filter the products based on the product level
    for instrument, index in instrument_contact.items():
        products = mission.get_product_list( # type: ignore
            program_query['fileSetName'][index],
        )
        selected_products = mission.filter_products(
            products,
            category = product_level
        )
        products_suffix_dict[instrument] = [p['filename'].split(p['dataset'])[-1] for p in selected_products]
        
        if not include_rateint:
            products_suffix_dict[instrument] = [
                suffix for suffix in products_suffix_dict[instrument]
                if not 'rateint' in suffix
            ]
        
        if show_suffix:
            console.print(f"[yellow]Suffixes for {instrument}:[/yellow]")
            console.print(products_suffix_dict[instrument])
        
        
    access_counts = Counter(program_query['access'])
    for access, count in access_counts.items():
        table_access.add_row(access, str(count))
    console.print(table_access)
    
    fileset_counts = Counter(program_query['instrume'])
    file_counts = {instrument: len(products_suffix_dict[instrument]) * fileset_counts[instrument] for instrument in products_suffix_dict}
    for instrument, count in file_counts.items():
        table_instrument.add_row(str(instrument), str(count))
    console.print(table_instrument)
    
    if show_example:
        table_example = Table(title="Example Products")
        table_example.add_column("Product File Name", justify="left", style="cyan", width = 10, no_wrap=False)
        table_example.add_column("Accessibility", justify="left", style="red")
        table_example.add_column("Instrument", justify="left", style="green")
        for product in selected_products[:5]:
            table_example.add_row(
                product['product_key'],
                product['access'],
                product['instrument']
            )


## To do list:
## 1. Add the download function to download the products.
## 2. Add the progress bar to show the download progress.
## 3. Add the live view to show the running time of the retrieval.

mast_jwst_base_url = "https://mast.stsci.edu/search/jwst/api/v0.1"
mast_jwst_search_url = f"{mast_jwst_base_url}/search"
mast_jwst_product_url = f"{mast_jwst_base_url}/list_products"
async def get_product_list(
    proposal_id: str,
    product_level: str = '1b',
    include_rateint: bool = False,
):
    async with aiohttp.ClientSession() as session:
        search_json = await fetch_json_async(
            mast_jwst_search_url,
            session,
            method='POST',
            body={"conditions": [{"program": proposal_id, "productLevel": product_level}]}
        )
        search_filesets = search_json['results']
        
        soon_values = []
        async with asyncer.create_task_group() as tg:
            for fileset in search_filesets:
                fileset_name = fileset['fileSetName']
                products_soon = tg.soonify(fetch_json_async)(
                    mast_jwst_product_url,
                    session,
                    method='GET',
                    params={"dataset_ids": fileset_name}
                )
                soon_values.append(products_soon)
        
        results = []
        for soon_value in soon_values:
            product_json = soon_value.value
            products = product_json["products"]
            results.extend(
                [p for p in products if p["category"] == product_level]
            )
        if not include_rateint:
            results = [p for p in results if 'rateint' not in p['file_suffix']]
    return search_filesets, results

@app.command(name="retrieve", help="Check the JWST data of given proposal id from MAST.")
def cli_retrieve_check_async(
    proposal_id: Annotated[str, typer.Argument(help="Proposal ID to check, 5 digits, e.g. '01895'.")],
    product_level: Annotated[str, 
                             typer.Option('-l', '--product-level', 
                                          callback=product_level_callback,
                                          help="Product stage to check. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
                                          )] = '1b',
    show_example: Annotated[bool, typer.Option('-s', '--show-example',
                                               help="Show first 5 products of output, default is False.",
                                               )] = False,
    include_rateint: Annotated[bool, typer.Option('-r', '--include-rateint',
                                                  help="Include rateint products in the output, default is False, in most cases rateint products are just the same as rate products.",
                                                  )] = False,
    download_folder: Annotated[Path, typer.Option('-d', '--download-folder', 
                                                  help="Folder to download the products, default is current directory.",
                                                  exists=True, file_okay=False, dir_okay=True, resolve_path=True)] = Path.cwd(),
):

    console = Console()
    search_filesets, results = asyncer.runnify(get_product_list)(   
        proposal_id=proposal_id,
        product_level=product_level,
        include_rateint=include_rateint,
    )
    
    ## Show the summary of the search results
    fileset_access = Counter([fs['access'] for fs in search_filesets])
    table_access = Table(title=f"Accessibility of {proposal_id}")
    table_access.add_column("Accessibility", justify="left", style="cyan", width=20)
    table_access.add_column("number of dataset", justify="right", style='green')
    for access, count in fileset_access.items():
        table_access.add_row(access, str(count))
    console.print(table_access)
    ## Show the summary of the instruments
    products_instrument = Counter([p['instrument_name'] for p in results])
    table_instrument = Table(title=f"Instrument file numbers of {proposal_id} ({product_level})")
    table_instrument.add_column("Instrument ", justify="left", style="cyan", width=20)
    table_instrument.add_column("number of files", justify="right", style='green')
    for instrument, count in products_instrument.items():
        table_instrument.add_row(str(instrument), str(count))
    console.print(table_instrument)
    ## Show the example products
    if show_example:
        table_example = Table(title="Example Products")
        table_example.add_column("Product File Name", justify="left", style="cyan", width=10, no_wrap=False)
        table_example.add_column("Accessibility", justify="left", style="red")
        table_example.add_column("Instrument", justify="left", style="green")
        table_example.add_column("File Size", justify="left", style="magenta")
        for product in results[:5]:
            table_example.add_row(
                product['filename'],
                product['access'],
                product['instrument_name'],
                f"{product['size'] / (1024 * 1024):.2f} MB" if product['size'] else "N/A"
            )
        console.print(table_example)
            
if __name__ == "__main__":
    app()    
    