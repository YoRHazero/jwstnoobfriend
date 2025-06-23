import typer
from typing import Annotated, Iterable, Callable
from pathlib import Path
from collections import Counter
from jwstnoobfriend.utils.display import track_func, track, console
from astroquery.mast.missions import MastMissionsClass
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
import aiohttp
import asyncer
from jwstnoobfriend.utils.network import ConnectionSession

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
    global console
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
## 4. Sometimes the number of products has problem, solve this potential bug.

mast_jwst_base_url = "https://mast.stsci.edu/search/jwst/api/v0.1"
mast_jwst_search_url = f"{mast_jwst_base_url}/search"
mast_jwst_product_url = f"{mast_jwst_base_url}/list_products"

async def search_proposal_id(
    proposal_id: str,
    product_level: str,
):
    async with ConnectionSession.session() as session:
        search_json = await ConnectionSession.fetch_json_async(
            mast_jwst_search_url,
            session,
            method='POST',
            body={"conditions": [{"program": proposal_id, "productLevel": product_level}]}
        )
        return search_json['results']
    
async def send_products_request(
    fileset_name: str,
    product_level: str,
    include_rateint: bool,
):
    async with ConnectionSession.session() as session:
        product_json = await ConnectionSession.fetch_json_async(
            mast_jwst_product_url,
            session,
            method='GET',
            params={"dataset_ids": fileset_name}
        )
        products = product_json["products"]
        if not include_rateint:
            products = [p for p in products if 'rateint' not in p['file_suffix']]
        products_filtered = [p for p in products if p["category"] == product_level]
        if len(products_filtered) != 10:
            print(f"Found {len(products_filtered)} products for {fileset_name} with product level {product_level}.")
        return products_filtered

async def get_products(
    search_results: Iterable[dict],
    product_level: str,
    include_rateint: bool,
    error_table: Table | None = None,
):
    tasks = []
    async with asyncer.create_task_group() as task_group:
        for result in search_results:
            fileset_name = result['fileSetName']
            soon_products = task_group.soonify(send_products_request)(
                fileset_name=fileset_name,
                product_level=product_level,
                include_rateint=include_rateint,
            )
            tasks.append(soon_products)
    
    results = []
    for task in tasks:
        products = task.value
        if not products and error_table is not None:
            error_table.add_row(
                'error',
            )
        results.extend(products)
    return results
        


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

    global console
    search_filesets = asyncer.runnify(search_proposal_id)(   
        proposal_id=proposal_id,
        product_level=product_level,
    )
    
    ## Show the summary of the search results
    fileset_access = Counter([fs['access'] for fs in search_filesets])
    table_access = Table(title=f"Accessibility of {proposal_id}")
    table_access.add_column("Accessibility", justify="left", style="cyan", width=20)
    table_access.add_column("number of dataset", justify="right", style='green')
    for access, count in fileset_access.items():
        table_access.add_row(access, str(count))
    console.print(table_access)
    
    results = asyncer.runnify(get_products)(
        search_results=search_filesets,
        product_level=product_level,
        include_rateint=include_rateint,
    )
    

    
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
    