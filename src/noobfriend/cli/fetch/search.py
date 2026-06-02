"""The ``fetch search`` command: discover a proposal's products on MAST."""

import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.fetch._io import save_products
from noobfriend.cli.fetch._presenters import (
    create_search_progress,
    make_accessibility_table,
    make_instrument_table,
    make_missing_products_table,
    make_product_example_table,
    time_footer,
)
from noobfriend.cli.fetch._search_service import (
    fetch_products_for_file_sets,
    fetch_proposal_file_sets,
)
from noobfriend.cli.fetch._options import (
    DEFAULT_RETRY_LIMIT,
    product_level_callback,
    proposal_id_callback,
)
from noobfriend.core.console import console
from noobfriend.core.display.progress import track_time
from noobfriend.core.io import HTTPSession


@time_footer
def cli_search(
    proposal_id: Annotated[
        str,
        typer.Argument(
            help="Proposal ID to search, 5 digits, e.g. '01895'.",
            callback=proposal_id_callback,
        ),
    ],
    product_level: Annotated[
        str,
        typer.Option(
            "-l",
            "--product-level",
            callback=product_level_callback,
            help="Product stage to search. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
        ),
    ] = "1b",
    retry_limit: Annotated[
        int,
        typer.Option(
            "-r",
            "--retry",
            min=1,
            help="How many times to re-query a fileset that returns no products before giving up.",
        ),
    ] = DEFAULT_RETRY_LIMIT,
    show_example: Annotated[
        bool,
        typer.Option(
            "-s",
            "--show-example",
            help="Show first 5 products of output, default is False.",
        ),
    ] = False,
    output_file: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-file",
            help="File to save the products, default is 'products.json' in the current directory. If this option is not provided, the products will not be saved.",
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
) -> None:
    """Search MAST for the products of a JWST proposal and optionally save them."""
    http = HTTPSession()

    # Phase 1: discover file sets (single request) -> indeterminate spinner.
    search_filesets = track_time("Searching MAST file sets")(asyncio.run)(
        fetch_proposal_file_sets(http, proposal_id, product_level)
    )
    console.print(
        make_accessibility_table(
            proposal_id,
            Counter(fileset["access"] for fileset in search_filesets),
        )
    )

    # Phase 2: fetch products per file set -> determinate bar with elapsed / ETA.
    public_count = sum(
        1 for fileset in search_filesets if fileset["access"] != "private"
    )
    with create_search_progress() as progress:
        task_id = progress.add_task("Fetching products from MAST", total=public_count)
        results, missing_filesets = asyncio.run(
            fetch_products_for_file_sets(
                http,
                search_filesets,
                product_level,
                retry_limit,
                on_fileset_done=lambda: progress.advance(task_id),
            )
        )

    if missing_filesets:
        console.print(make_missing_products_table(missing_filesets))
        console.print("This problem can be solved by rerunning the command.")

    console.print(
        make_instrument_table(
            proposal_id,
            product_level,
            Counter(product["instrument_name"] for product in results),
        )
    )

    if show_example:
        console.print(make_product_example_table(results))

    output_option_used = "-o" in sys.argv or "--output-file" in sys.argv
    if output_option_used and output_file is not None:
        console.print(f"[teal]Opening {output_file} [/teal]...")
        save_products(results, output_file)
        console.print("[green] Saved [/green]")
