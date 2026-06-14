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
    is_fits_product,
    is_rateint_product,
)
from noobfriend.cli.fetch._options import (
    DEFAULT_RETRY_LIMIT,
    product_level_callback,
    proposal_id_callback,
    resolve_product_level,
)
from noobfriend.core.console import console
from noobfriend.core.env import get_settings
from noobfriend.core.display.progress import track_time
from noobfriend.core.io import HTTPSession


def _resolve_output_target(output_file: Path | None) -> Path | None:
    """Decide where (if anywhere) to save the search results.

    An explicit ``--output-file`` is used as-is. When it is omitted, the user is
    asked — before the minute-long query, so a forgotten ``-o`` does not discard
    the results — whether to save to ``products.json`` in the current directory.
    In a non-interactive context (no TTY) the question is skipped and nothing is
    saved, preserving the opt-in default.

    Parameters
    ----------
    output_file : Path or None
        The ``--output-file`` value, or ``None`` when the option was omitted.

    Returns
    -------
    Path or None
        The file to save the products to, or ``None`` to skip saving.
    """
    if output_file is not None:
        return output_file
    if not sys.stdin.isatty():
        return None
    default_path = Path.cwd() / "products.json"
    if typer.confirm(
        f"No -o given. Save results to {default_path} after the search?",
        default=True,
    ):
        return default_path
    return None


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
        str | None,
        typer.Option(
            "-l",
            "--product-level",
            callback=product_level_callback,
            help="Product stage to search. Defaults to the START_STAGE setting, or '1b' if unset. The naming convention is based on https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/stages.html",
        ),
    ] = None,
    retry_limit: Annotated[
        int,
        typer.Option(
            "-r",
            "--retry",
            min=1,
            help="How many times to re-query a fileset that returns no products before giving up.",
        ),
    ] = DEFAULT_RETRY_LIMIT,
    no_rateint: Annotated[
        bool,
        typer.Option(
            "-R",
            "--no-rateint",
            help="Exclude stage-2a per-integration 'rateints' products (and their previews), keeping only the 2D 'rate' images. No effect on other stages.",
        ),
    ] = False,
    all_formats: Annotated[
        bool,
        typer.Option(
            "-a",
            "--all-formats",
            help="Include all file formats. By default only .fits data files are kept; this also adds the preview/thumbnail .jpg files.",
        ),
    ] = False,
    show_example: Annotated[
        bool,
        typer.Option(
            "-s",
            "--show-example",
            help="Show first 5 products of output, default is False.",
        ),
    ] = False,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-file",
            help="File to save the products to. If omitted, you are asked before the search whether to save to 'products.json' in the current directory.",
            rich_help_panel="Output",
            metavar="FILE",
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Search MAST for the products of a JWST proposal and optionally save them."""
    product_level = resolve_product_level(product_level, get_settings().start_stage)
    save_target = _resolve_output_target(output_file)
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

    if not all_formats:
        before = len(results)
        results = [product for product in results if is_fits_product(product)]
        console.print(
            f"[dim]Excluded {before - len(results)} non-FITS product(s); "
            f"{len(results)} remaining.[/dim]"
        )

    if no_rateint:
        before = len(results)
        results = [product for product in results if not is_rateint_product(product)]
        console.print(
            f"[dim]Excluded {before - len(results)} rateints product(s); "
            f"{len(results)} remaining.[/dim]"
        )

    console.print(
        make_instrument_table(
            proposal_id,
            product_level,
            Counter(product["instrument_name"] for product in results),
        )
    )

    if show_example:
        console.print(make_product_example_table(results))

    if save_target is not None:
        console.print(f"[teal]Opening {save_target} [/teal]...")
        save_products(results, save_target)
        console.print("[green] Saved [/green]")
