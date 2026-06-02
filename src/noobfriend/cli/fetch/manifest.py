"""The ``fetch manifest`` command group: inspect and filter saved manifests."""

from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.fetch._io import save_products
from noobfriend.cli.fetch._manifest_service import (
    count_manifest_columns,
    deduplicate_columns,
    filter_manifest_terms,
    load_manifest_terms,
    summarize_manifest_column,
    validate_columns,
)
from noobfriend.cli.fetch._presenters import (
    make_manifest_schema_table,
    make_manifest_summary_table,
)
from noobfriend.core.console import console

manifest_app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Inspect and filter downloaded product manifests.",
)


@manifest_app.command(name="schema")
def cli_manifest_schema(
    products_file: Annotated[
        Path,
        typer.Argument(
            help="Products manifest JSON file to inspect.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
) -> None:
    """Show which columns exist in a manifest and how many terms contain each."""
    products = load_manifest_terms(products_file)
    if not products:
        console.print("[yellow]Manifest is empty.[/yellow]")
        return

    column_counts = count_manifest_columns(products)
    console.print(
        make_manifest_schema_table(products_file, column_counts, len(products))
    )


@manifest_app.command(name="summary")
def cli_manifest_summary(
    products_file: Annotated[
        Path,
        typer.Argument(
            help="Products manifest JSON file to summarize.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    columns: Annotated[
        list[str],
        typer.Argument(
            help="One or more column names to summarize.",
            metavar="COLUMN",
        ),
    ],
) -> None:
    """Summarize distinct values and term counts for one or more manifest columns."""
    products = load_manifest_terms(products_file)
    if not products:
        console.print("[yellow]Manifest is empty.[/yellow]")
        return

    columns = deduplicate_columns(columns)
    column_counts = count_manifest_columns(products)
    validate_columns(columns, column_counts)

    for column in columns:
        value_counts = summarize_manifest_column(products, column)
        console.print(
            make_manifest_summary_table(column, value_counts, column_counts[column])
        )


@manifest_app.command(name="filter")
def cli_manifest_filter(
    products_file: Annotated[
        Path,
        typer.Argument(
            help="Products manifest JSON file to filter.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    column: Annotated[
        str,
        typer.Option(
            "-c",
            "--column",
            help="Column name to match against.",
        ),
    ],
    values: Annotated[
        list[str],
        typer.Option(
            "-v",
            "--value",
            help="Value to keep. Repeat this option to provide multiple values. Values are parsed as JSON literals when possible.",
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-file",
            help="Path to save the filtered manifest. Defaults to overwriting PRODUCTS_FILE.",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Filter a manifest by column values and write the matching terms to a JSON file."""
    if not values:
        raise typer.BadParameter("At least one --value must be provided.")

    products = load_manifest_terms(products_file)
    column_counts = count_manifest_columns(products)
    validate_columns([column], column_counts)
    filtered_products = filter_manifest_terms(products, column, values)

    destination = output_file or products_file
    save_products(filtered_products, destination)
    console.print(
        f"[green]Saved {len(filtered_products)} out of {len(products)} terms to {destination}.[/green]"
    )

    if not filtered_products:
        console.print("[yellow]No terms matched the requested column values.[/yellow]")
