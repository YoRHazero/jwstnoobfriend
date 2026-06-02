"""Rich renderers for the fetch CLI: summary tables, progress, and a timing footer."""

import functools
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ParamSpec, TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table

from noobfriend.core.console import console

P = ParamSpec("P")
R = TypeVar("R")


def time_footer(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a command so it prints the elapsed wall-clock time when it returns."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            console.print(f"[dim]Elapsed: {time.perf_counter() - start:.2f}s[/dim]")

    return wrapper


def make_accessibility_table(
    proposal_id: str, access_counts: Mapping[str, int]
) -> Table:
    """Build a table of dataset counts grouped by accessibility."""
    table = Table(title=f"Accessibility of {proposal_id}")
    table.add_column("Accessibility", justify="left", style="cyan", width=20)
    table.add_column("number of dataset", justify="right", style="green")
    for access, count in access_counts.items():
        table.add_row(access, str(count))
    return table


def make_instrument_table(
    proposal_id: str, product_level: str, instrument_counts: Mapping[str, int]
) -> Table:
    """Build a table of file counts grouped by instrument."""
    table = Table(title=f"Instrument file numbers of {proposal_id} ({product_level})")
    table.add_column("Instrument ", justify="left", style="cyan", width=20)
    table.add_column("number of files", justify="right", style="green")
    for instrument, count in instrument_counts.items():
        table.add_row(str(instrument), str(count))
    return table


def make_product_example_table(products: Sequence[dict]) -> Table:
    """Build a preview table of the first few products."""
    table = Table(title="Example Products")
    table.add_column(
        "Product File Name", justify="left", style="cyan", width=10, no_wrap=False
    )
    table.add_column("Accessibility", justify="left", style="red")
    table.add_column("Instrument", justify="left", style="green")
    table.add_column("File Size", justify="left", style="magenta")
    for product in products[:5]:
        table.add_row(
            product["filename"],
            product["access"],
            product["instrument_name"],
            f"{product['size'] / (1024 * 1024):.2f} MB" if product["size"] else "N/A",
        )
    return table


def make_missing_products_table(fileset_names: Sequence[str]) -> Table:
    """Build a table listing file sets that returned no products."""
    table = Table(title="No products found or an error occurred.")
    table.add_column("Fileset", justify="left", style="red", width=30)
    for fileset_name in fileset_names:
        table.add_row(fileset_name)
    return table


def make_failed_downloads_table(filenames: Sequence[str]) -> Table:
    """Build a table listing products that failed to download."""
    table = Table(title="Failed downloads")
    table.add_column("Filename", justify="left", style="red", width=40)
    for filename in filenames:
        table.add_row(filename)
    return table


def make_manifest_schema_table(
    products_file: Path,
    column_counts: Mapping[str, int],
    total_terms: int,
) -> Table:
    """Build a table of manifest columns and how many terms contain each."""
    table = Table(title=f"Manifest Schema: {products_file.name}")
    table.add_column("Column", justify="left", style="cyan")
    table.add_column("Terms", justify="right", style="green")
    table.add_column("Coverage", justify="right", style="magenta")
    for column_name, term_count in sorted(column_counts.items()):
        coverage = f"{term_count}/{total_terms}" if total_terms else "0/0"
        table.add_row(column_name, str(term_count), coverage)
    return table


def make_manifest_summary_table(
    column: str,
    value_counts: Mapping[str, int],
    total_terms_with_column: int,
) -> Table:
    """Build a table of distinct values (and counts) for one manifest column."""
    table = Table(
        title=(
            f"Column Summary: {column} "
            f"({len(value_counts)} distinct values, {total_terms_with_column} terms)"
        )
    )
    table.add_column("Value", justify="left", style="cyan")
    table.add_column("Terms", justify="right", style="green")
    for value, count in sorted(
        value_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        table.add_row(value, str(count))
    return table


def create_search_progress() -> Progress:
    """Build the determinate progress display for the product-discovery phase.

    Shows a live spinner, a bar tracking how many file sets have been queried,
    and both elapsed and estimated-remaining time.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_download_progress() -> Progress:
    """Build the multi-bar progress display used by the download command."""
    return Progress(
        TextColumn(
            "[bold blue]{task.description}[/bold blue]",
            justify="left",
            table_column=Column(width=48, no_wrap=True, overflow="ellipsis"),
        ),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False,
        console=console,
    )
