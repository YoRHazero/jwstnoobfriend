"""Rich renderers for the box CLI: a per-stage summary and a product listing."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rich.table import Table

from noobfriend.navigation import NooBook, NooBox


def make_box_summary_table(target: Path, box: NooBox) -> Table:
    """Build a per-stage breakdown of ``box`` (caption carries the totals).

    Parameters
    ----------
    target : Path
        Manifest path the box was read from / written to (titles the table).
    box : NooBox
        The collection to summarise.

    Returns
    -------
    rich.table.Table
        One row per stage with its book and probed counts; the caption reports
        the total books, distinct stages, and lineage boundary sizes.
    """
    stage_counts: Counter[str] = Counter(book.stage for book in box)
    probed_counts: Counter[str] = Counter(book.stage for book in box if book.is_probed)

    table = Table(title=f"NooBox: {target.name}")
    table.add_column("Stage", justify="left", style="cyan")
    table.add_column("Books", justify="right", style="green")
    table.add_column("Probed", justify="right", style="magenta")
    for stage in sorted(stage_counts):
        table.add_row(stage, str(stage_counts[stage]), str(probed_counts[stage]))

    table.caption = (
        f"{len(box)} books · {len(stage_counts)} stages · "
        f"{len(box.leaves())} leaves · {len(box.roots())} roots"
    )
    return table


def make_box_listing_table(books: Sequence[NooBook]) -> Table:
    """Build a one-row-per-product table.

    Parameters
    ----------
    books : sequence of NooBook
        The products to list.

    Returns
    -------
    rich.table.Table
    """
    table = Table(title="Products")
    table.add_column("ID", justify="left", style="cyan")
    table.add_column("Stage", justify="left", style="green")
    table.add_column("Detector", justify="left")
    table.add_column("Filter", justify="left")
    table.add_column("Exposure", justify="left")
    table.add_column("Parents", justify="right", style="magenta")
    for book in books:
        table.add_row(
            book.id,
            book.stage,
            book.detector or "—",
            book.filter or "—",
            "+".join(book.exposure),
            str(len(book.parent_ids)),
        )
    return table
