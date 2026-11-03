"""Plain-text tables for summaries (agents read text, not HTML)."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["format_rows"]


def format_rows(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-align columns under their headers, indented by two spaces."""
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [
            max(width, len(cell)) for width, cell in zip(widths, row, strict=True)
        ]
    rendered = []
    for row in [list(headers), *rows]:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=True)]
        rendered.append("  " + "  ".join(cells).rstrip())
    return rendered
