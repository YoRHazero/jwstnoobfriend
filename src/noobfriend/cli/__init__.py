"""Command-line interface for noobfriend.

Re-exports the root Typer application defined in ``cli/app.py``. The console
entry point declared in ``pyproject.toml`` targets :func:`main`.
"""

from noobfriend.cli.app import app, main

__all__ = ["app", "main"]
