"""Shared :mod:`rich` console for logging, progress, and ad-hoc output.

The module exposes a single :data:`console` instance for the whole package.
Import it directly::

    from noobfriend.core.console import console

    console.print("[bold green]hello[/bold green]")

The instance is created with no constructor arguments so that ``rich`` can
auto-detect the environment (Jupyter, terminal, dumb pipe) and pick the
appropriate renderer.
"""

from rich.console import Console

console: Console = Console()
