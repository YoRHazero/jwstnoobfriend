"""Rich renderers for the ``env`` CLI: coloured path-existence status."""

from pathlib import Path

from noobfriend.cli.env._io import split_existing


def render_path_status(path: Path) -> str:
    """Return a rich-markup string colouring a path by what exists on disk.

    A fully existing path is shown green; a partially existing one shows the
    existing prefix in green and the missing remainder in red.

    Parameters
    ----------
    path : Path
        The path to describe (resolved by the caller).

    Returns
    -------
    str
        Rich console markup, e.g. ``"[green]/data[/green]/[red]missing[/red]"``.
    """
    existing, missing = split_existing(path)
    if missing is None:
        return f"[green]{existing}[/green]"
    return f"[green]{existing}[/green]/[red]{missing}[/red]"
