"""Filesystem helpers for the box CLI: locating the NooBox manifest."""

from pathlib import Path

import typer

from noobfriend.core.env import get_settings


def resolve_box_path(path: Path | None) -> Path:
    """Return ``path``, else the ``NOOBOX_PATH`` configured in the ``.env``.

    Parameters
    ----------
    path : Path or None
        Explicit manifest path; used as-is when given.

    Returns
    -------
    Path
        The resolved manifest path.

    Raises
    ------
    typer.BadParameter
        If ``path`` is ``None`` and ``NOOBOX_PATH`` is unset.
    """
    if path is not None:
        return path
    configured = get_settings().noobox_path
    if configured is None:
        raise typer.BadParameter(
            "no --path given and NOOBOX_PATH is not set in the .env."
        )
    return configured
