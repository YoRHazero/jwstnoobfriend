"""Logging configuration helpers for the ``noobfriend`` package."""

import logging

from rich.logging import RichHandler

from noobfriend.core.console import console


def setup_logging(
    level: str | int = "INFO",
    *,
    show_path: bool = True,
) -> None:
    """Configure rich-formatted logging for the ``noobfriend`` logger tree.

    This is the convenience entry point for callers (notebooks, scripts, CLI).
    Library code never calls this; it only emits records via
    ``logging.getLogger(__name__)`` and lets the caller decide how to route them.

    The function is idempotent: calling it again replaces handlers rather than
    stacking them, which prevents duplicated output when a Jupyter cell is
    re-executed.

    Parameters
    ----------
    level : str or int, default ``"INFO"``
        Logging level applied to the ``noobfriend`` logger. Accepts standard
        level names (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ...) or numeric
        values such as ``logging.DEBUG``.
    show_path : bool, default ``True``
        Whether ``RichHandler`` should display the source path and line number
        for each record.

    Returns
    -------
    None

    Notes
    -----
    The handler is bound to the shared :data:`noobfriend.core.console.console`,
    so log output and any future progress displays cooperate instead of
    fighting for the terminal.
    """
    handler = RichHandler(
        console=console,
        show_path=show_path,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger("noobfriend")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
