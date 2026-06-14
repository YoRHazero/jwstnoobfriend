"""Shared option helpers for the box CLI."""

import typer

from noobfriend.core.env import get_settings


def resolve_stage(stage: str | None) -> str:
    """Return ``stage``, else the ``START_STAGE`` configured in the ``.env``.

    Parameters
    ----------
    stage : str or None
        Explicit stage label; used as-is when given. Stage labels are an open
        set, so the value is not format-checked.

    Returns
    -------
    str
        The resolved stage label.

    Raises
    ------
    typer.BadParameter
        If ``stage`` is ``None`` and ``START_STAGE`` is unset.
    """
    if stage is not None:
        return stage
    configured = get_settings().start_stage
    if not configured:
        raise typer.BadParameter(
            "no STAGE given and START_STAGE is not set in the .env."
        )
    return configured
