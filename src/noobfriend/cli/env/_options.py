"""Typer parameter callbacks for the ``env`` CLI."""

import typer


def start_stage_callback(value: str | None) -> str | None:
    """Validate that a stage label begins with a digit.

    Parameters
    ----------
    value : str or None
        Raw ``--start-stage`` input, or ``None`` when not supplied.

    Returns
    -------
    str or None
        The lower-cased stage label, or ``None`` if not supplied.

    Raises
    ------
    typer.BadParameter
        If ``value`` is non-empty and its first character is not a digit.
    """
    if value is None or value == "":
        return None
    if not value[0].isdigit():
        raise typer.BadParameter("The first character of a stage must be a digit.")
    return value.lower()


def stage_list_callback(values: list[str]) -> list[str]:
    """Validate and normalise a list of stage labels.

    Parameters
    ----------
    values : list of str
        Raw stage labels.

    Returns
    -------
    list of str
        Lower-cased labels.

    Raises
    ------
    typer.BadParameter
        If any label's first character is not a digit.
    """
    for value in values:
        if not value or not value[0].isdigit():
            raise typer.BadParameter(
                f"The first character of stage '{value}' must be a digit."
            )
    return [value.lower() for value in values]
