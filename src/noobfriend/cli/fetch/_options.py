"""Static MAST endpoints and Typer parameter callbacks for the fetch CLI."""

from enum import Enum

import typer

MAST_JWST_BASE_URL = "https://mast.stsci.edu/search/jwst/api/v0.1"
MAST_JWST_SEARCH_URL = f"{MAST_JWST_BASE_URL}/search"
MAST_JWST_PRODUCT_URL = f"{MAST_JWST_BASE_URL}/list_products"
MAST_JWST_DOWNLOAD_URL = f"{MAST_JWST_BASE_URL}/retrieve_product"

PRODUCT_LEVELS: tuple[str, ...] = ("1b", "2a", "2b", "2c")
DEFAULT_PRODUCT_LEVEL: str = "1b"
DEFAULT_RETRY_LIMIT: int = 3


class DownloadMode(str, Enum):
    """How a remote download decides between remote-direct and relay.

    Attributes
    ----------
    auto
        Probe the remote; use remote-direct if it can reach MAST, else relay.
    remote
        Force remote-direct; error out if the remote cannot reach MAST.
    relay
        Force relay through the local machine.
    """

    auto = "auto"
    remote = "remote"
    relay = "relay"


def product_level_callback(value: str | None) -> str | None:
    """Validate explicit ``--product-level`` input, passing ``None`` through.

    ``None`` (the option was not given) is returned unchanged so the command can
    fall back to the ``START_STAGE`` setting via :func:`resolve_product_level`.

    Parameters
    ----------
    value : str or None
        Raw ``--product-level`` input, or ``None`` when the option is omitted.

    Returns
    -------
    str or None
        The validated product level, or ``None``.

    Raises
    ------
    typer.BadParameter
        If ``value`` is given but not in :data:`PRODUCT_LEVELS`.
    """
    if value is None:
        return None
    if value not in PRODUCT_LEVELS:
        raise typer.BadParameter(
            f"Invalid product level '{value}'. Choose from {list(PRODUCT_LEVELS)}."
        )
    return value


def resolve_product_level(cli_value: str | None, env_value: str | None) -> str:
    """Resolve the search product level: ``--product-level`` > ``START_STAGE`` > default.

    An explicit ``cli_value`` (already validated by :func:`product_level_callback`)
    always wins. Otherwise ``env_value`` — the ``START_STAGE`` setting — is used,
    but validated just as strictly: ``START_STAGE`` names the stage the reduction
    starts from, so a value that is not a searchable product level is a
    configuration error rather than a soft default. With neither set, falls back
    to :data:`DEFAULT_PRODUCT_LEVEL`.

    Parameters
    ----------
    cli_value : str or None
        The validated ``--product-level`` value, or ``None`` when omitted.
    env_value : str or None
        The ``START_STAGE`` setting, or ``None`` when unset.

    Returns
    -------
    str
        The product level to search.

    Raises
    ------
    typer.BadParameter
        If ``cli_value`` is ``None`` but ``env_value`` is set to a value that is
        not in :data:`PRODUCT_LEVELS`.
    """
    if cli_value is not None:
        return cli_value
    if env_value:
        candidate = env_value.strip().lower()
        if candidate not in PRODUCT_LEVELS:
            raise typer.BadParameter(
                f"START_STAGE='{env_value}' is not a searchable product level "
                f"{list(PRODUCT_LEVELS)}. Set a valid START_STAGE or pass "
                f"-l/--product-level explicitly."
            )
        return candidate
    return DEFAULT_PRODUCT_LEVEL


def proposal_id_callback(value: str) -> str:
    """Normalize a proposal ID to a zero-padded five-digit string.

    Parameters
    ----------
    value : str
        Raw proposal ID; surrounding whitespace and leading zeros are stripped
        before validation.

    Returns
    -------
    str
        The proposal ID padded to five digits, e.g. ``"1895"`` -> ``"01895"``.

    Raises
    ------
    typer.BadParameter
        If the stripped value is not purely numeric, or is longer than five
        digits.
    """
    value = value.strip().lstrip("0")
    if not value.isdigit():
        raise typer.BadParameter("Proposal ID must be a numeric string.")
    if len(value) > 5:
        raise typer.BadParameter("Proposal ID too long.")
    return value.zfill(5)
