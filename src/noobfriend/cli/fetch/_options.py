"""Static MAST endpoints and Typer parameter callbacks for the fetch CLI."""

from enum import Enum

import typer

MAST_JWST_BASE_URL = "https://mast.stsci.edu/search/jwst/api/v0.1"
MAST_JWST_SEARCH_URL = f"{MAST_JWST_BASE_URL}/search"
MAST_JWST_PRODUCT_URL = f"{MAST_JWST_BASE_URL}/list_products"
MAST_JWST_DOWNLOAD_URL = f"{MAST_JWST_BASE_URL}/retrieve_product"

PRODUCT_LEVELS: tuple[str, ...] = ("1b", "2a", "2b", "2c")
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


def product_level_callback(value: str) -> str:
    """Validate that ``value`` is one of the supported product levels.

    Parameters
    ----------
    value : str
        Raw ``--product-level`` input.

    Returns
    -------
    str
        The validated product level.

    Raises
    ------
    typer.BadParameter
        If ``value`` is not in :data:`PRODUCT_LEVELS`.
    """
    if value not in PRODUCT_LEVELS:
        raise typer.BadParameter(
            f"Invalid product level '{value}'. Choose from {list(PRODUCT_LEVELS)}."
        )
    return value


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
