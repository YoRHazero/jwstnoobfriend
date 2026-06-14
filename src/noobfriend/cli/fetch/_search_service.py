"""Async MAST queries: discover filesets and their products for a proposal."""

import asyncio
import logging
from collections.abc import Callable, Iterable

from noobfriend.cli.fetch._options import (
    DEFAULT_RETRY_LIMIT,
    MAST_JWST_PRODUCT_URL,
    MAST_JWST_SEARCH_URL,
)
from noobfriend.core.io import HTTPSession

logger = logging.getLogger(__name__)

#: Filename / ``file_suffix`` tokens that mark a per-integration countrate product.
RATEINT_SUFFIXES: frozenset[str] = frozenset({"rateint", "rateints"})


def is_rateint_product(product: dict) -> bool:
    """Return whether ``product`` is a stage-2a per-integration (rateints) product.

    The 3D ``_rateints`` countrate cube is, for many single-integration
    exposures, redundant with the 2D ``_rate`` image, so it is often worth
    excluding from a download. A product is recognised by its ``file_suffix``
    field, falling back to the trailing token of its ``filename`` — the latter
    also catches the matching preview ``.jpg``, whose ``file_suffix`` is the
    generic ``_preview``. The ``_rate`` image is never matched, despite
    ``rate`` being a prefix of ``rateints``, because only the whole token is
    compared.

    Parameters
    ----------
    product : dict
        Product descriptor as returned by MAST; uses ``file_suffix`` and
        ``filename`` when present.

    Returns
    -------
    bool
        ``True`` for a ``_rateints`` (or ``_rateint``) product or its preview.
    """
    suffix = (product.get("file_suffix") or "").strip().lstrip("_").lower()
    if suffix in RATEINT_SUFFIXES:
        return True
    stem = product.get("filename", "").rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[-1].lower() in RATEINT_SUFFIXES


def is_fits_product(product: dict) -> bool:
    """Return whether ``product`` is a FITS data file (not a preview or thumbnail).

    MAST lists, alongside each FITS file, preview and thumbnail ``.jpg`` images.
    Those are rarely wanted for a data download, so the search keeps only FITS
    files unless the caller opts into all formats.

    Parameters
    ----------
    product : dict
        Product descriptor as returned by MAST; uses ``filename``.

    Returns
    -------
    bool
        ``True`` when ``filename`` ends in ``.fits`` or ``.fits.gz``.
    """
    name = product.get("filename", "").lower()
    return name.endswith((".fits", ".fits.gz"))


async def fetch_proposal_file_sets(
    http: HTTPSession,
    proposal_id: str,
    product_level: str,
) -> list[dict]:
    """Fetch the file sets for a proposal at a given product level.

    Parameters
    ----------
    http : HTTPSession
        Shared session used for the request.
    proposal_id : str
        Five-digit proposal ID.
    product_level : str
        Product level to query (see :data:`options.PRODUCT_LEVELS`).

    Returns
    -------
    list of dict
        The ``results`` from the MAST search endpoint; each entry describes a
        file set (``fileSetName``, ``access``, ...).
    """
    data = await http.fetch_json(
        MAST_JWST_SEARCH_URL,
        method="POST",
        json_body={
            "conditions": [{"program": proposal_id, "productLevel": product_level}]
        },
    )
    return data["results"]


async def fetch_products_for_fileset(
    http: HTTPSession,
    fileset_name: str,
    product_level: str,
    retry_limit: int = DEFAULT_RETRY_LIMIT,
) -> list[dict]:
    """Fetch products for a single file set, retrying while MAST returns nothing.

    Parameters
    ----------
    http : HTTPSession
        Shared session used for the request.
    fileset_name : str
        Name of the file set to query.
    product_level : str
        Only products whose ``category`` matches this level are returned.
    retry_limit : int, default :data:`options.DEFAULT_RETRY_LIMIT`
        Maximum number of attempts; MAST occasionally returns an empty list on
        the first call, so empty responses are retried (with a one-second pause
        between attempts) up to this many times.

    Returns
    -------
    list of dict
        Products belonging to ``fileset_name`` filtered to ``product_level``.
    """
    products: list[dict] = []
    for _ in range(retry_limit):
        data = await http.fetch_json(
            MAST_JWST_PRODUCT_URL,
            method="GET",
            params={"dataset_ids": fileset_name},
        )
        products = data["products"]
        if products:
            break
        await asyncio.sleep(1)
    return [product for product in products if product["category"] == product_level]


async def fetch_products_for_file_sets(
    http: HTTPSession,
    search_results: Iterable[dict],
    product_level: str,
    retry_limit: int = DEFAULT_RETRY_LIMIT,
    on_fileset_done: Callable[[], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Fetch products for all public file sets, reporting any that came back empty.

    Private file sets (``access == "private"``) are skipped. The remaining file
    sets are queried concurrently over a single shared session.

    Parameters
    ----------
    http : HTTPSession
        Shared session; all per-fileset requests run inside one acquisition.
    search_results : iterable of dict
        File-set descriptors as returned by :func:`fetch_proposal_file_sets`.
    product_level : str
        Product level to filter on.
    retry_limit : int, default :data:`options.DEFAULT_RETRY_LIMIT`
        Forwarded to :func:`fetch_products_for_fileset`.
    on_fileset_done : callable or None, optional
        Invoked with no arguments each time one file set finishes being
        queried, regardless of completion order. Suitable for advancing a
        :class:`rich.progress.Progress` task by one. The callback must not
        block the event loop.

    Returns
    -------
    products : list of dict
        Flattened products across all public file sets.
    missing_filesets : list of str
        Names of public file sets that returned no products.
    """
    public_results = [
        result for result in search_results if result["access"] != "private"
    ]

    async def _fetch_one(result: dict) -> list[dict]:
        products = await fetch_products_for_fileset(
            http, result["fileSetName"], product_level, retry_limit
        )
        if on_fileset_done is not None:
            on_fileset_done()
        return products

    async with http.acquire():
        product_lists = await asyncio.gather(
            *(_fetch_one(result) for result in public_results)
        )

    results: list[dict] = []
    missing_filesets: list[str] = []
    for result, products in zip(public_results, product_lists, strict=True):
        if not products:
            missing_filesets.append(result["fileSetName"])
        results.extend(products)

    return results, missing_filesets
