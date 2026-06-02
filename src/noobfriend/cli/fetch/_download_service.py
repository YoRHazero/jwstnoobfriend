"""Async streaming download of MAST products into a target directory."""

import asyncio
import logging
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from noobfriend.cli.fetch._options import MAST_JWST_DOWNLOAD_URL
from noobfriend.core.io import HTTPSession

logger = logging.getLogger(__name__)

ExistMode = Literal["skip", "overwrite"]
MAX_CONCURRENT_DOWNLOADS: int = 5


async def download_single_product(
    http: HTTPSession,
    product: dict,
    output_dir: Path,
    progress: Progress,
    total_task_id: int,
    exist_mode: ExistMode = "skip",
) -> bool:
    """Download one product and report whether a new file was written.

    Parameters
    ----------
    http : HTTPSession
        Shared session; the per-host connection cap throttles concurrency.
    product : dict
        Product descriptor with at least ``filename`` and ``size`` keys.
    output_dir : Path
        Directory the file is written into.
    progress : Progress
        Live progress display; a transient per-file task is created and the
        shared total task is advanced by one when this product is done.
    total_task_id : int
        Task id of the overall progress bar.
    exist_mode : {"skip", "overwrite"}, default "skip"
        ``"skip"`` leaves an existing file in place only when its size matches
        the expected product size; a size mismatch is overwritten. ``"overwrite"``
        always re-downloads.

    Returns
    -------
    bool
        ``True`` if the product was downloaded, ``False`` if it was skipped or
        the download failed.
    """
    filename = product["filename"]
    output_path = output_dir / filename

    if output_path.exists() and exist_mode == "skip":
        if output_path.stat().st_size == product["size"]:
            logger.info("File %s already exists, skipping download.", output_path)
            progress.update(total_task_id, advance=1)
            return False
        logger.warning(
            "File %s exists but size does not match. Will overwrite.", output_path
        )

    task_id = progress.add_task(
        f"Downloading {filename}", total=product["size"] or None
    )
    try:
        await http.download_to_file(
            MAST_JWST_DOWNLOAD_URL,
            output_path,
            method="GET",
            params={"product_name": filename},
            progress_callback=lambda downloaded, total: progress.update(
                task_id, completed=downloaded, total=total or None
            ),
        )
        return True
    except Exception as error:
        logger.error("Failed to download %s: %s", filename, error)
        return False
    finally:
        progress.remove_task(task_id)
        progress.update(total_task_id, advance=1)


async def download_products(
    products: list[dict],
    output_dir: Path,
    progress: Progress,
    exist_mode: ExistMode = "skip",
) -> list[bool]:
    """Download every product concurrently and return per-product statuses.

    Parameters
    ----------
    products : list of dict
        Product descriptors to download.
    output_dir : Path
        Destination directory (must already exist).
    progress : Progress
        Live progress display owned by the caller.
    exist_mode : {"skip", "overwrite"}, default "skip"
        Forwarded to :func:`download_single_product`.

    Returns
    -------
    list of bool
        One status per input product, in order; ``True`` marks a new download.
    """
    http = HTTPSession(max_connections=MAX_CONCURRENT_DOWNLOADS)
    total_task_id = progress.add_task("Total Progress:", total=len(products))
    async with http.acquire():
        statuses = await asyncio.gather(
            *(
                download_single_product(
                    http, product, output_dir, progress, total_task_id, exist_mode
                )
                for product in products
            )
        )
    return list(statuses)
