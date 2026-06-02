"""Filesystem helpers for the fetch CLI: product manifests and the download dir."""

import json
import logging
import os
from pathlib import Path

from noobfriend.core.env import load_environment

logger = logging.getLogger(__name__)


def env_stage_path() -> str | None:
    """Resolve the configured stage directory from the layered ``.env`` config.

    Reads ``START_STAGE`` and the matching ``STAGE_<STAGE>_PATH`` after loading
    any project / cwd ``.env`` files via :func:`noobfriend.core.env.load_environment`.
    The path is returned verbatim (not created) because it may name a directory
    on a remote host rather than the local machine.

    Returns
    -------
    str or None
        The configured stage path, or ``None`` if ``START_STAGE`` is unset or
        its corresponding path variable is missing.
    """
    load_environment()
    start_stage = os.getenv("START_STAGE")
    if start_stage is None:
        return None

    start_stage_path = os.environ.get(f"STAGE_{start_stage.upper()}_PATH")
    if start_stage_path is None:
        logger.warning(
            "START_STAGE is set to %s, but STAGE_%s_PATH is not set. "
            "Check whether the .env file is written correctly.",
            start_stage,
            start_stage.upper(),
        )
        return None

    return start_stage_path


def load_products(products_file: Path) -> list[dict]:
    """Load a JSON manifest of products from ``products_file``."""
    with open(products_file) as f:
        return json.load(f)


def save_products(products: list[dict], output_file: Path) -> None:
    """Save a JSON manifest of products to ``output_file``, creating parents."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(products, f, indent=4)
