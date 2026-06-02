"""Filesystem helpers for the fetch CLI: product manifests on disk."""

import json
from pathlib import Path


def load_products(products_file: Path) -> list[dict]:
    """Load a JSON manifest of products from ``products_file``."""
    with open(products_file) as f:
        return json.load(f)


def save_products(products: list[dict], output_file: Path) -> None:
    """Save a JSON manifest of products to ``output_file``, creating parents."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(products, f, indent=4)
