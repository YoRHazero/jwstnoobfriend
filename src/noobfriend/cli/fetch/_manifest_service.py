"""Pure helpers for inspecting and filtering a saved products manifest."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from noobfriend.cli.fetch._io import load_products


def load_manifest_terms(products_file: Path) -> list[dict[str, Any]]:
    """Load a products manifest and validate it is a list of JSON objects.

    Parameters
    ----------
    products_file : Path
        Path to the manifest written by the ``search`` command.

    Returns
    -------
    list of dict
        The manifest terms.

    Raises
    ------
    typer.BadParameter
        If the manifest is not a JSON list, or contains non-object entries.
    """
    products = load_products(products_file)
    if not isinstance(products, list):
        raise typer.BadParameter("Manifest must be a JSON list of objects.")

    invalid_indexes = [
        index for index, product in enumerate(products) if not isinstance(product, dict)
    ]
    if invalid_indexes:
        raise typer.BadParameter(
            "Manifest must contain only JSON objects. "
            f"Invalid entries at indexes: {invalid_indexes[:5]}"
        )
    return products


def count_manifest_columns(products: list[dict[str, Any]]) -> Counter[str]:
    """Count how many terms contain each column."""
    counts: Counter[str] = Counter()
    for product in products:
        counts.update(product.keys())
    return counts


def normalize_manifest_value(value: Any) -> str:
    """Normalize a manifest value for display and set-membership checks.

    Values are rendered as canonical JSON when possible so that, for example,
    the integer ``2`` and the parsed filter ``2`` compare equal regardless of
    how they were spelled on the command line.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def parse_filter_value(raw_value: str) -> Any:
    """Parse a filter value as a JSON literal, falling back to the raw string.

    This lets ``--value 2`` match the integer ``2`` while ``--value NIRCAM``
    stays a plain string.
    """
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def deduplicate_columns(columns: list[str]) -> list[str]:
    """Remove duplicate column names while preserving their first-seen order."""
    return list(dict.fromkeys(columns))


def validate_columns(columns: list[str], available_columns: Counter[str]) -> None:
    """Ensure every requested column exists in the manifest.

    Raises
    ------
    typer.BadParameter
        If any column in ``columns`` is absent from ``available_columns``.
    """
    missing_columns = [column for column in columns if column not in available_columns]
    if not missing_columns:
        return

    available_preview = ", ".join(sorted(available_columns)[:10])
    raise typer.BadParameter(
        f"Unknown column(s): {', '.join(missing_columns)}. "
        f"Available columns include: {available_preview}"
    )


def summarize_manifest_column(
    products: list[dict[str, Any]],
    column: str,
) -> Counter[str]:
    """Count normalized values for a single manifest column."""
    return Counter(
        normalize_manifest_value(product[column])
        for product in products
        if column in product
    )


def filter_manifest_terms(
    products: list[dict[str, Any]],
    column: str,
    raw_values: list[str],
) -> list[dict[str, Any]]:
    """Keep only terms whose ``column`` matches any of the provided values.

    Each value in ``raw_values`` is parsed with :func:`parse_filter_value` and
    normalized with :func:`normalize_manifest_value` before comparison.
    """
    kept_values = {
        normalize_manifest_value(parse_filter_value(value)) for value in raw_values
    }
    return [
        product
        for product in products
        if column in product
        and normalize_manifest_value(product[column]) in kept_values
    ]
