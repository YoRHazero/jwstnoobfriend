"""Semantic defaults shared by MLE and MCMC spectrum plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


DATA_COLOR = "#1A1A1A"
CONTINUUM_COLOR = "#7F7F7F"
TOTAL_COLOR = "#D62728"
COMPONENT_PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#9467BD",
    "#72B7B2",
    "#B279A2",
    "#9D755D",
    "#D8B70A",
)


def resolve_component_colors(
    component_names: Sequence[str],
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Assign the shared palette in component order with keyed overrides."""
    names = tuple(component_names)
    supplied = {} if overrides is None else dict(overrides)
    unknown = set(supplied).difference(names)
    if unknown:
        valid = ", ".join(names)
        raise KeyError(
            f"unknown component color key(s) {sorted(unknown)!r}; "
            f"valid components: {valid}"
        )
    return {
        name: supplied.get(name, COMPONENT_PALETTE[index % len(COMPONENT_PALETTE)])
        for index, name in enumerate(names)
    }
