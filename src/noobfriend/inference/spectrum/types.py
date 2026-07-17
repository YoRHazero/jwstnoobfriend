"""Shared spectrum type aliases."""

from __future__ import annotations

from typing import Literal


type WaveUnit = Literal["angstrom", "nm", "um"]

VALID_WAVE_UNITS = frozenset({"angstrom", "nm", "um"})
_ANGSTROM_PER_UNIT = {
    "angstrom": 1.0,
    "nm": 10.0,
    "um": 1.0e4,
}


def convert_wavelength(
    value: float, *, from_unit: WaveUnit, to_unit: WaveUnit
) -> float:
    """Convert a wavelength between supported units."""
    _check_unit(from_unit)
    _check_unit(to_unit)
    return value * _ANGSTROM_PER_UNIT[from_unit] / _ANGSTROM_PER_UNIT[to_unit]


def _check_unit(unit: WaveUnit) -> None:
    if unit not in VALID_WAVE_UNITS:
        raise ValueError(
            f"unit {unit!r} must be one of {tuple(sorted(VALID_WAVE_UNITS))}."
        )


__all__ = [
    "WaveUnit",
    "convert_wavelength",
]
