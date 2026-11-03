"""Wavelength units, intervals and the velocity scale of the spectrum package.

Wavelengths are plain floats in one of three units, named by string. Line
widths and velocity offsets are always in km/s, related to wavelength through
the speed of light :data:`C_KMS`. Wavelength ranges (masks, fit windows) are
``(lo, hi)`` pairs or sequences of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np

__all__ = [
    "C_KMS",
    "VALID_UNITS",
    "Interval",
    "WaveUnit",
    "check_unit",
    "convert_wavelength",
    "in_intervals",
    "parse_intervals",
]

#: Speed of light in km/s.
C_KMS = 299792.458

type WaveUnit = Literal["angstrom", "nm", "um"]
#: A closed wavelength range ``(lo, hi)``.
type Interval = tuple[float, float]

VALID_UNITS: tuple[str, ...] = ("angstrom", "nm", "um")
_ANGSTROM_PER_UNIT = {"angstrom": 1.0, "nm": 10.0, "um": 1.0e4}


def check_unit(unit: object, *, name: str = "unit") -> WaveUnit:
    """Validate a wavelength unit string.

    Parameters
    ----------
    unit
        Candidate unit.
    name
        Argument name used in the error message.

    Returns
    -------
    str
        The unit, unchanged.

    Raises
    ------
    ValueError
        If ``unit`` is not one of ``"angstrom"``, ``"nm"``, ``"um"``.
    """
    if unit not in _ANGSTROM_PER_UNIT:
        raise ValueError(f"{name} must be one of {VALID_UNITS}; got {unit!r}.")
    return unit  # type: ignore[return-value]


def convert_wavelength(
    value: float, *, from_unit: WaveUnit, to_unit: WaveUnit
) -> float:
    """Convert a wavelength (or wavelength difference) between units.

    Parameters
    ----------
    value
        Wavelength in ``from_unit``.
    from_unit, to_unit
        Source and target units.

    Returns
    -------
    float
        ``value`` expressed in ``to_unit``.
    """
    return float(value) * _ANGSTROM_PER_UNIT[from_unit] / _ANGSTROM_PER_UNIT[to_unit]


def parse_intervals(value: object, *, name: str) -> tuple[Interval, ...]:
    """Normalize ``(lo, hi)`` or a sequence of such pairs.

    Parameters
    ----------
    value
        One pair or a sequence of pairs.
    name
        Argument name used in error messages.

    Returns
    -------
    tuple of (float, float)
        The intervals, each with ``lo < hi``.

    Raises
    ------
    TypeError
        If ``value`` is not a pair or a sequence of pairs of numbers.
    ValueError
        If an interval is not finite and increasing.
    """
    if _is_pair(value):
        pairs = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = list(value)
        if not pairs or not all(_is_pair(pair) for pair in pairs):
            raise TypeError(f"{name} must be (lo, hi) or a sequence of (lo, hi) pairs.")
    else:
        raise TypeError(f"{name} must be (lo, hi) or a sequence of (lo, hi) pairs.")
    intervals = []
    for lo, hi in pairs:  # type: ignore[misc]
        lo, hi = float(lo), float(hi)
        if not (isfinite(lo) and isfinite(hi) and lo < hi):
            raise ValueError(
                f"{name} interval ({lo:g}, {hi:g}) must be finite and increasing."
            )
        intervals.append((lo, hi))
    return tuple(intervals)


def in_intervals(wavelength: np.ndarray, intervals: Sequence[Interval]) -> np.ndarray:
    """Return ``True`` where ``wavelength`` lies in any closed interval."""
    inside = np.zeros(np.shape(wavelength), dtype=bool)
    for lo, hi in intervals:
        inside |= (wavelength >= lo) & (wavelength <= hi)
    return inside


def _is_pair(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in value)
    )
