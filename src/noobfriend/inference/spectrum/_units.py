"""Wavelength units and the velocity scale shared across the spectrum package.

Wavelengths are plain floats in one of three units, named by string. Line
widths and velocity offsets are always in km/s, related to wavelength through
the speed of light :data:`C_KMS`.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["C_KMS", "VALID_UNITS", "WaveUnit", "check_unit", "convert_wavelength"]

#: Speed of light in km/s.
C_KMS = 299792.458

type WaveUnit = Literal["angstrom", "nm", "um"]

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
