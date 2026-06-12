"""Wavelength unit tokens and conversion (astropy-free).

A closed set of wavelength units (:data:`WaveUnit`) plus a one-line factor table,
so unit handling is a statically-checkable literal and conversion is a single
multiply -- no astropy, and so no ``"A"``-is-ampere ambiguity. Everything
downstream of :meth:`NoobSpectrum.setup` works in the spectrum's own
``wave_unit``; only a line's rest wavelength (given in a possibly different
``WaveUnit``) is converted into that frame, at setup.

This module is internal; :data:`WaveUnit` is re-exported from
:mod:`noobfriend.inference.spectrum`.
"""

from __future__ import annotations

from typing import Literal, get_args

__all__ = ["WAVE_UNITS", "WaveUnit", "check_unit", "convert"]

#: The supported wavelength units: ``"A"`` (Angstrom), ``"nm"``, ``"um"`` (micron).
WaveUnit = Literal["A", "nm", "um"]

#: Angstroms per one of each unit.
_ANGSTROM_PER: dict[str, float] = {"A": 1.0, "nm": 10.0, "um": 1.0e4}

#: The accepted unit tokens (the values of :data:`WaveUnit`).
WAVE_UNITS: tuple[str, ...] = get_args(WaveUnit)


def check_unit(unit: str | None, where: str) -> str:
    """Validate a wavelength unit token.

    Parameters
    ----------
    unit : str or None
        The token to check.
    where : str
        Context for the error message.

    Returns
    -------
    str
        ``unit`` unchanged, if valid.

    Raises
    ------
    ValueError
        If ``unit`` is not one of :data:`WAVE_UNITS`.
    """
    if unit not in _ANGSTROM_PER:
        raise ValueError(f"{where}: wave unit {unit!r} must be one of {WAVE_UNITS}.")
    return unit


def convert(value: float, frm: str, to: str) -> float:
    """Convert a wavelength ``value`` from one :data:`WaveUnit` to another.

    Parameters
    ----------
    value : float
        Wavelength in ``frm`` units.
    frm, to : str
        Source and target :data:`WaveUnit` tokens.

    Returns
    -------
    float
        ``value`` expressed in ``to`` units.

    Raises
    ------
    ValueError
        If ``frm`` or ``to`` is not a valid unit.
    """
    check_unit(frm, "convert (from)")
    check_unit(to, "convert (to)")
    return value * _ANGSTROM_PER[frm] / _ANGSTROM_PER[to]
