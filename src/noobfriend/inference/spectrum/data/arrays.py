"""Array normalization for spectrum data."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum.types import VALID_WAVE_UNITS, WaveUnit


@dataclass(frozen=True, slots=True)
class _SpectrumArrays:
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    mask_excluded: np.ndarray | None

    @property
    def valid_mask(self) -> np.ndarray:
        valid = (
            np.isfinite(self.wavelength)
            & np.isfinite(self.flux)
            & np.isfinite(self.error)
            & (self.error > 0)
        )
        if self.mask_excluded is not None:
            valid &= ~self.mask_excluded
        valid.setflags(write=False)
        return valid


def _normalize_arrays(
    obs: ArrayLike,
    flux: ArrayLike,
    error: ArrayLike,
    mask_excluded: ArrayLike | None,
) -> _SpectrumArrays:
    wl = _as_1d_float(obs, name="obs")
    fl = _as_1d_float(flux, name="flux")
    er = _as_1d_float(error, name="error")
    if not (wl.shape == fl.shape == er.shape):
        raise ValueError(
            f"obs {wl.shape}, flux {fl.shape}, error {er.shape} must have equal length."
        )
    if wl.size == 0:
        raise ValueError("NoobSpectrum requires at least one pixel.")

    mask = _normalize_mask(mask_excluded, wl)
    arrays = _SpectrumArrays(wavelength=wl, flux=fl, error=er, mask_excluded=mask)
    if not np.any(arrays.valid_mask):
        raise ValueError("NoobSpectrum requires at least one valid unmasked pixel.")
    return arrays


def _as_1d_float(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    array.setflags(write=False)
    return array


def _normalize_mask(
    mask: ArrayLike | None, wavelength: np.ndarray
) -> np.ndarray | None:
    if mask is None:
        return None
    out = np.array(mask, dtype=bool, copy=True)
    if out.shape != wavelength.shape:
        raise ValueError(
            f"mask_excluded shape {out.shape} must match wavelength {wavelength.shape}."
        )
    out.setflags(write=False)
    return out


def _normalize_unit(unit: WaveUnit) -> WaveUnit:
    if unit not in VALID_WAVE_UNITS:
        raise ValueError(
            f"unit {unit!r} must be one of {tuple(sorted(VALID_WAVE_UNITS))}."
        )
    return unit


def _normalize_redshift(z: float | None) -> float | None:
    if z is None:
        return None
    parsed = float(z)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError("z must be strictly positive when provided.")
    return parsed


def _normalize_resolving_power(resolving_power: float | None) -> float | None:
    if resolving_power is None:
        return None
    parsed = float(resolving_power)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError("resolving_power must be positive when provided.")
    return parsed
