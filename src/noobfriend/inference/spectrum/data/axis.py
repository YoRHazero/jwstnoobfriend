"""Spectral axis normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class _SpectrumAxis:
    obs: np.ndarray
    rest: np.ndarray | None


def _normalize_axis(*, obs: ArrayLike | None, rest: ArrayLike | None, z: float | None) -> _SpectrumAxis:
    obs_array = _optional_1d_float(obs, name="obs")
    rest_array = _optional_1d_float(rest, name="rest")

    if obs_array is None and rest_array is None:
        raise ValueError("NoobSpectrum requires obs or rest wavelength.")
    if obs_array is None:
        if z is None:
            raise ValueError("z is required when rest is provided without obs.")
        obs_array = np.array(rest_array * (1.0 + z), dtype=float, copy=True)
        obs_array.setflags(write=False)
    if rest_array is not None and obs_array.shape != rest_array.shape:
        raise ValueError(f"rest shape {rest_array.shape} must match obs {obs_array.shape}.")
    if rest_array is not None and z is not None:
        expected_obs = rest_array * (1.0 + z)
        if not np.allclose(obs_array, expected_obs, rtol=1e-8, atol=1e-8):
            raise ValueError("rest, obs, and z are inconsistent.")

    if not np.all(np.isfinite(obs_array)):
        raise ValueError("obs wavelength must be finite.")
    if not np.all(np.diff(obs_array) > 0):
        raise ValueError("obs wavelength must be strictly increasing.")
    return _SpectrumAxis(obs=obs_array, rest=rest_array)


def _format_center_label(center: float) -> str:
    if isclose(center, round(center), rel_tol=0.0, abs_tol=1e-6):
        return str(int(round(center)))
    return f"{center:.6g}".replace("-", "m").replace(".", "p")


def _optional_1d_float(value: ArrayLike | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    array.setflags(write=False)
    return array
