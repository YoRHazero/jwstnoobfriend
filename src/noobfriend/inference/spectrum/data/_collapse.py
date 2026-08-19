"""2-D spectrum collapse for NoobSpectrum."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum.data.types import DispersionAxis, NoiseSource


def _collapse_to_1d(
    obs: ArrayLike,
    flux: ArrayLike,
    error: ArrayLike,
    *,
    collapse_window: tuple[int, int],
    dispersion: DispersionAxis,
    noise: NoiseSource,
    error_boost: float,
    source_mask: ArrayLike | None,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dispersion not in ("row", "column"):
        raise ValueError(f"dispersion must be 'row' or 'column', got {dispersion!r}.")
    if noise not in ("manual", "continuum", "background"):
        raise ValueError(
            f"noise must be 'manual', 'continuum', or 'background', got {noise!r}."
        )
    if not np.isfinite(error_boost) or error_boost <= 0:
        raise ValueError("error_boost must be positive.")
    if noise != "manual" and error_boost != 1.0:
        raise ValueError("error_boost is only used with noise='manual'.")

    fl = np.asarray(flux, dtype=float)
    er = np.asarray(error, dtype=float)
    wl = np.asarray(obs, dtype=float)
    if fl.ndim != 2 or er.ndim != 2:
        raise ValueError("flux and error must be 2-D.")
    if fl.shape != er.shape:
        raise ValueError(f"flux {fl.shape} and error {er.shape} must match.")

    dispersion_axis = 1 if dispersion == "row" else 0
    cross_axis = 1 - dispersion_axis
    if wl.ndim != 1 or wl.shape[0] != fl.shape[dispersion_axis]:
        raise ValueError(
            f"obs length {wl.shape} must match dispersion extent {fl.shape[dispersion_axis]}."
        )

    lo, hi = int(collapse_window[0]), int(collapse_window[1])
    if not 0 <= lo < hi <= fl.shape[cross_axis]:
        raise ValueError(
            f"collapse_window {collapse_window} out of range for cross-dispersion extent {fl.shape[cross_axis]}."
        )

    from noobfriend.core.specutils import collapse

    flux1d, error1d = collapse(fl, er, window=(lo, hi), dispersion_axis=dispersion_axis)
    boost = _resolve_error_boost(
        noise,
        error_boost,
        fl,
        er,
        wl,
        flux1d,
        error1d,
        collapse_window=(lo, hi),
        dispersion_axis=dispersion_axis,
        source_mask=source_mask,
        max_lag=max_lag,
    )
    return wl, flux1d, error1d * boost


def _resolve_error_boost(
    noise: NoiseSource,
    manual: float,
    flux2d: np.ndarray,
    error2d: np.ndarray,
    wavelength: np.ndarray,
    flux1d: np.ndarray,
    error1d: np.ndarray,
    *,
    collapse_window: tuple[int, int],
    dispersion_axis: int,
    source_mask: ArrayLike | None,
    max_lag: int,
) -> float:
    if noise == "manual":
        return float(manual)

    from noobfriend.core.specutils import continuum_boost, cross_dispersion_boost

    if noise == "continuum":
        return continuum_boost(wavelength, flux1d, error1d)
    return cross_dispersion_boost(
        flux2d,
        error2d,
        window=collapse_window,
        dispersion_axis=dispersion_axis,
        source_mask=source_mask,
        max_lag=max_lag,
    )
