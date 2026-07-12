"""Retained two-dimensional source data for a collapsed spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noobfriend.inference.spectrum.data.types import DispersionAxis


@dataclass(frozen=True, slots=True, eq=False)
class Spectrum2DSource:
    """Canonical ``(spatial, wavelength)`` arrays retained by ``from_2d``."""

    flux: np.ndarray
    error: np.ndarray
    collapse_window: tuple[int, int]
    original_dispersion: DispersionAxis

    def __post_init__(self) -> None:
        """Copy source arrays into immutable canonical storage."""
        flux = _readonly_2d(self.flux, name="flux")
        error = _readonly_2d(self.error, name="error")
        if flux.shape != error.shape:
            raise ValueError(
                f"2-D source flux {flux.shape} and error {error.shape} must match."
            )
        lo, hi = self.collapse_window
        if not 0 <= lo < hi <= flux.shape[0]:
            raise ValueError(
                f"collapse_window {self.collapse_window} out of range for "
                f"cross-dispersion extent {flux.shape[0]}."
            )
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "collapse_window", (int(lo), int(hi)))

    @property
    def spatial(self) -> np.ndarray:
        """Default cross-dispersion pixel centers."""
        value = np.arange(self.flux.shape[0], dtype=float)
        value.setflags(write=False)
        return value

    @property
    def spatial_window(self) -> tuple[float, float]:
        """Collapse-aperture edges in the default spatial pixel coordinates."""
        lo, hi = self.collapse_window
        return float(lo) - 0.5, float(hi) - 0.5


def build_spectrum_2d_source(
    flux: np.ndarray,
    error: np.ndarray,
    *,
    collapse_window: tuple[int, int],
    dispersion: DispersionAxis,
) -> Spectrum2DSource:
    """Normalize input orientation into ``(spatial, wavelength)`` order."""
    canonical_flux = np.asarray(flux, dtype=float)
    canonical_error = np.asarray(error, dtype=float)
    if dispersion == "column":
        canonical_flux = canonical_flux.T
        canonical_error = canonical_error.T
    return Spectrum2DSource(
        flux=canonical_flux,
        error=canonical_error,
        collapse_window=(int(collapse_window[0]), int(collapse_window[1])),
        original_dispersion=dispersion,
    )


def _readonly_2d(value: np.ndarray, *, name: str) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    if output.ndim != 2:
        raise ValueError(f"2-D source {name} must be 2-D, got shape {output.shape}.")
    output.setflags(write=False)
    return output
