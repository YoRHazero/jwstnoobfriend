"""Public line type aliases."""

from __future__ import annotations

from typing import Literal

from noobfriend.inference.spectrum.line.distributions import Prior


type ComponentName = Literal["narrow", "broad"]
type ContributionName = Literal["emission", "absorption"]
type ProfileName = Literal["gaussian", "lorentzian", "exponential"]
#: The original fast path: a fixed value or a ``(lower, upper)`` range.
type FixedOrBounded = float | tuple[float, float]
#: What every line parameter accepts: the fast path, or an explicit prior.
type ParameterSpec = float | tuple[float, float] | Prior

VALID_COMPONENTS = frozenset({"narrow", "broad"})
VALID_CONTRIBUTIONS = frozenset({"emission", "absorption"})
VALID_PROFILES = frozenset({"gaussian", "lorentzian", "exponential"})
DEFAULT_CENTER_RANGE_KMS = (-300.0, 300.0)
#: Bounds a component's FWHM (km/s) defaults to when left free.
DEFAULT_FWHM_RANGES: dict[ComponentName, tuple[float, float]] = {
    "narrow": (10.0, 700.0),
    "broad": (800.0, 5000.0),
}
#: Nominal FWHM (km/s) per component, clipped into DEFAULT_FWHM_RANGES. Fit
#: backends use it wherever a representative width is needed before the width
#: itself is known: optimizer starting points and data-derived flux scales.
NOMINAL_FWHM_KMS: dict[ComponentName, float] = {
    "narrow": 200.0,
    "broad": 1200.0,
}

__all__ = [
    "ComponentName",
    "ContributionName",
    "DEFAULT_CENTER_RANGE_KMS",
    "DEFAULT_FWHM_RANGES",
    "FixedOrBounded",
    "NOMINAL_FWHM_KMS",
    "ParameterSpec",
    "ProfileName",
]
