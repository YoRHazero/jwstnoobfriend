"""Public line type aliases."""

from __future__ import annotations

from typing import Literal

from noobfriend.inference.spectrum.types import WaveUnit

type ComponentName = Literal["narrow", "broad"]
type ContributionName = Literal["emission", "absorption"]
type ProfileName = Literal["gaussian", "lorentzian", "exponential"]
type FixedOrBounded = float | tuple[float, float]

VALID_COMPONENTS = frozenset({"narrow", "broad"})
VALID_CONTRIBUTIONS = frozenset({"emission", "absorption"})
VALID_PROFILES = frozenset({"gaussian", "lorentzian", "exponential"})
DEFAULT_CENTER_RANGE_KMS = (-300.0, 300.0)
DEFAULT_FWHM_RANGES: dict[ComponentName, tuple[float, float]] = {
    "narrow": (10.0, 700.0),
    "broad": (800.0, 5000.0),
}

__all__ = [
    "ComponentName",
    "ContributionName",
    "DEFAULT_CENTER_RANGE_KMS",
    "DEFAULT_FWHM_RANGES",
    "FixedOrBounded",
    "ProfileName",
    "WaveUnit",
]
