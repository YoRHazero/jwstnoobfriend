"""Line contract for spectrum fitting."""

from noobfriend.inference.spectrum.line.line import NoobLine
from noobfriend.inference.spectrum.line.types import (
    ComponentName,
    ContributionName,
    DEFAULT_CENTER_RANGE_KMS,
    DEFAULT_FWHM_RANGES,
    FixedOrBounded,
    ProfileName,
    WaveUnit,
)

__all__ = [
    "ComponentName",
    "ContributionName",
    "DEFAULT_CENTER_RANGE_KMS",
    "DEFAULT_FWHM_RANGES",
    "FixedOrBounded",
    "NoobLine",
    "ProfileName",
    "WaveUnit",
]
