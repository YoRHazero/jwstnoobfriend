"""Spectrum inference tools."""

from noobfriend.inference.spectrum.data import NoobSpectrum, NoobSpectrumSet
from noobfriend.inference.spectrum.line import (
    ComponentName,
    ContributionName,
    DEFAULT_CENTER_RANGE_KMS,
    DEFAULT_FWHM_RANGES,
    FixedOrBounded,
    NoobLine,
    ProfileName,
    WaveUnit,
)
from noobfriend.inference.spectrum.model import NoobSpectrumModel
from noobfriend.inference.spectrum.workspace import (
    ContinuumSharing,
    ContinuumSpec,
    LineHandle,
    NoobFitWorkspace,
)

__all__ = [
    "ComponentName",
    "ContributionName",
    "ContinuumSharing",
    "ContinuumSpec",
    "DEFAULT_CENTER_RANGE_KMS",
    "DEFAULT_FWHM_RANGES",
    "FixedOrBounded",
    "LineHandle",
    "NoobFitWorkspace",
    "NoobLine",
    "NoobSpectrum",
    "NoobSpectrumSet",
    "NoobSpectrumModel",
    "ProfileName",
    "WaveUnit",
]
