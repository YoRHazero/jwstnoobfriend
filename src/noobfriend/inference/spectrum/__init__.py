"""Spectrum inference tools."""

from noobfriend.inference.spectrum.data import NoobSpectrum, NoobSpectrumSet
from noobfriend.inference.spectrum.line import NoobLine
from noobfriend.inference.spectrum.model import NoobSpectrumModel
from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

__all__ = [
    "NoobFitWorkspace",
    "NoobLine",
    "NoobSpectrum",
    "NoobSpectrumModel",
    "NoobSpectrumSet",
]
