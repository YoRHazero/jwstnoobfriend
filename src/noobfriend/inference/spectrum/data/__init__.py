"""Spectrum data preparation."""

from noobfriend.inference.spectrum.data.source2d import Spectrum2DSource
from noobfriend.inference.spectrum.data.spectrum import NoobSpectrum
from noobfriend.inference.spectrum.data.spectrum_set import NoobSpectrumSet

__all__ = [
    "NoobSpectrum",
    "NoobSpectrumSet",
    "Spectrum2DSource",
]
