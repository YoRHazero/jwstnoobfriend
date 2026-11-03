"""Spectrum data: the head of the fitting chain.

:class:`NoobSpectrum` holds one source in one or more read-only
:class:`SpectrumFrame` objects. It is built by its constructor (one frame),
:meth:`NoobSpectrum.from_2d` (one frame plus its :class:`Source2D`) or
:meth:`NoobSpectrum.join` (several frames), transformed by ``select`` /
``drop`` / ``exclude`` / ``with_noise``, and turned into a fit workspace by
``prepare``.
"""

from noobfriend.inference.spectrum.data._frame import (
    ResolvingPowerTable,
    Source2D,
    SpectrumFrame,
)
from noobfriend.inference.spectrum.data._spectrum import NoobSpectrum

__all__ = ["NoobSpectrum", "ResolvingPowerTable", "Source2D", "SpectrumFrame"]
