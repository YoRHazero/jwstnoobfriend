"""Emission-line MCMC fitting of 1-D spectra.

Build a :class:`NoobSpectrum` from a 1-D (or collapsed 2-D) spectrum, declare
the lines as :class:`NoobLine` objects -- deriving related lines with
:meth:`NoobLine.derive` to tie their velocity, width, and flux -- then call
:meth:`NoobSpectrum.setup` to compile and inspect the model
(:class:`LineFitSetup`) before sampling. Component shapes (``narrow`` / ``broad``
/ ``absorption``) come from :class:`ComponentTemplate`; add custom ones with
:func:`register_template`.

The PyMC sampling layer and its :class:`LineFitResult` are wired in a later
step; today's surface compiles, validates, and previews the model graph.
"""

from noobfriend.inference.spectrum._line import NoobLine
from noobfriend.inference.spectrum._result import LineFitResult, compare_models
from noobfriend.inference.spectrum._setup import LineFitSetup
from noobfriend.inference.spectrum._spectrum import NoobSpectrum
from noobfriend.inference.spectrum._template import (
    ComponentTemplate,
    register_template,
)
from noobfriend.inference.spectrum._units import WaveUnit

__all__ = [
    "ComponentTemplate",
    "LineFitResult",
    "LineFitSetup",
    "NoobLine",
    "NoobSpectrum",
    "WaveUnit",
    "compare_models",
    "register_template",
]
