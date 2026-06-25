"""Emission-line MCMC fitting of 1-D spectra.

Build a :class:`NoobSpectrum` from a 1-D (or collapsed 2-D) spectrum, declare
the lines as :class:`NoobLine` objects -- deriving related lines with
:meth:`NoobLine.derive` to tie their velocity, width, and flux -- then call
:meth:`NoobSpectrum.setup` to compile and inspect the model
(:class:`LineFitSetup`), and :meth:`LineFitSetup.run` to sample it into a
:class:`LineFitResult`. Component shapes (``narrow`` / ``broad`` /
``absorption``) come from :class:`ComponentTemplate`; add custom ones with
:func:`register_template`.

Lines are rest-frame by default; for blind-search features of unknown identity
and redshift, build observed-frame lines with :meth:`NoobLine.observed` (the
spectrum's ``z`` may then be ``None``). For a stubborn fit, hand
:meth:`LineFitSetup.run` an ``init_guess`` of physical start values -- a
sampling start only, which never changes the data-driven priors.
"""

from noobfriend.inference.spectrum._line import NoobLine
from noobfriend.inference.spectrum._result import LineFitResult
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
    "register_template",
]
